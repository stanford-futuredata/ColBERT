import os
import ujson
import torch
import numpy as np
import tqdm

from colbert.search.index_loader import IndexLoader
from colbert.indexing.index_saver import IndexSaver
from colbert.indexing.collection_encoder import CollectionEncoder

from colbert.utils.utils import lengths2offsets, print_message, dotdict, flatten
from colbert.indexing.codecs.residual import ResidualCodec
from colbert.indexing.utils import optimize_ivf
from colbert.search.strided_tensor import StridedTensor
from colbert.modeling.checkpoint import Checkpoint
from colbert.utils.utils import print_message, batch
from colbert.data import Collection
from colbert.indexing.codecs.residual_embeddings import ResidualEmbeddings
from colbert.indexing.codecs.residual_embeddings_strided import ResidualEmbeddingsStrided
from colbert.indexing.utils import optimize_ivf

DEFAULT_CHUNKSIZE = 25000

class IndexUpdater:
    
    def __init__(self, config, searcher, checkpoint=None):
        self.config = config
        self.searcher = searcher
        self.has_checkpoint = False
        if checkpoint:
            self.has_checkpoint = True
            self.checkpoint = Checkpoint(checkpoint, config)
            self.encoder = CollectionEncoder(config, self.checkpoint)
        self.index_path = searcher.index
        self._load_disk_ivf()
        self.removed_pids = []
        self.first_new_emb = sum(self.searcher.ranker.doclens.tolist())
        self.first_new_pid = len(self.searcher.ranker.doclens.tolist())
        print("IndexUpdater CONSTRUCTOR")
        print(self.first_new_emb, self.first_new_pid)
        
    def remove(self, pids):
        self._remove_pid_from_ivf(pids)
        self.removed_pids.extend(pids)
        
    def add(self, passages):
        if not self.has_checkpoint:
            raise ValueError("No checkpoint was provided at IndexUpdater initialization.")
            
        # find first pid to be added
        start_pid = len(self.searcher.ranker.doclens.tolist())
        curr_pid = start_pid
        
        # extend doclens and embs of self.searcher.ranker
        embs, doclens = self.encoder.encode_passages(passages)
        compressed_embs = self.searcher.ranker.codec.compress(embs)
        
        # update searcher
        self.searcher.ranker.embeddings.codes = torch.cat((self.searcher.ranker.embeddings.codes[:-512], compressed_embs.codes, self.searcher.ranker.embeddings.codes[-512:]))
        self.searcher.ranker.embeddings.residuals = torch.cat((self.searcher.ranker.embeddings.residuals[:-512], compressed_embs.residuals, self.searcher.ranker.embeddings.residuals[-512:]), dim=0)
        print(self.searcher.ranker.doclens)
        self.searcher.ranker.doclens = torch.cat((self.searcher.ranker.doclens, torch.tensor(doclens)))
        print(self.searcher.ranker.doclens)
               
        # build partitions for each pid and update updater's current ivf
        start = 0
        for doclen in doclens:
            
            end = start + doclen
            codes = compressed_embs.codes[start:end]
            partitions, _ = self._build_passage_partitions(codes)
            self._add_pid_to_ivf(partitions, curr_pid)
            
            start = end
            curr_pid += 1
            
        assert start == sum(doclens)
        
        # update new ivf in searcher
        new_ivf_tensor = StridedTensor(self.curr_ivf, self.curr_ivf_lengths, use_gpu=False)
        assert new_ivf_tensor != self.searcher.ranker.ivf
        self.searcher.ranker.ivf = new_ivf_tensor

        # rebuild StridedTensor within searcher
        self.searcher.ranker.embeddings_strided = ResidualEmbeddingsStrided(self.searcher.ranker.codec, self.searcher.ranker.embeddings, self.searcher.ranker.doclens)
        
        print(f"Added {len(passages)} passages from pid {start_pid}.")
        return [i for i in range(start_pid, start_pid + len(passages))]

    
    def _build_passage_partitions(self, codes):
        codes = codes.sort()
        ivf, values = codes.indices, codes.values
        partitions, ivf_lengths = values.unique_consecutive(return_counts=True)
        return partitions, ivf_lengths
    
    def _add_pid_to_ivf(self, partitions, pid):
        new_ivf = []
        new_ivf_lengths = []
        old_ivf = self.curr_ivf.tolist()
        old_ivf_lengths = self.curr_ivf_lengths.tolist()
        
        partitions_runner = 0
        ivf_runner = 0
        for i in range(len(old_ivf_lengths)):
            # first copy partition pids to new ivf
            new_ivf.extend(old_ivf[ivf_runner : ivf_runner + old_ivf_lengths[i]])
            new_ivf_lengths.append(old_ivf_lengths[i])
            ivf_runner += old_ivf_lengths[i]
            
            # add pid if i in partitions
            if partitions_runner < len(partitions) and i == partitions[partitions_runner]:
                new_ivf.append(pid)
                new_ivf_lengths[-1] += 1
                partitions_runner += 1
            
        assert ivf_runner == len(old_ivf)
        assert sum(new_ivf_lengths) == len(new_ivf)
        
        self.curr_ivf = torch.tensor(new_ivf)
        self.curr_ivf_lengths = torch.tensor(new_ivf_lengths)
        
    # NOTE: for now we're not updating metadata['avg_doclen']
    def persist_to_disk(self):
        
        # propagate all removed passages to disk
        self._load_metadata()
        for pid in self.removed_pids:
            self._remove_passage_from_disk(pid)
            
        # propagate all added passages to disk
        # Rationale: keep record of all added passages in IndexUpdater,
        # divide passages into chunks and create / write chunks here
        
        self._load_metadata() # reload after removal
        
        # calculate avg # passages per chunk
        curr_num_chunks = self.metadata['num_chunks']
        last_chunk_metadata = self._load_chunk_metadata(curr_num_chunks - 1)
        if curr_num_chunks == 1:
            avg_chunksize = DEFAULT_CHUNKSIZE
        else:
            avg_chunksize = last_chunk_metadata['passage_offset'] / (curr_num_chunks - 1)
        print(avg_chunksize)
        # calculate space left in last chunk
        last_chunk_capacity = max(0, avg_chunksize - last_chunk_metadata['num_passages'])
        print(last_chunk_capacity)
        # divide passages into chunks -> [last chunk], [new chunk 1], ... record: num_new_chunks
        pid_start = self.first_new_pid
        emb_start = self.first_new_emb
        pid_last = len(self.searcher.ranker.doclens.tolist())
        emb_last = emb_start + sum(self.searcher.ranker.doclens.tolist()[pid_start:])
        
        # first populate last chunk
        if last_chunk_capacity > 0:
            pid_end = min(pid_last, pid_start + last_chunk_capacity)
            emb_end = emb_start + sum(self.searcher.ranker.doclens.tolist()[pid_start: pid_end])
            print(f"last chunk: {emb_start} to {emb_end}")
            # TODO: write to last chunk
            self._write_to_last_chunk(pid_start, pid_end, emb_start, emb_end)
            pid_start = pid_end
            emb_start = emb_end
        
        while pid_start < pid_last:
            pid_end = min(pid_last, pid_start + avg_chunksize)
            emb_end = emb_start + sum(self.searcher.ranker.doclens.tolist()[pid_start: pid_end])
            # TODO: write new chunk with id = curr_num_chunks
            
            # create chunk metadata
            chunk_metadata = {'passage_offset': pid_start, 'num_passages': pid_end - pid_start, 'embedding_ofset':emb_start , 'num_embeddings': emb_end - emb_start}
            chunk_metadata_path = os.path.join(self.index_path, f'{curr_num_chunks}.metadata.json')
            with open(chunk_metadata_path, 'w') as output_chunk_metadata:
                ujson.dump(chunk_metadata, output_chunk_metadata)
            print(curr_num_chunks, chunk_metadata)
            
            curr_num_chunks += 1
            pid_start = pid_end
            emb_start = emb_end
        
        assert pid_start == pid_last
        assert emb_start == emb_last
        
        # update metadata
        print("updating metadata for added passages...")
        self.metadata['num_chunks'] = curr_num_chunks
        self.metadata['num_embeddings'] = sum(self.searcher.ranker.doclens.tolist())
        metadata_path = os.path.join(self.index_path, 'metadata.json')
        with open(metadata_path, 'w') as output_metadata:
            ujson.dump(self.metadata, output_metadata)
        
        # save current ivf to disk
        optimized_ivf_path = os.path.join(self.index_path, 'ivf.pid.pt')
        
#         import pdb
#         pdb.set_trace()
        
        torch.save((self.curr_ivf, self.curr_ivf_lengths), optimized_ivf_path)
        print_message(f"#> Saved optimized IVF to {optimized_ivf_path}")
    
    def _write_to_last_chunk(self, pid_start, pid_end, emb_start, emb_end):
        
        print(f"writing {pid_end - pid_start} passages to the last chunk...")
        num_chunks = self.metadata['num_chunks']
        # append to current last chunk
        curr_embs = ResidualEmbeddings.load(self.index_path, num_chunks - 1)
        
#         curr_embs.codes = torch.cat((curr_embs.codes[:-512], self.searcher.ranker.embeddings.codes[emb_start:emb_end], curr_embs.codes[-512:]))
#         curr_embs.residuals = torch.cat((curr_embs.residuals[:-512], self.searcher.ranker.embeddings.residuals[emb_start:emb_end], curr_embs.residuals[-512:]))

        curr_embs.codes = torch.cat((curr_embs.codes, self.searcher.ranker.embeddings.codes[emb_start:emb_end]))
        curr_embs.residuals = torch.cat((curr_embs.residuals, self.searcher.ranker.embeddings.residuals[emb_start:emb_end]))
        path_prefix = os.path.join(self.index_path, f'{num_chunks - 1}')
        curr_embs.save(path_prefix)
        
        # update doclen
        curr_doclens = self._load_chunk_doclens(num_chunks - 1).tolist()
        curr_doclens.extend(self.searcher.ranker.doclens.tolist()[pid_start:pid_end])
        doclens_path = os.path.join(self.index_path, f'doclens.{num_chunks - 1}.json')
        with open(doclens_path, 'w') as output_doclens:
            ujson.dump(curr_doclens, output_doclens)
        
        # update chunk metadata
        print("updating metadata of current last chunk...")
        chunk_metadata = self._load_chunk_metadata(num_chunks - 1)
        chunk_metadata['num_passages'] += pid_end - pid_start
        chunk_metadata['num_embeddings'] += emb_end - emb_start
        chunk_metadata_path = os.path.join(self.index_path, f'{num_chunks - 1}.metadata.json')
        with open(chunk_metadata_path, 'w') as output_chunk_metadata:
            ujson.dump(chunk_metadata, output_chunk_metadata)
            
    def _write_to_new_chunk(self, chunk_idx, pid_start, pid_end, emb_start, emb_end):
        pass
            
    def _load_disk_ivf(self):
        print_message(f"#> Loading IVF...")

        if os.path.exists(os.path.join(self.index_path, "ivf.pid.pt")):
            ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.pid.pt"), map_location='cpu')
        else:
            assert os.path.exists(os.path.join(self.index_path, "ivf.pt"))
            ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.pt"), map_location='cpu')
            ivf, ivf_lengths = optimize_ivf(ivf, ivf_lengths, self.index_path)

        self.curr_ivf = ivf
        self.curr_ivf_lengths = ivf_lengths
        
    def _remove_pid_from_ivf(self, pids):
        new_ivf = []
        new_ivf_lengths = []
        runner = 0
        for length in self.curr_ivf_lengths.tolist():
            num_removed = 0
            for i in range(runner, runner + length):
                if self.curr_ivf[i] not in pids:
                    new_ivf.append(self.curr_ivf[i])
                else:
                    num_removed += 1
            runner += length
            new_ivf_lengths.append(length - num_removed)
            
        assert runner == len(self.curr_ivf.tolist())
        assert sum(new_ivf_lengths) == len(new_ivf)
        
        new_ivf = torch.tensor(new_ivf)
        new_ivf_lengths = torch.tensor(new_ivf_lengths)
        
        new_ivf_tensor = StridedTensor(new_ivf, new_ivf_lengths, use_gpu=False)
        assert new_ivf_tensor != self.searcher.ranker.ivf
        self.searcher.ranker.ivf = new_ivf_tensor
        
        self.curr_ivf = new_ivf
        self.curr_ivf_lengths = new_ivf_lengths
    
    def _load_metadata(self):
        with open(os.path.join(self.index_path, 'metadata.json')) as f:
            self.metadata = ujson.load(f)
            
    def _load_chunk_doclens(self, chunk_idx):
        doclens = []

        print_message("#> Loading doclens...")

        with open(os.path.join(self.index_path, f'doclens.{chunk_idx}.json')) as f:
            chunk_doclens = ujson.load(f)
            doclens.extend(chunk_doclens)

        doclens = torch.tensor(doclens)
        return doclens
    
    def _load_chunk_codes(self, chunk_idx):
        codes_path = os.path.join(self.index_path, f'{chunk_idx}.codes.pt')
        return torch.load(codes_path, map_location='cpu')

    def _load_chunk_residuals(self, chunk_idx):
        residuals_path = os.path.join(self.index_path, f'{chunk_idx}.residuals.pt')
        return torch.load(residuals_path, map_location='cpu')
    
    def _load_chunk_metadata(self, chunk_idx):
        with open(os.path.join(self.index_path, f'{chunk_idx}.metadata.json')) as f:
            chunk_metadata = ujson.load(f)
        return chunk_metadata
    
    def _get_chunk_idx(self, pid):
        for i in range(self.metadata['num_chunks']):
            chunk_metadata = self._load_chunk_metadata(i)
            if chunk_metadata['passage_offset'] <= pid and chunk_metadata['passage_offset'] + chunk_metadata['num_passages'] > pid:
                return i
        raise ValueError('Passage ID out of range')
            
    def _remove_passage_from_disk(self, pid):
        chunk_idx = self._get_chunk_idx(pid)
        
        chunk_metadata = self._load_chunk_metadata(chunk_idx)
        i = pid - chunk_metadata['passage_offset']
        doclens = self._load_chunk_doclens(chunk_idx)
        codes, residuals = self._load_chunk_codes(chunk_idx), self._load_chunk_residuals(chunk_idx)
        
        # remove embeddings from codes and residuals
        start = sum(doclens[:i])
        end = start + doclens[i]
        codes = torch.cat((codes[:start], codes[end:]))
        residuals = torch.cat((residuals[:start], residuals[end:]))
        
        codes_path = os.path.join(self.index_path, f'{chunk_idx}.codes.pt')
        residuals_path = os.path.join(self.index_path, f'{chunk_idx}.residuals.pt')

        torch.save(codes, codes_path)
        torch.save(residuals, residuals_path)
        
        # change doclen for passage to 0
        doclens = doclens.tolist()
        doclen_to_remove = doclens[i]
        doclens[i] = 0
        doclens_path = os.path.join(self.index_path, f'doclens.{chunk_idx}.json')
        with open(doclens_path, 'w') as output_doclens:
            ujson.dump(doclens, output_doclens)
        
        # modify chunk_metadata['num_embeddings'] for chunk_idx
        chunk_metadata['num_embeddings'] -= doclen_to_remove
        chunk_metadata_path = os.path.join(self.index_path, f'{chunk_idx}.metadata.json')
        with open(chunk_metadata_path, 'w') as output_chunk_metadata:
            ujson.dump(chunk_metadata, output_chunk_metadata)
        
        # modify chunk_metadata['embedding_offset'] for all later chunks (minus num_embs_removed)
        for idx in range(chunk_idx + 1, self.metadata['num_chunks']):
            metadata = self._load_chunk_metadata(idx)
            metadata['embedding_offset'] -= doclen_to_remove
            metadata_path = os.path.join(self.index_path, f'{idx}.metadata.json')
            with open(metadata_path, 'w') as output_chunk_metadata:
                ujson.dump(metadata, output_chunk_metadata)
            
        # modify num_embeddings in overall metadata (minus num_embs_removed)
        self.metadata['num_embeddings'] -= doclen_to_remove
        metadata_path = os.path.join(self.index_path, 'metadata.json')
        with open(metadata_path, 'w') as output_metadata:
            ujson.dump(self.metadata, output_metadata)
    

    
#     def add_passage(self, passage):
#         if not self.has_checkpoint:
#             raise ValueError("No checkpoint was provided at IndexUpdater initialization.")
#         start_pid = len(self.searcher.ranker.doclens.tolist())
        
#         # extend doclens and embs of self.searcher.ranker
#         embs, doclens = self.encoder.encode_passages(passage)
#         compressed_embs = self.searcher.ranker.codec.compress(embs)
#         print("Compressing codes...")
#         print(compressed_embs.codes)
        
#         # !!!!!!!!!!!!!!!!
#         # TODO: WRITE COMMENT ABOUT WHAT HAPPENED W THIS 512
#         self.searcher.ranker.embeddings.codes = torch.cat((self.searcher.ranker.embeddings.codes[:-512], compressed_embs.codes, self.searcher.ranker.embeddings.codes[-512:]))
#         self.searcher.ranker.embeddings.residuals = torch.cat((self.searcher.ranker.embeddings.residuals[:-512], compressed_embs.residuals, self.searcher.ranker.embeddings.residuals[-512:]), dim=0)
#         print(self.searcher.ranker.doclens)
#         self.searcher.ranker.doclens = torch.cat((self.searcher.ranker.doclens, torch.tensor(doclens)))
#         print(self.searcher.ranker.doclens)
        
#         # update ivf of index updater
#         partitions, _ = self._build_passage_partitions(compressed_embs.codes)
#         self._add_pid_to_ivf(partitions, start_pid)
        
#         return start_pid



        
#     def add(self, passages):
#         pids = []
#         for passage in passages:
#             pid = self.add_passage([passage])
#             pids.append(pid)
            
#         # update new ivf in searcher
#         new_ivf_tensor = StridedTensor(self.curr_ivf, self.curr_ivf_lengths, use_gpu=False)
#         assert new_ivf_tensor != self.searcher.ranker.ivf
#         self.searcher.ranker.ivf = new_ivf_tensor
        
#         self.searcher.ranker.embeddings_strided = ResidualEmbeddingsStrided(self.searcher.ranker.codec, self.searcher.ranker.embeddings, self.searcher.ranker.doclens)
#         return pids