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
from colbert.indexing.utils import optimize_ivf

class PassageAppender:
    
    def __init__(self, checkpoint, index_path, config):
        self.checkpoint = Checkpoint(checkpoint, config)
        self.index_path = index_path
        self.encoder = CollectionEncoder(config, self.checkpoint)
        # load avg_residuals, buckets, centroids from index
        self.codec = ResidualCodec.load(index_path)
        self._load_metadata()
        self._load_ivf()
    
    def _load_metadata(self):
        with open(os.path.join(self.index_path, 'metadata.json')) as f:
            self.metadata = ujson.load(f)

    def _load_ivf(self):
        print_message(f"#> Loading IVF...")

        if os.path.exists(os.path.join(self.index_path, "ivf.pid.pt")):
            ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.pid.pt"), map_location='cpu')
        else:
            assert os.path.exists(os.path.join(self.index_path, "ivf.pt"))
            ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.pt"), map_location='cpu')
            ivf, ivf_lengths = optimize_ivf(ivf, ivf_lengths, self.index_path)

        self.ivf = ivf
        self.ivf_lengths = ivf_lengths
        
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
    
    def _build_ivf(self, codes):
        codes = codes.sort()
        ivf, values = codes.indices, codes.values
        partitions, ivf_lengths = values.unique_consecutive(return_counts=True)
#         assert partitions.size(0) == self.metadata['num_partitions']
#         return optimize_ivf(ivf, ivf_lengths, self.config.index_path_)
        print(partitions, ivf_lengths)
        return partitions, ivf_lengths
    
    def _add_pid_to_ivf(self, partitions, pid):
        new_ivf = []
        new_ivf_lengths = []
        old_ivf = self.ivf
        old_ivf_lengths = self.ivf_lengths.tolist()
        
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
        
        optimized_ivf_path = os.path.join(self.index_path, 'ivf.pid.pt')
        torch.save((torch.tensor(new_ivf), torch.tensor(new_ivf_lengths)), optimized_ivf_path)
        print_message(f"#> Saved optimized IVF to {optimized_ivf_path}")
        
    def append_passage(self, passage):
        if passage[-4:] != '.csv' or passage[-4:] != '.json':
            passage = [passage]
            passage = Collection(data=passage)
        else:
            raise NotImplementedError()
        # encode passage into embs, doclens
        embs, doclens = self.encoder.encode_passages(passage)
        
        # transform embs into codes and residuals
        compressed_embs = self.codec.compress(embs)
        
        # get metadata and pid
        num_chunks = self.metadata['num_chunks']
        avg_emb_per_chunk = self.metadata['num_embeddings'] / num_chunks
        last_chunk_metadata = self._load_chunk_metadata(num_chunks - 1)
        pid = last_chunk_metadata['passage_offset'] + last_chunk_metadata['num_passages']
        print(pid)
        
        # update ivf
        partitions, _ = self._build_ivf(compressed_embs.codes)
        self._add_pid_to_ivf(partitions, pid)
        
        # write new embs to index
#         if last_chunk_metadata['num_embeddings'] > avg_emb_per_chunk:
#             # create and write to new chunk
#             pass
#         else:
        # append to current last chunk
        curr_embs = ResidualEmbeddings.load(self.index_path, num_chunks - 1)
        curr_embs.codes = torch.cat((curr_embs.codes, compressed_embs.codes))
        curr_embs.residuals = torch.cat((curr_embs.residuals, compressed_embs.residuals))
        path_prefix = os.path.join(self.index_path, f'{num_chunks - 1}')
        curr_embs.save(path_prefix)
        
        # update doclen
        curr_doclens = self._load_chunk_doclens(num_chunks - 1).tolist()
        curr_doclens.extend(doclens)
        doclens_path = os.path.join(self.index_path, f'doclens.{num_chunks - 1}.json')
        with open(doclens_path, 'w') as output_doclens:
            ujson.dump(curr_doclens, output_doclens)
            
        # update chunk metadata
        chunk_metadata = self._load_chunk_metadata(num_chunks - 1)
        chunk_metadata['num_passages'] += 1
        chunk_metadata['num_embeddings'] += doclens[0]
        chunk_metadata_path = os.path.join(self.index_path, f'{num_chunks - 1}.metadata.json')
        with open(chunk_metadata_path, 'w') as output_chunk_metadata:
            ujson.dump(chunk_metadata, output_chunk_metadata)
        
        # update index metadata
        self.metadata['num_embeddings'] += doclens[0]
        metadata_path = os.path.join(self.index_path, 'metadata.json')
        with open(metadata_path, 'w') as output_metadata:
            ujson.dump(self.metadata, output_metadata)
    
        return pid
    
class PassageRemover:

    def __init__(self, index_path):
        self.index_path = index_path
        self._load_ivf()
        self._load_metadata()
    
    def _load_metadata(self):
        with open(os.path.join(self.index_path, 'metadata.json')) as f:
            self.metadata = ujson.load(f)

    def _load_ivf(self):
        print_message(f"#> Loading IVF...")

        if os.path.exists(os.path.join(self.index_path, "ivf.pid.pt")):
            ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.pid.pt"), map_location='cpu')
        else:
            assert os.path.exists(os.path.join(self.index_path, "ivf.pt"))
            ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.pt"), map_location='cpu')
            ivf, ivf_lengths = optimize_ivf(ivf, ivf_lengths, self.index_path)

#         if False:
#             ivf = ivf.tolist()
#             ivf = [ivf[offset:endpos] for offset, endpos in lengths2offsets(ivf_lengths)]
#         else:
#             # ivf, ivf_lengths = ivf.cuda(), torch.LongTensor(ivf_lengths).cuda()  # FIXME: REMOVE THIS LINE!
#             ivf = StridedTensor(ivf, ivf_lengths, use_gpu=False)

        self.ivf = ivf
        self.ivf_lengths = ivf_lengths
        
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
    
    def _remove_pid_from_ivf(self, pid):
        new_ivf = []
        new_ivf_lengths = []
        runner = 0
        for l in self.ivf_lengths.tolist():
            num_removed = 0
            for i in range(runner, runner + l):
                if self.ivf[i] != pid:
                    new_ivf.append(self.ivf[i])
                else:
                    num_removed += 1
            runner += l
            new_ivf_lengths.append(l - num_removed)
            
        assert runner == len(self.ivf.tolist())
        assert sum(new_ivf_lengths) == len(new_ivf)
        
        optimized_ivf_path = os.path.join(self.index_path, 'ivf.pid.pt')
        torch.save((torch.tensor(new_ivf), torch.tensor(new_ivf_lengths)), optimized_ivf_path)
        print_message(f"#> Saved optimized IVF to {optimized_ivf_path}")
        
    def remove_passage(self, pid):
        chunk_idx = self._get_chunk_idx(pid)
        
        chunk_metadata = self._load_chunk_metadata(chunk_idx)
        i = pid - chunk_metadata['passage_offset']
        doclens = self._load_chunk_doclens(chunk_idx)
        codes, residuals = self._load_chunk_codes(chunk_idx), self._load_chunk_residuals(chunk_idx)
        print(doclens.size())
        print(self.metadata)
        
        # remove pid from inv.pid.pt
        # this step alone should prevent the passage from being returned as result
        self._remove_pid_from_ivf(pid)
        
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
