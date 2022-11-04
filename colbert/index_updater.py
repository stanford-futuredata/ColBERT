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

class IndexUpdater:
    
    def __init__(self, config, searcher, checkpoint=None):
        self.config = config
        self.searcher = searcher
        if checkpoint:
            self.checkpoint = Checkpoint(checkpoint, config)
            self.encoder = CollectionEncoder(config, self.checkpoint)
        self.index_path = searcher.index
        self._load_disk_ivf()
        
        
    def remove(self, pids, persist_to_disk=False):
        self._remove_pid_from_ivf(pids, persist_to_disk)
        
        if persist_to_disk:
            self._load_metadata()
            for pid in pids:
                self._remove_passage_from_disk(pid)

    
    def add(self, pids, persist_to_disk=False):
        if not self.checkpoint:
            raise ValueError("No checkpoint was provided at IndexUpdater initialization.")
        # TODO: implement the add-passage functionalities
        
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
        
    def _remove_pid_from_ivf(self, pids, persist_to_disk):
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
        
        if persist_to_disk:
            optimized_ivf_path = os.path.join(self.index_path, 'ivf.pid.pt')
            torch.save((self.curr_ivf, self.curr_ivf_lengths), optimized_ivf_path)
            print_message(f"#> Saved optimized IVF to {optimized_ivf_path}")
            
    
    
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
    