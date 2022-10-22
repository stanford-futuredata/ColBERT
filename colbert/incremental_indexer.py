import os
import ujson
import torch
import numpy as np
import tqdm

from colbert.search.index_loader import IndexLoader
from colbert.indexing.index_saver import IndexSaver

from colbert.utils.utils import lengths2offsets, print_message, dotdict, flatten
from colbert.indexing.codecs.residual import ResidualCodec
from colbert.indexing.utils import optimize_ivf
from colbert.search.strided_tensor import StridedTensor

class IncrementalIndexer:
        
#     def remove_passage(self, pid):
#         pass
#         # Loading Approach 1: Naively load everything then use overall offsets
#         # Calculate embedding (offset, offset+doclen) from index.doclens and pid
#         # Remove data in above range from index.embeddings
#         # Modify doclen & overall metadata
#         # Modify ivf: remove pid from all centroids
#         # ??? Individual CHUNK metadata.json never loaded in IndexLoader -- do we use this info ever?
        
#         # Storing

# add(passage:str, title: optional str) -> pid: int

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

    def _load_chunk_embeddings(self, chunk_idx):
        embeddings = ResidualCodec.Embeddings.load(self.index_path, chunk_idx)
        return embeddings
    
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
        print(chunk_idx)
        
        chunk_metadata = self._load_chunk_metadata(chunk_idx)
        doclens, embs = self._load_chunk_doclens(chunk_idx), self._load_chunk_embeddings(chunk_idx)
        print(doclens.size())
        print(self.metadata)
        
        # remove embeddings from codes and residuals
        # change doclen for passage to 0
        # modify chunk_metadata['num_embeddings'] and ['embedding_offset'] (minus num_embs_removed)
        # modify num_embeddings in overall metadata (minus num_embs_removed)
        
        # remove pid from inv.pid.pt
        # this step alone should prevent the passage from being returned as result
        self._remove_pid_from_ivf(pid)