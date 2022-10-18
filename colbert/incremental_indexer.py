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

        if False:
            ivf = ivf.tolist()
            ivf = [ivf[offset:endpos] for offset, endpos in lengths2offsets(ivf_lengths)]
        else:
            # ivf, ivf_lengths = ivf.cuda(), torch.LongTensor(ivf_lengths).cuda()  # FIXME: REMOVE THIS LINE!
            ivf = StridedTensor(ivf, ivf_lengths, use_gpu=False)

        self.ivf = ivf
        
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
        # TODO: implement this
        return 0
        
    def remove_passage(self, pid):
        chunk_idx = self._get_chunk_idx(pid)
        print(chunk_idx)
        
        chunk_metadata = self._load_chunk_metadata(chunk_idx)
        doclens, embs = self._load_chunk_doclens(chunk_idx), self._load_chunk_embeddings(chunk_idx)
        print(doclens.size())
        
        #