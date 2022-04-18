import os
import ujson
import torch
import numpy as np
import tqdm

from colbert.utils.utils import lengths2offsets, print_message, dotdict, flatten
from colbert.indexing.loaders import load_doclens
from colbert.indexing.codecs.residual import ResidualCodec
from colbert.search.strided_tensor import StridedTensor


class IndexLoader:
    def __init__(self, index_path, use_gpu=True):
        self.index_path = index_path
        self.use_gpu = use_gpu

        self._load_codec()
        self._load_ivf()

        self._load_doclens()
        self._load_embeddings()

    def _load_codec(self):
        self.codec = ResidualCodec.load(self.index_path)

    def _load_ivf(self):
        if os.path.exists(os.path.join(self.index_path, "ivf.pid.pt")):
            ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.pid.pt"), map_location='cpu')
        else:
            assert os.path.exists(os.path.join(self.index_path, "ivf.pt"))
            ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.pt"), map_location='cpu')
            ivf, ivf_lengths = self._optimize_ivf(ivf, ivf_lengths)

        if False:
            ivf = ivf.tolist()
            ivf = [ivf[offset:endpos] for offset, endpos in lengths2offsets(ivf_lengths)]
        else:
            # ivf, ivf_lengths = ivf.cuda(), torch.LongTensor(ivf_lengths).cuda()  # FIXME: REMOVE THIS LINE!
            ivf = StridedTensor(ivf, ivf_lengths, use_gpu=self.use_gpu)

        self.ivf = ivf

    def _load_doclens(self):
        doclens = []

        for chunk_idx in range(self.num_chunks):
            with open(os.path.join(self.index_path, f'doclens.{chunk_idx}.json')) as f:
                chunk_doclens = ujson.load(f)
                doclens.extend(chunk_doclens)

        self.doclens = torch.tensor(doclens)

    def _load_embeddings(self):
        self.embeddings = ResidualCodec.Embeddings.load_chunks(self.index_path, range(self.num_chunks),
                                                               self.num_embeddings)

    def _optimize_ivf(self, orig_ivf, orig_ivf_lengths):
        print_message("#> Optimizing IVF to store map from centroids to list of pids..")

        print_message("#> Building the emb2pid mapping..")
        all_doclens = load_doclens(self.index_path, flatten=False)

        assert self.num_embeddings == sum(flatten(all_doclens))

        all_doclens = flatten(all_doclens)
        total_num_embeddings = sum(all_doclens)

        emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

        """
        EVENTUALLY: Use two tensors. emb2pid_offsets will have every 256th element. emb2pid_delta will have the delta
                    from the corresponding offset,
        """

        offset_doclens = 0
        for pid, dlength in enumerate(all_doclens):
            emb2pid[offset_doclens: offset_doclens + dlength] = pid
            offset_doclens += dlength

        print_message("len(emb2pid) =", len(emb2pid))

        ivf = emb2pid[orig_ivf]
        unique_pids_per_centroid = []
        ivf_lengths = []

        offset = 0
        for length in tqdm.tqdm(orig_ivf_lengths.tolist()):
            pids = torch.unique(ivf[offset:offset+length])
            unique_pids_per_centroid.append(pids)
            ivf_lengths.append(pids.shape[0])
            offset += length
        ivf = torch.cat(unique_pids_per_centroid)
        ivf_lengths = torch.tensor(ivf_lengths)

        torch.save((ivf, ivf_lengths), os.path.join(self.index_path, f'ivf.pid.pt'))
        print_message(f"#> Saved optimized IVF to {os.path.join(self.index_path, 'ivf.pid.pt')}")
        print_message(f"#> Original IVF at path \"{os.path.join(self.index_path, 'ivf.pt')}\" can now be removed")

        return ivf, ivf_lengths

    @property
    def metadata(self):
        try:
            self._metadata
        except:
            with open(os.path.join(self.index_path, 'metadata.json')) as f:
                self._metadata = ujson.load(f)

        return self._metadata

    @property
    def config(self):
        raise NotImplementedError()  # load from dict at metadata['config']

    @property
    def num_chunks(self):
        # EVENTUALLY: If num_chunks doesn't exist (i.e., old index), fall back to counting doclens.*.json files.
        return self.metadata['num_chunks']

    @property
    def num_embeddings(self):
        # EVENTUALLY: If num_embeddings doesn't exist (i.e., old index), sum the values in doclens.*.json files.
        return self.metadata['num_embeddings']

