import torch

from colbert.utils.utils import flatten, print_message

from colbert.indexing.loaders import load_doclens
from colbert.indexing.codecs.residual_embeddings_strided import ResidualEmbeddingsStrided
from colbert.indexing.codecs import residual_embeddings

from colbert.search.strided_tensor import StridedTensor
from colbert.search.candidate_generation import CandidateGeneration

from .index_loader import IndexLoader
from colbert.modeling.colbert import colbert_score, colbert_score_packed, colbert_score_reduce

from math import ceil

import os
import pathlib
from torch.utils.cpp_extension import load

filter_pids_cpp = load(
    name="filter_pids_cpp",
    sources=[
        os.path.join(
            pathlib.Path(__file__).parent.resolve(), "filter_pids.cpp"
        ),
    ],
    extra_cflags=["-O3"],
)

decompress_residuals_cpp = load(
    name="decompress_residuals_cpp",
    sources=[
        os.path.join(
            pathlib.Path(__file__).parent.resolve(), "decompress_residuals.cpp"
        ),
    ],
    extra_cflags=["-O3"],
)

class IndexScorer(IndexLoader, CandidateGeneration):
    def __init__(self, index_path, use_gpu=True, mmap_index=False):
        super().__init__(index_path=index_path, use_gpu=use_gpu, mmap_index=mmap_index)

        if not mmap_index:
            self.embeddings_strided = ResidualEmbeddingsStrided(self.codec, self.embeddings, self.doclens)
        else:
            self.embeddings_strided = None

    def lookup_eids(self, embedding_ids, codes=None, out_device='cuda'):
        return self.embeddings_strided.lookup_eids(embedding_ids, codes=codes, out_device=out_device)

    def lookup_pids(self, passage_ids, out_device='cuda', return_mask=False):
        if self.embeddings_strided is None:
            codes_packed, codes_lengths = self.embeddings.lookup_codes(passage_ids)
            residuals_packed, _ = self.embeddings.lookup_pids(passage_ids)
            embeddings_packed = self.codec.decompress(residual_embeddings.ResidualEmbeddings(codes_packed, residuals_packed))
            return embeddings_packed, codes_lengths 
        return self.embeddings_strided.lookup_pids(passage_ids, out_device)

    def lookup_codes(self, passage_ids, out_device='cuda'):
        if self.embeddings_strided is None:
           return self.embeddings.lookup_codes(passage_ids)
        return self.embeddings_strided.lookup_codes(passage_ids)
        
    def retrieve(self, config, Q):
        Q = Q[:, :config.query_maxlen]   # NOTE: Candidate generation uses only the query tokens
        embedding_ids, centroid_scores = self.generate_candidates(config, Q)

        return embedding_ids, centroid_scores

    def embedding_ids_to_pids(self, embedding_ids):
        all_pids = torch.unique(self.emb2pid[embedding_ids.long()].cuda(), sorted=False)
        return all_pids

    def rank(self, config, Q):
        with torch.inference_mode():
            pids, centroid_scores = self.retrieve(config, Q)
            scores, pids = self.score_pids(config, Q, pids, centroid_scores)

            scores_sorter = scores.sort(descending=True)
            pids, scores = pids[scores_sorter.indices].tolist(), scores_sorter.values.tolist()

            return pids, scores

    def score_pids(self, config, Q, pids, centroid_scores):
        """
            Always supply a flat list or tensor for `pids`.

            Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
            If Q.size(0) is 1, the matrix will be compared with all passages.
            Otherwise, each query matrix will be compared against the *aligned* passage.
        """

        # TODO: Remove batching?
        batch_size = 2 ** 20

        if self.use_gpu:
            centroid_scores = centroid_scores.cuda()

        idx = centroid_scores.max(-1).values >= config.centroid_score_threshold

        if self.use_gpu:
            approx_scores = []

            # Filter docs using pruned centroid scores
            for i in range(0, ceil(len(pids) / batch_size)):
                pids_ = pids[i * batch_size : (i+1) * batch_size]
                codes_packed, codes_lengths = self.lookup_codes(pids_)
                idx_ = idx[codes_packed.long()]
                pruned_codes_strided = StridedTensor(idx_, codes_lengths, use_gpu=self.use_gpu)
                pruned_codes_padded, pruned_codes_mask = pruned_codes_strided.as_padded_tensor()
                pruned_codes_lengths = (pruned_codes_padded * pruned_codes_mask).sum(dim=1)
                codes_packed_ = codes_packed[idx_]
                approx_scores_ = centroid_scores[codes_packed_.long()]
                if approx_scores_.shape[0] == 0:
                    approx_scores.append(torch.zeros((len(pids_),), dtype=approx_scores_.dtype))
                    continue
                approx_scores_strided = StridedTensor(approx_scores_, pruned_codes_lengths, use_gpu=self.use_gpu)
                approx_scores_padded, approx_scores_mask = approx_scores_strided.as_padded_tensor()
                approx_scores_ = colbert_score_reduce(approx_scores_padded, approx_scores_mask, config)
                approx_scores.append(approx_scores_)
            approx_scores = torch.cat(approx_scores, dim=0)
            if config.ndocs < len(approx_scores):
                pids = pids[torch.topk(approx_scores, k=config.ndocs).indices]

            # Filter docs using full centroid scores
            codes_packed, codes_lengths = self.lookup_codes(pids)
            approx_scores = centroid_scores[codes_packed.long()]
            approx_scores_strided = StridedTensor(approx_scores, codes_lengths, use_gpu=self.use_gpu)
            approx_scores_padded, approx_scores_mask = approx_scores_strided.as_padded_tensor()
            approx_scores = colbert_score_reduce(approx_scores_padded, approx_scores_mask, config)
            if config.ndocs // 4 < len(approx_scores):
                pids = pids[torch.topk(approx_scores, k=(config.ndocs // 4)).indices]
        else:
            pids = filter_pids_cpp.filter_pids_cpp(
                    pids, centroid_scores, self.embeddings.codes, self.doclens,
                    self.embeddings_strided.codes_strided.offsets, idx, config.ndocs
                )

        # Rank final list of docs using full approximate embeddings (including residuals)
        if self.use_gpu:
            D_packed, D_mask = self.lookup_pids(pids)
        else:
            D_packed = decompress_residuals_cpp.decompress_residuals_cpp(
                    pids,
                    self.doclens,
                    self.embeddings_strided.codes_strided.offsets,
                    self.codec.bucket_weights,
                    self.codec.reversed_bit_map,
                    self.codec.decompression_lookup_table,
                    self.embeddings.residuals,
                    self.embeddings.codes,
                    self.codec.centroids,
                    self.codec.dim,
                    self.codec.nbits
                )
            D_packed = torch.nn.functional.normalize(D_packed.to(torch.float32), p=2, dim=-1)
            D_mask = self.doclens[pids.long()]

        if Q.size(0) == 1:
            return colbert_score_packed(Q, D_packed, D_mask, config), pids

        D_strided = StridedTensor(D_packed, D_mask, use_gpu=self.use_gpu)
        D_padded, D_lengths = D_strided.as_padded_tensor()

        return colbert_score(Q, D_padded, D_lengths, config), pids
