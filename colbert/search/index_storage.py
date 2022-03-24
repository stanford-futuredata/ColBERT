import torch

from colbert.utils.utils import flatten, print_message

from colbert.indexing.loaders import load_doclens
from colbert.indexing.codecs.residual_embeddings_strided import ResidualEmbeddingsStrided

from colbert.search.strided_tensor import StridedTensor
from colbert.search.candidate_generation import CandidateGeneration

from .index_loader import IndexLoader
from colbert.modeling.colbert import colbert_score, colbert_score_packed, colbert_score_reduce

from math import ceil

import os
import pathlib
from torch.utils.cpp_extension import load

segmented_maxsim_cpp = load(
    name="segmented_maxsim_cpp",
    sources=[
        os.path.join(
            pathlib.Path(__file__).parent.resolve(), "segmented_maxsim.cpp"
        ),
    ],
    extra_cflags=["-fopenmp"],
    extra_ldflags=["-lgomp"],
)

class IndexScorer(IndexLoader, CandidateGeneration):
    def __init__(self, index_path, use_gpu=True):
        super().__init__(index_path=index_path, use_gpu=use_gpu)

        self.embeddings_strided = ResidualEmbeddingsStrided(self.codec, self.embeddings, self.doclens)

        print_message("#> Building the emb2pid mapping..")
        all_doclens = load_doclens(index_path, flatten=False)

        assert self.num_embeddings == sum(flatten(all_doclens))

        all_doclens = flatten(all_doclens)
        total_num_embeddings = sum(all_doclens)

        self.emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

        """
        EVENTUALLY: Build this in advance and load it from disk.

        EVENTUALLY: Use two tensors. emb2pid_offsets will have every 256th element. emb2pid_delta will have the delta
                    from the corresponding offset,
        """

        offset_doclens = 0
        for pid, dlength in enumerate(all_doclens):
            self.emb2pid[offset_doclens: offset_doclens + dlength] = pid
            offset_doclens += dlength

        print_message("len(self.emb2pid) =", len(self.emb2pid))

    def lookup_eids(self, embedding_ids, codes=None, out_device='cuda'):
        return self.embeddings_strided.lookup_eids(embedding_ids, codes=codes, out_device=out_device)

    def lookup_pids(self, passage_ids, out_device='cuda', return_mask=False):
        return self.embeddings_strided.lookup_pids(passage_ids, out_device)

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

        # Filter docs using pruned centroid scores
        approx_scores = []
        idx = centroid_scores.max(-1).values >= config.centroid_score_threshold
        for i in range(0, ceil(len(pids) / batch_size)):
            pids_ = pids[i * batch_size : (i+1) * batch_size]
            codes_packed, codes_lengths = self.embeddings_strided.lookup_codes(pids_)
            idx_ = idx[codes_packed.long()]
            pruned_codes_strided = StridedTensor(idx_, codes_lengths, use_gpu=self.use_gpu)
            pruned_codes_padded, pruned_codes_mask = pruned_codes_strided.as_padded_tensor()
            pruned_codes_lengths = (pruned_codes_padded * pruned_codes_mask).sum(dim=1)
            codes_packed_ = codes_packed[idx_]
            approx_scores_ = centroid_scores[codes_packed_.long()]
            if approx_scores_.shape[0] == 0:
                approx_scores.append(torch.zeros((len(pids_),), dtype=approx_scores_.dtype))
                continue
            if self.use_gpu:
                approx_scores_strided = StridedTensor(approx_scores_, pruned_codes_lengths, use_gpu=self.use_gpu)
                approx_scores_padded, approx_scores_mask = approx_scores_strided.as_padded_tensor()
                approx_scores_ = colbert_score_reduce(approx_scores_padded, approx_scores_mask, config)
            else:
                approx_scores_ = segmented_maxsim_cpp.segmented_maxsim_cpp(approx_scores_, pruned_codes_lengths)
            approx_scores.append(approx_scores_)
        approx_scores = torch.cat(approx_scores, dim=0)

        # Filter docs using full centroid scores
        if config.ndocs < len(approx_scores):
            pids = pids[torch.topk(approx_scores, k=config.ndocs).indices]
        codes_packed, codes_lengths = self.embeddings_strided.lookup_codes(pids)
        approx_scores = centroid_scores[codes_packed.long()]
        if self.use_gpu:
            approx_scores_strided = StridedTensor(approx_scores, codes_lengths, use_gpu=self.use_gpu)
            approx_scores_padded, approx_scores_mask = approx_scores_strided.as_padded_tensor()
            approx_scores = colbert_score_reduce(approx_scores_padded, approx_scores_mask, config)
        else:
            approx_scores = segmented_maxsim_cpp.segmented_maxsim_cpp(approx_scores, codes_lengths)

        # Rank final list of docs using full approximate embeddings (including residuals)
        if config.ndocs // 4 < len(approx_scores):
            pids = pids[torch.topk(approx_scores, k=(config.ndocs // 4)).indices]
        D_packed, D_mask = self.lookup_pids(pids)
        if Q.size(0) == 1:
            return colbert_score_packed(Q, D_packed, D_mask, config), pids

        D_strided = StridedTensor(D_packed, D_mask, use_gpu=self.use_gpu)
        D_padded, D_lengths = D_strided.as_padded_tensor()

        return colbert_score(Q, D_padded, D_lengths, config), pids
