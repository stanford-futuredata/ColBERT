import torch
import ujson

from colbert.utils.utils import flatten, print_message

from colbert.indexing.loaders import load_doclens
from colbert.indexing.codecs.residual_embeddings_strided import ResidualEmbeddingsStrided

from colbert.search.strided_tensor import StridedTensor
from colbert.search.candidate_generation import CandidateGeneration

from .index_loader import IndexLoader
from colbert.modeling.colbert import colbert_score, colbert_score_packed

"""
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)
"""


class IndexScorer(IndexLoader, CandidateGeneration):
    def __init__(self, index_path):
        super().__init__(index_path)

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

        # self.emb2pid = self.emb2pid.cuda() # FIXME: REMOVE THIS LINE!

    def lookup_eids(self, embedding_ids, codes=None, out_device='cuda'):
        return self.embeddings_strided.lookup_eids(embedding_ids, codes=codes, out_device=out_device)

    def lookup_pids(self, passage_ids, out_device='cuda', return_mask=False):
        return self.embeddings_strided.lookup_pids(passage_ids, out_device)

    def retrieve(self, config, Q):
        # embedding_ids = self.queries_to_embedding_ids(faiss_depth, nprobe, Q)
        embedding_ids = self.generate_candidates(config, Q)

        return embedding_ids
        
        pids = self.embedding_ids_to_pids(embedding_ids)

        return pids

    def embedding_ids_to_pids(self, embedding_ids):
        all_pids = torch.unique(self.emb2pid[embedding_ids.long()].cuda(), sorted=False)
        return all_pids

    def rank(self, config, Q, k):
        with torch.inference_mode():
            pids = self.retrieve(config, Q)
            scores = self.score_pids(config, Q, pids, k)

            scores_sorter = scores.sort(descending=True)
            pids, scores = pids[scores_sorter.indices].tolist(), scores_sorter.values.tolist()

            return pids, scores

    def score_pids(self, config, Q, pids, k):
        """
            Always supply a flat list or tensor for `pids`.

            Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
            If Q.size(0) is 1, the matrix will be compared with all passages.
            Otherwise, each query matrix will be compared against the *aligned* passage.
        """


        # """ # OLD code: 
        D_packed, D_mask = self.lookup_pids(pids)

        if Q.size(0) == 1:
            return colbert_score_packed(Q, D_packed, D_mask)

        D_padded, D_lengths = StridedTensor(D_packed, D_mask).as_padded_tensor()

        return colbert_score(Q, D_padded, D_lengths)
        # """



        assert isinstance(pids[0], int) or pids.dim() == 1

        ncandidates = min(config.ncandidates, pids.size(0))
        # bsizes = [min(k * 4, ncandidates)]
        # total = bsizes[0]

        # while total < ncandidates:
        #     bsizes.append(min(bsizes[-1] * 2, ncandidates - total))
        #     total += bsizes[-1]

        all_scores = []
        thresholds = []
        maxscores = []

        for batch_pids in pids.split(min(1024, ncandidates)):
            D_packed, D_mask = self.lookup_pids(batch_pids)

            if Q.size(0) == 1:
                scores = colbert_score_packed(Q, D_packed, D_mask)
            else:
                D_padded, D_lengths = StridedTensor(D_packed, D_mask).as_padded_tensor()
                scores = colbert_score(Q, D_padded, D_lengths)

            all_scores.append(scores)

            threshold = torch.cat(all_scores).float().topk(k).values[-1].item()
            thresholds.append(threshold)

            maxscore = scores.max().item()
            maxscores.append(maxscore)
        
        with open('logs/index_storage.txt', 'a') as f:
            f.write(ujson.dumps((ncandidates, thresholds, maxscores)) + '\n')

        return torch.cat(all_scores)
