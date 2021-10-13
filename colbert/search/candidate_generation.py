from colbert.utils.utils import batch
import torch

from colbert.search.strided_tensor import StridedTensor

"""
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)
"""

from .strided_tensor_core import StridedTensorCore, _create_mask, _create_view



class CandidateGeneration:
    def lengths2idxs(self, lengths, nprobe):
        """
            The batch version is slower. At least with a small number of [large] lengths.

            stride = lengths.max().item()
            tensor = torch.arange(0, lengths.size(0), device='cuda').unsqueeze(0).repeat(stride, 1)
            tensor = tensor[_create_mask(lengths, stride).T.contiguous()] // nprobe

            # TODO: Replace repeat with expand and test speed!
        """

        idxs = torch.empty(lengths.sum().item(), dtype=torch.long, device='cuda')

        offset = 0
        for idx, length in enumerate(lengths.tolist()):
            endpos = offset + length
            idxs[offset:endpos] = idx // nprobe
            offset = endpos

        return idxs

    def generate_candidate_eids(self, Q, nprobe):
        cells = (self.codec.centroids @ Q.T).topk(nprobe, dim=0, sorted=False).indices.permute(1, 0)  # (32, nprobe)
        cells = cells.flatten().contiguous()  # (32 * nprobe,)

        # print(f' CELLS:  {cells} ')

        # TODO: Try both lookup_staggered() and lookup() here to compare speed with large IVFs.
        # It seems that lookup() is faster with small IVFs.

        eids, cell_lengths = self.ivf.lookup(cells)  # eids = (packedlen,)  lengths = (32 * nprobe,)

        return eids.cuda(), cells.cuda(), cell_lengths.cuda()

    def generate_candidate_scores(self, nprobe, Q, eids, cells, cell_lengths):
        idxs = self.lengths2idxs(cell_lengths, nprobe)

        E = self.lookup_eids(eids.long(), codes=cells[idxs].long()).cuda()  # (packedlen, 128)
        Q = Q[idxs]  # (packedlen, 128)

        scores = torch.bmm(Q.unsqueeze(1), E.unsqueeze(2)).flatten()   # (packedlen,)

        return scores

    # @profile
    def generate_candidates(self, config, Q):
        nprobe = config.nprobe
        ncandidates = config.ncandidates

        assert isinstance(self.ivf, StridedTensor)

        Q = Q.squeeze(0).cuda().half()
        assert Q.dim() == 2

        eids, cells, cell_lengths = self.generate_candidate_eids(Q, nprobe)

        # print(f' NPROBE: {nprobe} ')
        # print(f" DEPTH:  {depth} \t\t {min(depth, scores.size(-1))} ")

        pids = self.emb2pid[eids.long()].cuda()
        scores = self.generate_candidate_scores(nprobe, Q, eids, cells, cell_lengths).cuda()

        # EVENTUALLY: it's probably A LOT faster to sort in advance! And then collect the right ones in the loop using the output of lengths2idxs from before, permuted according to the sort.
        
        q_pids = []
        q_scores = []

        # print(cell_lengths.view(-1, 2).sum(-1).max() * 32, cell_lengths.view(-1, 2).sum(-1).sum())

        offset = 0
        for idx, qlengths in enumerate(batch(cell_lengths.tolist(), nprobe)):
            """
                For each query vector's scores.

                # TODO: Each kernel launch x 32 = 0.5 ms even if it does no work! Below is at least 20 kernel launches, so that's at least 10 ms FOR NO WORK!
                # Because of that, let's batch the sort, it really works. And we can easily run unique_consecutive batch too, but we lose the dims in pids_counts :(
                # So: let's run ONLY unique_consecutive without batches. That adds just 1 ms of overhead.
                # See notebooks/2021/09/pytorch/sorting.ipynb for how to batch the rest, including take_along_dim.
            """
            length = sum(qlengths)
            endpos = offset + length
            
            scores_ = scores[offset:endpos]
            pids_ = pids[offset:endpos].contiguous()

            sorter = pids_.sort()
            pids_, scores_ = sorter.values, scores_[sorter.indices]

            pids_, pids_counts = torch.unique_consecutive(pids_, return_counts=True)
            pids_, pids_counts = pids_.cuda(), pids_counts.cuda()

            pids_offsets = pids_counts.cumsum(dim=0) - pids_counts[0]
            
            stride = pids_counts.max().item()

            # if idx == Q.size(0)-1:
            scores_ = torch.nn.functional.pad(scores_, (0, stride)).cuda()

            scores_padded = _create_view(scores_, stride, [])[pids_offsets] * _create_mask(pids_counts, stride)
            scores_maxsim = scores_padded.max(-1).values

            q_pids.append(pids_)
            q_scores.append(scores_maxsim)

            offset = endpos

        # assert endpos == pids.size(0), (pids.size(), endpos)
        
        pids = torch.cat(q_pids)
        scores = torch.cat(q_scores)

        indices = pids.sort().indices.cuda()

        pids = pids[indices]
        scores = scores[indices]

        pids, pids_counts = torch.unique_consecutive(pids, return_counts=True)
        pids, pids_counts = pids.cuda(), pids_counts.cuda()

        pids_offsets = pids_counts.cumsum(dim=0) - pids_counts[0]

        stride = pids_counts.max().item()

        scores = torch.nn.functional.pad(scores, (0, stride)) 
        scores_padded = _create_view(scores, stride, [])[pids_offsets] * _create_mask(pids_counts, stride)
        scores_lb = scores_padded.sum(-1)


        assert scores_lb.size(0) == pids.size(0)

        if scores_lb.size(0) > ncandidates:
            pids = pids[scores_lb.topk(ncandidates, dim=-1, sorted=True).indices]

            # TODO: Interestingly, this new approach might mean that smaller depth but larger nprobe leads to better quality on MARCO, because that affects the LB.
        
        # TODO: Eventually, if depth isn't given, it should be based on the requested k. Let's say max(256, k*8)?
        # Also, we can easily do scores_ub by padding with the bottom score from the scores_maxsim per query vector.

        # global EXECUTIONS
        # EXECUTIONS += 1

        # if EXECUTIONS > 10:
        #     exit()

        # # if depth < eids.size(0):

        #     """
        #     TODO: topk is too slow, do the following instead.

        #     Binary search for a threshold that admits [*very* roughly] the right number of candidates.

        #     Start with 0.5, and do (scores < threshold).sum().
        #     If close enough, do arange()[scores < threshold] -> top_indices.
        #     Else, increase or decrease the threshold to mid(0, 0.5) or mid(0.5, 1).
        #     """
            
        #     # top_indices = scores.topk(depth, dim=-1, sorted=False).indices
        #     # # top_indices = self.topk_indices(scores, depth)
        #     # eids = eids[top_indices]


        return pids #torch.unique(eids.cuda(), sorted=False)


EXECUTIONS = 0

"""
TODO: Consider the code below for *per-vector* topk!
----- However, there's no need to create a StridedTensor just to pad. We can pad the packed thing directly with a view!

scores_padded, scores_mask = StridedTensor(scores, cell_lengths).as_padded_tensor()  # (32 * nprobe, maxl,)
scores_padded = scores_padded * scores_mask  # CORRECTION: mask with -inf

&

scores_padded = self.generate_candidate_scores(nprobe, Q, eids, cell_lengths)

depth = min(depth // 16, scores_padded.size(-1))  # Decide depth per Q embedding!

top_indices = scores_padded.topk(depth, dim=-1).indices
top_indices = top_indices + cell_pfxsum.unsqueeze(1)
top_indices = top_indices.clamp(max=eids.size(0)-1)  # Clamping due to padding/zeros leading to indices.
        
eids = eids[top_indices]
"""


"""
OLD approach:

    def queries_to_embedding_ids(self, faiss_depth, nprobe, Q):
        assert isinstance(self.ivf, list)

        Q = Q.squeeze(0).cuda().half()
        assert Q.dim() == 2

        # 1- Find the nearest 2 centroids to each query embedding -> (32,2) -> (64,)
        probed_centroid_ids = torch.topk((Q @ self.codec.centroids.T), k=nprobe, dim=-1).indices
        probed_centroid_ids = probed_centroid_ids.flatten().tolist()

        # 2- Lookup the embeddings from self.ivf[idx].
        embedding_ids = [self.ivf[idx] for idx in probed_centroid_ids]

        qvector_ids = [[idx] * len(eids) for idx, eids in enumerate(embedding_ids)]
        qvector_ids = torch.LongTensor(flatten(qvector_ids)).cuda() // nprobe
        embedding_ids = torch.LongTensor(flatten(embedding_ids))

        Q = Q.half().cuda()[qvector_ids]
        E = self.lookup_eids(embedding_ids).cuda()

        scores_idx = torch.einsum("bd,bd->b", Q, E).topk(min(faiss_depth, E.size(0)), dim=-1).indices

        return torch.unique(embedding_ids[scores_idx].cuda())

"""


