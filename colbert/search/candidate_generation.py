import torch

from colbert.search.strided_tensor import StridedTensor
from .strided_tensor_core import _create_mask, _create_view


class CandidateGeneration:

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

    def generate_candidate_eids(self, Q, nprobe):
        if self.use_gpu:
            cells = (self.codec.centroids @ Q.T).topk(nprobe, dim=0, sorted=False).indices.permute(1, 0)  # (32, nprobe)
        else:
            assert self.codec.centroids.dtype == torch.float32
            assert Q.dtype == torch.float32
            cells = (self.codec.centroids @ Q.T).topk(nprobe, dim=0, sorted=False).indices.permute(1, 0)  # (32, nprobe)
        cells = cells.flatten().contiguous()  # (32 * nprobe,)
        cells = cells.unique(sorted=False)

        eids, cell_lengths = self.ivf.lookup(cells)  # eids = (packedlen,)  lengths = (32 * nprobe,)

        if self.use_gpu:
            return eids.cuda(), cells.cuda(), cell_lengths.cuda()
        else:
            return eids, cells, cell_lengths

    def generate_candidate_scores(self, nprobe, Q, eids, cells, cell_lengths):
        if self.use_gpu:
            eids = eids.cuda().long()
        else:
            eids = eids.long()
        eids = torch.unique(eids, sorted=False)
        E = self.lookup_eids(eids)
        if self.use_gpu:
            E = E.cuda()

        scores = (Q.unsqueeze(0) @ E.unsqueeze(2)).squeeze(-1).T
        if self.use_gpu:
            scores = scores.cuda()

        return scores, eids

    def generate_candidates(self, config, Q):
        nprobe = config.nprobe
        ncandidates = config.ncandidates

        assert isinstance(self.ivf, StridedTensor)

        Q = Q.squeeze(0)
        if self.use_gpu:
            Q = Q.cuda().half()
        assert Q.dim() == 2

        eids, cells, cell_lengths = self.generate_candidate_eids(Q, nprobe)

        scores, eids = self.generate_candidate_scores(nprobe, Q, eids, cells, cell_lengths)
        pids = self.emb2pid[eids.long()]
        if self.use_gpu:
            pids = pids.cuda()

        sorter = pids.sort()
        pids, scores = sorter.values, torch.take_along_dim(scores, sorter.indices.unsqueeze(0), dim=-1)

        pids, pids_counts = torch.unique_consecutive(pids, return_counts=True)
        if self.use_gpu:
            pids, pids_counts = pids.cuda(), pids_counts.cuda()
        pids_offsets = pids_counts.cumsum(dim=0) - pids_counts[0]

        if len(pids) <= ncandidates:
            return pids

        stride = pids_counts.max().item()
        scores = torch.nn.functional.pad(scores, (0, stride))
        if self.use_gpu:
            scores = scores.cuda()

        q_scores = []

        for idx in range(scores.size(0)):
            scores_ = scores[idx]

            scores_padded = _create_view(scores_, stride, [])[pids_offsets] * _create_mask(pids_counts, stride, use_gpu=self.use_gpu)
            scores_maxsim = scores_padded.max(-1).values

            q_scores.append(scores_maxsim)

        # TODO: The code below can be dramatically optimized. No need for sorting or cumsum or unique or pad
        # There's also no need for permuting with [indices]
        # Just need to view() and permute() scores

        pids = pids.unsqueeze(0).repeat(scores.size(0), 1).flatten()
        scores = torch.cat(q_scores)

        indices = pids.sort().indices
        if self.use_gpu:
            indices = indices.cuda()

        pids = pids[indices]
        scores = scores[indices]

        pids, pids_counts = torch.unique_consecutive(pids, return_counts=True)
        if self.use_gpu:
            pids, pids_counts = pids.cuda(), pids_counts.cuda()

        pids_offsets = pids_counts.cumsum(dim=0) - pids_counts[0]

        stride = pids_counts.max().item()

        scores = torch.nn.functional.pad(scores, (0, stride))
        scores_padded = _create_view(scores, stride, [])[pids_offsets] * _create_mask(pids_counts, stride, use_gpu=self.use_gpu)
        scores_lb = scores_padded.sum(-1)

        assert scores_lb.size(0) == pids.size(0)

        if scores_lb.size(0) > ncandidates:
            if self.use_gpu:
                pids = pids[scores_lb.topk(ncandidates, dim=-1, sorted=True).indices]
            else:
                pids = pids[scores_lb.to(torch.float32).topk(ncandidates, dim=-1, sorted=True).indices]

        return pids
