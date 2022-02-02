import torch

from colbert.search.strided_tensor import StridedTensor
from .strided_tensor_core import _create_mask, _create_view


class CandidateGeneration:
    def generate_candidate_eids(self, Q, nprobe):
        cells = (self.codec.centroids @ Q.T).topk(nprobe, dim=0, sorted=False).indices.permute(1, 0)  # (32, nprobe)
        cells = cells.flatten().contiguous()  # (32 * nprobe,)
        cells = cells.unique(sorted=False)

        eids, cell_lengths = self.ivf.lookup(cells)  # eids = (packedlen,)  lengths = (32 * nprobe,)

        return eids.cuda().long()

    def generate_candidate_scores(self, Q, eids):
        E = self.lookup_eids(eids).cuda()
        return (Q.unsqueeze(0) @ E.unsqueeze(2)).squeeze(-1).T

    def generate_candidates(self, config, Q):
        nprobe = config.nprobe
        ncandidates = config.ncandidates

        assert isinstance(self.ivf, StridedTensor)

        Q = Q.squeeze(0).cuda().half()
        assert Q.dim() == 2

        eids = self.generate_candidate_eids(Q, nprobe)
        eids = torch.unique(eids, sorted=False)

        pids = self.emb2pid[eids.long()].cuda()
        sorter = pids.sort()
        pids = sorter.values

        pids, pids_counts = torch.unique_consecutive(pids, return_counts=True)
        pids, pids_counts = pids.cuda(), pids_counts.cuda()

        if len(pids) <= ncandidates:
            return pids

        pids_offsets = pids_counts.cumsum(dim=0) - pids_counts[0]

        eids = eids[sorter.indices]
        scores = self.generate_candidate_scores(Q, eids)

        stride = pids_counts.max().item()

        scores_dim1, scores_dim2 = scores.size()
        scores = scores.flatten()
        scores = torch.nn.functional.pad(scores, (0, stride)).cuda()

        pids_offsets2 = pids_offsets.repeat(scores_dim1, 1)
        pids_offsets2 += torch.arange(scores_dim1).cuda().unsqueeze(1) * scores_dim2
        pids_offsets2 = pids_offsets2.flatten()

        pids_counts2 = pids_counts.repeat(scores_dim1, 1).flatten()

        scores_padded = _create_view(scores, stride, [])[pids_offsets2] * _create_mask(pids_counts2, stride)
        scores_maxsim = scores_padded.max(-1).values

        num_pids = pids.size(0)
        pids = pids.unsqueeze(0).repeat(scores_dim1, 1).flatten()
        scores = scores_maxsim

        assert pids.size() == scores.size()

        indices = torch.arange(0, scores_dim1).cuda() * num_pids
        indices = indices.repeat(num_pids, 1)
        indices += torch.arange(num_pids).cuda().unsqueeze(1)
        indices = indices.flatten()

        pids = pids[indices]
        scores = scores[indices]

        pids, pids_counts = torch.unique_consecutive(pids, return_counts=True)
        pids_offsets = pids_counts.cumsum(dim=0) - pids_counts[0]
        stride = pids_counts.max().item()

        scores = torch.nn.functional.pad(scores, (0, stride))
        scores_padded = _create_view(scores, stride, [])[pids_offsets] * _create_mask(pids_counts, stride)
        scores_lb = scores_padded.sum(-1)

        assert scores_lb.size(0) == pids.size(0)

        if scores_lb.size(0) > ncandidates:
            pids = pids[scores_lb.topk(ncandidates, dim=-1, sorted=True).indices]

        return pids
