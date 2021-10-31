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
            idxs[offset:endpos] = idx  # % nprobe
            offset = endpos

        return idxs

    def generate_candidate_eids(self, Q, nprobe):
        cells = (self.codec.centroids @ Q.T).topk(nprobe, dim=0, sorted=False).indices.permute(1, 0)  # (32, nprobe)
        cells = cells.flatten().contiguous()  # (32 * nprobe,)
        cells = cells.unique(sorted=False)

        # print(f'cells = {cells}')

        # print(f' CELLS:  {cells} ')

        # TODO: Try both lookup_staggered() and lookup() here to compare speed with large IVFs.
        # It seems that lookup() is faster with small IVFs.

        eids, cell_lengths = self.ivf.lookup(cells)  # eids = (packedlen,)  lengths = (32 * nprobe,)

        # print(f'cell_lengths = {cell_lengths}')
        # print(f'eids[420:440] = {eids[420:440]}')

        return eids.cuda(), cells.cuda(), cell_lengths.cuda()

    def generate_candidate_scores(self, nprobe, Q, eids, cells, cell_lengths):
        eids = eids.cuda().long()
        eids = torch.unique(eids, sorted=False)
        E = self.lookup_eids(eids).cuda()

        scores = (Q.unsqueeze(0) @ E.unsqueeze(2)).squeeze(-1).T

        return scores.cuda(), eids

        # idxs = self.lengths2idxs(cell_lengths, nprobe)

        # print(f'idxs[420:440] = {idxs[0:540]}')
        # print(f'cells[idxs][420:440] = {cells[idxs][420:440]}')

        # E = self.lookup_eids(eids.long(), codes=cells[idxs].long()).cuda()  # (packedlen, 128)

        E = self.lookup_eids(eids.long()).cuda()  # (packedlen, 128)

        scores = (Q.unsqueeze(0) @ E.unsqueeze(2)).squeeze(-1).T

        # Q = Q[idxs]  # (packedlen, 128)

        # scores = torch.bmm(Q.unsqueeze(1), E.unsqueeze(2)).flatten()   # (packedlen,)

        return scores

    # @profile
    def generate_candidates(self, config, Q):
        nprobe = config.nprobe
        ncandidates = config.ncandidates

        assert isinstance(self.ivf, StridedTensor)

        Q = Q.squeeze(0).cuda().half()
        assert Q.dim() == 2

        eids, cells, cell_lengths = self.generate_candidate_eids(Q, nprobe)

        scores, eids = self.generate_candidate_scores(nprobe, Q, eids, cells, cell_lengths)
        pids = self.emb2pid[eids.long()].cuda()

        sorter = pids.sort()
        pids, scores = sorter.values, torch.take_along_dim(scores, sorter.indices.unsqueeze(0), dim=-1)

        pids, pids_counts = torch.unique_consecutive(pids, return_counts=True)
        pids, pids_counts = pids.cuda(), pids_counts.cuda()
        pids_offsets = pids_counts.cumsum(dim=0) - pids_counts[0]

        if len(pids) <= ncandidates:
            return pids

        stride = pids_counts.max().item()
        scores = torch.nn.functional.pad(scores, (0, stride)).cuda()

        q_scores = []

        for idx in range(scores.size(0)):
            scores_ = scores[idx]

            scores_padded = _create_view(scores_, stride, [])[pids_offsets] * _create_mask(pids_counts, stride)
            scores_maxsim = scores_padded.max(-1).values

            q_scores.append(scores_maxsim)

        # TODO: The code below can be dramatically optimized. No need for sorting or cumsum or unique or pad
        # There's also no need for permuting with [indices]
        # Just need to view() and permute() scores

        pids = pids.unsqueeze(0).repeat(scores.size(0), 1).flatten()
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

        return pids
