"""
EVENTUALLY: Tune the batch sizes selected here for a good balance of speed and generality.
"""

import os
import cupy
import torch

from colbert.infra.config import ColBERTConfig
from colbert.indexing.codecs.residual_embeddings import ResidualEmbeddings


class ResidualCodec:
    Embeddings = ResidualEmbeddings

    def __init__(self, config, centroids, avg_residual=None, bucket_cutoffs=None, bucket_weights=None):
        self.dim, self.nbits = config.dim, config.nbits
        self.centroids = centroids.half().cuda()
        self.avg_residual = avg_residual

        if torch.is_tensor(self.avg_residual):
            self.avg_residual = self.avg_residual.half().cuda()
        
        if torch.is_tensor(bucket_cutoffs):
            bucket_cutoffs = bucket_cutoffs.cuda()
            bucket_weights = bucket_weights.half().cuda()

        self.bucket_cutoffs = bucket_cutoffs
        self.bucket_weights = bucket_weights

        self.arange_bits = torch.arange(0, self.nbits, device='cuda', dtype=torch.uint8)

    @classmethod
    def load(cls, index_path):
        config = ColBERTConfig.load_from_index(index_path)
        centroids_path = os.path.join(index_path, 'centroids.pt')
        avgresidual_path = os.path.join(index_path, 'avg_residual.pt')
        buckets_path = os.path.join(index_path, 'buckets.pt')

        centroids = torch.load(centroids_path, map_location='cpu')
        avg_residual = torch.load(avgresidual_path, map_location='cpu')
        bucket_cutoffs, bucket_weights = torch.load(buckets_path, map_location='cpu')

        if avg_residual.dim() == 0:
            avg_residual = avg_residual.item()

        return cls(config=config, centroids=centroids, avg_residual=avg_residual, bucket_cutoffs=bucket_cutoffs, bucket_weights=bucket_weights)

    def save(self, index_path):
        assert self.avg_residual is not None
        assert torch.is_tensor(self.bucket_cutoffs), self.bucket_cutoffs
        assert torch.is_tensor(self.bucket_weights), self.bucket_weights

        centroids_path = os.path.join(index_path, 'centroids.pt')
        avgresidual_path = os.path.join(index_path, 'avg_residual.pt')
        buckets_path = os.path.join(index_path, 'buckets.pt')

        torch.save(self.centroids, centroids_path)
        torch.save((self.bucket_cutoffs, self.bucket_weights), buckets_path)

        if torch.is_tensor(self.avg_residual):
            torch.save(self.avg_residual, avgresidual_path)
        else:
            torch.save(torch.tensor([self.avg_residual]), avgresidual_path)

    def compress(self, embs):
        codes, residuals = [], []

        for batch in embs.split(1 << 18):
            batch = batch.cuda().half()
            codes_ = self.compress_into_codes(batch, out_device=batch.device)
            centroids_ = self.lookup_centroids(codes_, out_device=batch.device)

            residuals_ = (batch - centroids_)

            codes.append(codes_.cpu())
            residuals.append(self.binarize(residuals_).cpu())

        codes = torch.cat(codes)
        residuals = torch.cat(residuals)

        return ResidualCodec.Embeddings(codes, residuals)

    def binarize(self, residuals):
        residuals = torch.bucketize(residuals.float(), self.bucket_cutoffs).to(dtype=torch.uint8)
        residuals = residuals.unsqueeze(-1).expand(*residuals.size(), self.nbits)  # add a new nbits-wide dim
        residuals = residuals >> self.arange_bits  # divide by 2^bit for each bit position
        residuals = residuals & 1  # apply mod 2 to binarize

        assert self.dim % 8 == 0
        assert self.dim % (self.nbits * 8) == 0, (self.dim, self.nbits)

        residuals_packed = cupy.packbits(cupy.asarray(residuals.contiguous().flatten()))
        residuals_packed = torch.as_tensor(residuals_packed, dtype=torch.uint8)

        residuals_packed = residuals_packed.reshape(residuals.size(0), self.dim // 8 * self.nbits)

        return residuals_packed

    def compress_into_codes(self, embs, out_device):
        """
            EVENTUALLY: Fusing the kernels or otherwise avoiding materalizing the entire matrix before max(dim=0)
                        seems like it would help here a lot.
        """

        codes = []

        bsize = (1 << 29) // self.centroids.size(0)
        for batch in embs.split(bsize):
            indices = (self.centroids @ batch.T.cuda().half()).max(dim=0).indices.to(device=out_device)
            codes.append(indices)

        return torch.cat(codes)

    def lookup_centroids(self, codes, out_device):
        """
            Handles multi-dimensional codes too.

            EVENTUALLY: The .split() below should happen on a flat view.
        """

        centroids = []

        for batch in codes.split(1 << 20):
            centroids.append(self.centroids[batch.cuda().long()].to(device=out_device))

        return torch.cat(centroids)

    def decompress(self, compressed_embs: Embeddings):
        """
            We batch below even if the target device is CUDA to avoid large temporary buffers causing OOM.
        """

        codes, residuals = compressed_embs.codes, compressed_embs.residuals

        D = []
        for codes_, residuals_ in zip(codes.split(1 << 15), residuals.split(1 << 15)):
            codes_, residuals_ = codes_.cuda(), residuals_.cuda()
            centroids_ = self.lookup_centroids(codes_, out_device='cuda')
            residuals_ = self.decompress_residuals(residuals_).to(device=centroids_.device)

            centroids_.add_(residuals_)
            D_ = torch.nn.functional.normalize(centroids_, p=2, dim=-1).half()
            D.append(D_)
        
        return torch.cat(D)

    def decompress_residuals(self, binary_residuals):
        assert binary_residuals.dim() == 2, binary_residuals.size()
        assert binary_residuals.size(1) == self.dim // 8 * self.nbits, binary_residuals.size()

        residuals = cupy.unpackbits(cupy.asarray(binary_residuals.contiguous().flatten()))

        residuals = torch.as_tensor(residuals, dtype=torch.uint8, device='cuda')

        if self.nbits > 1:
            residuals = residuals.reshape(binary_residuals.size(0), self.dim, self.nbits)
            residuals = (residuals << self.arange_bits).sum(-1)

        residuals = residuals.reshape(binary_residuals.size(0), self.dim)
        residuals = self.bucket_weights[residuals.long()].cuda()

        return residuals
