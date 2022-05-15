import os
import torch
import ujson
from collections import defaultdict, namedtuple

from colbert.indexing.codecs.residual_embeddings_strided import ResidualEmbeddingsStrided
from colbert.utils.utils import print_message


class ResidualEmbeddings:
    Strided = ResidualEmbeddingsStrided

    def __init__(self, codes, residuals, mmap_index=False, pid_to_chunk_metadata=None):
        """
            Supply the already compressed residuals.
        """

        self.mmap_index = mmap_index
        if self.mmap_index:
            self.codes = codes
            self.residuals = residuals
            self.pid_to_chunk_metadata = pid_to_chunk_metadata
            return

        # assert isinstance(residuals, bitarray), type(residuals)
        assert codes.size(0) == residuals.size(0), (codes.size(), residuals.size())
        assert codes.dim() == 1 and residuals.dim() == 2, (codes.size(), residuals.size())
        assert residuals.dtype == torch.uint8

        self.codes = codes.to(torch.int32)  # (num_embeddings,) int32
        self.residuals = residuals   # (num_embeddings, compressed_dim) uint8

    @classmethod
    def load_chunks(cls, index_path, chunk_idxs, num_embeddings, mmap_index=False):

        num_embeddings += 512  # pad for access with strides

        dim, nbits = get_dim_and_nbits(index_path)
        packed_dim = dim // 8 * nbits

        if mmap_index:
            # Store the mmapped chunks directly without coalescing into in-memory buffer
            codes = []
            residuals = []
        else:
            codes = torch.empty(num_embeddings, dtype=torch.int32)
            residuals = torch.empty(num_embeddings, packed_dim, dtype=torch.uint8)

        codes_offset = 0
        pid_offset = 0

        ChunkMetadata = namedtuple('ChunkMetadata', 'chunk_id, passage_doclen, passage_offset')
        pid_to_chunk_metadata = {}  # pid -> [chunk id, passage doclen, passage offset in the chunk]

        for chunk_idx in chunk_idxs:
            with open(os.path.join(index_path, f'{chunk_idx}.metadata.json')) as f:
                metadata = ujson.load(f)

            with open(os.path.join(index_path, f'doclens.{chunk_idx}.json')) as f:
                chunk_doclens = ujson.load(f)

            pid_offset_in_chunk = 0
            for pid in range(pid_offset, pid_offset + metadata["num_passages"]):
                pid_doclen = chunk_doclens[pid - pid_offset]
                pid_to_chunk_metadata[pid] = ChunkMetadata(chunk_idx, pid_doclen, pid_offset_in_chunk)
                pid_offset_in_chunk += pid_doclen
            pid_offset += metadata["num_passages"]

            codes_endpos = codes_offset + metadata["num_embeddings"]

            chunk = cls.load(index_path, chunk_idx, codes_offset, codes_endpos, packed_dim, mmap_index)

            assert codes_endpos == codes_offset + chunk.codes.size(0)

            if mmap_index:
                codes.append(chunk.codes)
                residuals.append(chunk.residuals)
            else:
                # Copy the values over to the allocated space
                codes[codes_offset:codes_endpos] = chunk.codes
                residuals[codes_offset:codes_endpos] = chunk.residuals

            codes_offset = codes_endpos

        # codes, residuals = codes.cuda(), residuals.cuda()  # FIXME: REMOVE THIS LINE!

        return cls(codes, residuals, mmap_index=mmap_index, pid_to_chunk_metadata=pid_to_chunk_metadata)

    @classmethod
    def load(cls, index_path, chunk_idx, offset, endpos, packed_dim, mmap_index=False):
        codes = cls.load_codes(index_path, chunk_idx, offset, endpos, packed_dim, mmap_index)
        residuals = cls.load_residuals(index_path, chunk_idx, offset, endpos, packed_dim, mmap_index)

        return cls(codes, residuals)

    @classmethod
    def load_codes(self, index_path, chunk_idx, offset=None, endpos=None, packed_dim=None, mmap_index=False):
        codes_path = os.path.join(index_path, f'{chunk_idx}.codes.pt')

        if mmap_index:
            size = endpos - offset
            storage = torch.IntStorage.from_file(codes_path, True, size)
            return torch.IntTensor(storage)

        return torch.load(codes_path, map_location='cpu')

    @classmethod
    def load_residuals(self, index_path, chunk_idx, offset, endpos, packed_dim, mmap_index=False):
        residuals_path = os.path.join(index_path, f'{chunk_idx}.residuals.pt')  # f'{chunk_idx}.residuals.bn'
        # return _load_bitarray(residuals_path)

        if mmap_index:
            size = (endpos - offset) * packed_dim
            storage = torch.ByteStorage.from_file(residuals_path, True, size)
            return torch.ByteTensor(storage).reshape((endpos - offset), packed_dim)

        return torch.load(residuals_path, map_location='cpu')

    def save(self, path_prefix):
        codes_path = f'{path_prefix}.codes.pt'
        residuals_path = f'{path_prefix}.residuals.pt'  # f'{path_prefix}.residuals.bn'

        torch.save(self.codes, codes_path)
        torch.save(self.residuals, residuals_path)
        # _save_bitarray(self.residuals, residuals_path)

    def lookup_codes(self, pids):
        assert self.mmap_index
        # prev_pid = 0
        # for pid in pids:
        #     if pid.item() < prev_pid:
        #         print_message("not in order")
        #     prev_pid = pid.item()

        pids_per_chunk = defaultdict(list)
        codes_lengths = torch.zeros(len(pids))
        codes_size = 0
        for idx, pid in enumerate(pids):
            # print_message(f"pid shape: {pid.shape}, {len(pid.shape)}")
            pid_ = pid.item()
            chunk_idx, pid_doclen, _ = self.pid_to_chunk_metadata[pid_]
            pids_per_chunk[chunk_idx].append(pid_)
            codes_lengths[idx] = pid_doclen
            codes_size += pid_doclen
        codes = torch.zeros(codes_size)

        offset = 0
        for chunk_idx in sorted(pids_per_chunk.keys()):
            pids_ = pids_per_chunk[chunk_idx]
            for pid in pids_:
                _, pid_doclen, pid_offset_in_chunk = self.pid_to_chunk_metadata[pid]
                codes[offset:offset + pid_doclen] = \
                    self.codes[chunk_idx][pid_offset_in_chunk:pid_offset_in_chunk + pid_doclen]
                offset += pid_doclen

        return codes, codes_lengths

    def lookup_pids(self, pids):
        assert self.mmap_index
        print_message(f"mei-test residuals shape {self.residuals.shape}")
        packed_dim = self.residuals.shape[2]

        pids_per_chunk = defaultdict(list)
        residuals_lengths = torch.zeros(len(pids))
        residuals_size = 0
        for idx, pid in enumerate(pids):
            print_message(f"pid shape: {pid.shape}, {len(pid.shape)}")
            pid_ = pid.item()
            chunk_idx, pid_doclen, _ = self.pid_to_chunk_metadata[pid_]
            pids_per_chunk[chunk_idx].append(pid_)
            residuals_lengths[idx] = pid_doclen
            residuals_size += pid_doclen
        residuals = torch.zeros(residuals_size)

        offset = 0
        for chunk_idx in sorted(pids_per_chunk.keys()):
            pids_ = pids_per_chunk[chunk_idx]
            for pid in pids_:
                _, pid_doclen, pid_offset_in_chunk = self.pid_to_chunk_metadata[pid]
                residuals[offset:offset + pid_doclen, :packed_dim] = \
                    self.residuals[chunk_idx][pid_offset_in_chunk:pid_offset_in_chunk + pid_doclen, :packed_dim]
                offset += pid_doclen

        return residuals, residuals_lengths

    def __len__(self):
        return self.codes.size(0)


def get_dim_and_nbits(index_path):
    # TODO: Ideally load this using ColBERTConfig.load_from_index!
    with open(os.path.join(index_path, 'metadata.json')) as f:
        metadata = ujson.load(f)['config']

    dim = metadata['dim']
    nbits = metadata['nbits']

    assert (dim * nbits) % 8 == 0, (dim, nbits, dim * nbits)

    return dim, nbits
