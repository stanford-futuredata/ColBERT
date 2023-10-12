import os
import torch
import ujson
import tqdm

from colbert.indexing.codecs.residual_embeddings_strided import ResidualEmbeddingsStrided
from colbert.utils.utils import print_message

class ResidualEmbeddings:
    Strided = ResidualEmbeddingsStrided

    def __init__(self, codes, residuals):
        """
            Supply the already compressed residuals.
        """

        # assert isinstance(residuals, bitarray), type(residuals)
        assert codes.size(0) == residuals.size(0), (codes.size(), residuals.size())
        assert codes.dim() == 1 and residuals.dim() == 2, (codes.size(), residuals.size())
        assert residuals.dtype == torch.uint8

        self.codes = codes.to(torch.int32)  # (num_embeddings,) int32
        self.residuals = residuals   # (num_embeddings, compressed_dim) uint8

    @classmethod
    def load_chunks(cls, index_path, chunk_idxs, num_embeddings, load_index_with_mmap=False):
        num_embeddings += 512  # pad for access with strides

        dim, nbits = get_dim_and_nbits(index_path)

        if load_index_with_mmap:
            if len(chunk_idxs) != 1:
                raise ValueError(
                    "Index must only have 1 chunk to load with memory mapping!"
                    "Use the colbert/utils/coalesce.py to prepare index for memory mapping."
                )

            print_message("#> Loading codes and residuals with memory mapping...")

            residuals_path = os.path.join(index_path, f'0.residuals.pt')
            codes_path = os.path.join(index_path, f'0.codes.pt')

            codes_size = get_codes_size(index_path, 0)
            storage = torch.IntStorage.from_file(filename=codes_path, shared=True, size=codes_size + 80)
            # Trim the header, which is 320 bytes, or 80x 32-byte ints
            codes = torch.IntTensor(storage)[80:]

            residuals_size, codes_size, packed_dim = get_residuals_size(index_path, 0)
            storage = torch.ByteStorage.from_file(filename=residuals_path, shared=True, size=residuals_size + 320)
            ret = torch.ByteTensor(storage)
            # Trim to 320-byte header
            ret = ret[320:]
            ret = torch.reshape(ret, (codes_size, packed_dim))
            residuals = ret
        else:
            print_message("#> Loading codes and residuals...")

            codes = torch.empty(num_embeddings, dtype=torch.int32)
            residuals = torch.empty(num_embeddings, dim // 8 * nbits, dtype=torch.uint8)

            codes_offset = 0

            for chunk_idx in tqdm.tqdm(chunk_idxs):
                chunk = cls.load(index_path, chunk_idx)

                codes_endpos = codes_offset + chunk.codes.size(0)

                # Copy the values over to the allocated space
                codes[codes_offset:codes_endpos] = chunk.codes
                residuals[codes_offset:codes_endpos] = chunk.residuals

                codes_offset = codes_endpos

        return cls(codes, residuals)

    @classmethod
    def load(cls, index_path, chunk_idx):
        codes = cls.load_codes(index_path, chunk_idx)
        residuals = cls.load_residuals(index_path, chunk_idx)

        return cls(codes, residuals)

    @classmethod
    def load_codes(self, index_path, chunk_idx):
        codes_path = os.path.join(index_path, f'{chunk_idx}.codes.pt')
        return torch.load(codes_path, map_location='cpu')

    @classmethod
    def load_residuals(self, index_path, chunk_idx):
        residuals_path = os.path.join(index_path, f'{chunk_idx}.residuals.pt')  # f'{chunk_idx}.residuals.bn'
        # return _load_bitarray(residuals_path)

        return torch.load(residuals_path, map_location='cpu')

    def save(self, path_prefix):
        codes_path = f'{path_prefix}.codes.pt'
        residuals_path = f'{path_prefix}.residuals.pt'  # f'{path_prefix}.residuals.bn'

        torch.save(self.codes, codes_path)
        torch.save(self.residuals, residuals_path)
        # _save_bitarray(self.residuals, residuals_path)

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

def get_codes_size(index_path, chunk_idx):
    # TODO: Ideally load this using ColBERTConfig.load_from_index!
    with open(os.path.join(index_path, f'{chunk_idx}.metadata.json')) as f:
        metadata = ujson.load(f)

    return metadata['num_embeddings']

def get_residuals_size(index_path, chunk_idx):
    codes_size = get_codes_size(index_path, chunk_idx)
    dim, nbits = get_dim_and_nbits(index_path)

    packed_dim = dim // 8 * nbits
    return codes_size * packed_dim, codes_size, packed_dim
