
# Could be .tsv or .json. The latter always allows more customization via optional parameters.
# I think it could be worth doing some kind of parallel reads too, if the file exceeds 1 GiBs.
# Just need to use a datastructure that shares things across processes without too much pickling.
# I think multiprocessing.Manager can do that!

import os
import itertools
import mmap

from colbert.evaluation.loaders import load_collection
from colbert.infra.run import Run


class Collection:
    def __init__(self, path=None, data=None, load_collection_with_mmap=True):
        self.path = path
        self.load_collection_with_mmap = load_collection_with_mmap
        if self.load_collection_with_mmap:
            with open(self.path, 'r') as file:
                self.mmap_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            self.data = data or self._load_file(path)

    def __del__(self):
        if hasattr(self, "load_collection_with_mmap") and self.load_collection_with_mmap and self.mmap_file:
            self.mmap_file.close()

    def __iter__(self):
        if self.load_collection_with_mmap:
            self.mmap_file.seek(0)
            return self
        else:
            return iter(self.data)

    def __next__(self):
        if self.load_collection_with_mmap:
            line = self.mmap_file.readline()
            if not line:
                raise StopIteration
            return line.decode("utf-8").strip().split("\t")[1]
        else:
            raise StopIteration

    def __getitem__(self, item):
        if self.load_collection_with_mmap:
            with open(self.path, 'r') as file:
                self.mmap_file.seek(0)

                for idx, line in enumerate(self):
                    if idx == item:
                        return line
                raise IndexError("Index out of range")
        else:
            return self.data[item]

    def __len__(self):
        if self.load_collection_with_mmap:
            self.mmap_file.seek(0)
            count = 0
            while self.mmap_file.readline():
                count += 1
            return count
        else:
            return len(self.data)

    def _load_file(self, path):
        self.path = path
        return self._load_tsv(path) if path.endswith('.tsv') else self._load_jsonl(path)

    def _load_tsv(self, path):
        return load_collection(path)

    def _load_jsonl(self, path):
        raise NotImplementedError()

    def provenance(self):
        return self.path

    def toDict(self):
        return {'provenance': self.provenance()}

    def save(self, new_path):
        assert new_path.endswith('.tsv'), "TODO: Support .json[l] too."
        assert not os.path.exists(new_path), new_path

        with Run().open(new_path, 'w') as f:
            # TODO: expects content to always be a string here; no separate title!
            for pid, content in enumerate(self.data):
                content = f'{pid}\t{content}\n'
                f.write(content)

            return f.name

    def enumerate(self, rank):
        for _, offset, passages in self.enumerate_batches(rank=rank):
            for idx, passage in enumerate(passages):
                yield (offset + idx, passage)

    def enumerate_batches(self, rank, chunksize=None):
        assert rank is not None, "TODO: Add support for the rank=None case."

        chunksize = chunksize or self.get_chunksize()

        offset = 0
        iterator = iter(self)

        for chunk_idx, owner in enumerate(itertools.cycle(range(Run().nranks))):
            L = [line for _, line in zip(range(chunksize), iterator)]

            if len(L) > 0 and owner == rank:
                yield (chunk_idx, offset, L)

            offset += len(L)

            if len(L) < chunksize:
                return

    def get_chunksize(self):
        return min(25_000, 1 + len(self) // Run().nranks)  # 25k is great, 10k allows things to reside on GPU??

    @classmethod
    def cast(cls, obj, load_collection_with_mmap=False):
        if type(obj) is str:
            return cls(path=obj, load_collection_with_mmap=load_collection_with_mmap)

        if type(obj) is list:
            return cls(data=obj)

        if type(obj) is cls:
            return obj

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"


# TODO: Look up path in some global [per-thread or thread-safe] list.
