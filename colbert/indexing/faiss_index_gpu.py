"""
    Heavily based on: https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_1bn.py
"""


import sys
import time
import math
import faiss
import torch

import numpy as np
from colbert.utils.utils import print_message


class FaissIndexGPU():
    def __init__(self):
        self.ngpu = faiss.get_num_gpus()

        if self.ngpu == 0:
            return

        self.tempmem = 1 << 33
        self.max_add_per_gpu = 1 << 25
        self.max_add = self.max_add_per_gpu * self.ngpu
        self.add_batch_size = 65536

        self.gpu_resources = self._prepare_gpu_resources()

    def _prepare_gpu_resources(self):
        print_message(f"Preparing resources for {self.ngpu} GPUs.")

        gpu_resources = []

        for _ in range(self.ngpu):
            res = faiss.StandardGpuResources()
            if self.tempmem >= 0:
                res.setTempMemory(self.tempmem)
            gpu_resources.append(res)

        return gpu_resources

    def _make_vres_vdev(self):
        """
        return vectors of device ids and resources useful for gpu_multiple
        """

        assert self.ngpu > 0

        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()

        for i in range(self.ngpu):
            vdev.push_back(i)
            vres.push_back(self.gpu_resources[i])

        return vres, vdev

    def training_initialize(self, index, quantizer):
        """
        The index and quantizer should be owned by caller.
        """

        assert self.ngpu > 0

        s = time.time()
        self.index_ivf = faiss.extract_index_ivf(index)
        self.clustering_index = faiss.index_cpu_to_all_gpus(quantizer)
        self.index_ivf.clustering_index = self.clustering_index
        print(time.time() - s)

    def training_finalize(self):
        assert self.ngpu > 0

        s = time.time()
        self.index_ivf.clustering_index = faiss.index_gpu_to_cpu(self.index_ivf.clustering_index)
        print(time.time() - s)

    def adding_initialize(self, index):
        """
        The index should be owned by caller.
        """

        assert self.ngpu > 0

        self.co = faiss.GpuMultipleClonerOptions()
        self.co.useFloat16 = True
        self.co.useFloat16CoarseQuantizer = False
        self.co.usePrecomputed = False
        self.co.indicesOptions = faiss.INDICES_CPU
        self.co.verbose = True
        self.co.reserveVecs = self.max_add
        self.co.shard = True
        assert self.co.shard_type in (0, 1, 2)

        self.vres, self.vdev = self._make_vres_vdev()
        self.gpu_index = faiss.index_cpu_to_gpu_multiple(self.vres, self.vdev, index, self.co)

    def add(self, index, data, offset):
        assert self.ngpu > 0

        t0 = time.time()
        nb = data.shape[0]

        for i0 in range(0, nb, self.add_batch_size):
            i1 = min(i0 + self.add_batch_size, nb)
            xs = data[i0:i1]

            self.gpu_index.add_with_ids(xs, np.arange(offset+i0, offset+i1))

            if self.max_add > 0 and self.gpu_index.ntotal > self.max_add:
                self._flush_to_cpu(index, nb, offset)

            print('\r%d/%d (%.3f s)  ' % (i0, nb, time.time() - t0), end=' ')
            sys.stdout.flush()

        if self.gpu_index.ntotal > 0:
            self._flush_to_cpu(index, nb, offset)

        assert index.ntotal == offset+nb, (index.ntotal, offset+nb, offset, nb)
        print(f"add(.) time: %.3f s \t\t--\t\t index.ntotal = {index.ntotal}" % (time.time() - t0))

    def _flush_to_cpu(self, index, nb, offset):
        print("Flush indexes to CPU")

        for i in range(self.ngpu):
            index_src_gpu = faiss.downcast_index(self.gpu_index if self.ngpu == 1 else self.gpu_index.at(i))
            index_src = faiss.index_gpu_to_cpu(index_src_gpu)

            index_src.copy_subset_to(index, 0, offset, offset+nb)
            index_src_gpu.reset()
            index_src_gpu.reserveMemory(self.max_add)

        if self.ngpu > 1:
            try:
                self.gpu_index.sync_with_shard_indexes()
            except:
                self.gpu_index.syncWithSubIndexes()
