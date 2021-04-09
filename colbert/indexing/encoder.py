import os
import time
import torch
import ujson
import numpy as np

import itertools
import threading
import queue

from colbert.modeling.inference import ModelInference
from colbert.evaluation.loaders import load_colbert
from colbert.utils.utils import print_message

from colbert.indexing.index_manager import IndexManager


class CollectionEncoder():
    def __init__(self, args, process_idx, num_processes):
        self.args = args
        self.collection = args.collection
        self.process_idx = process_idx
        self.num_processes = num_processes

        assert 0.5 <= args.chunksize <= 128.0
        max_bytes_per_file = args.chunksize * (1024*1024*1024)

        max_bytes_per_doc = (self.args.doc_maxlen * self.args.dim * 2.0)

        # Determine subset sizes for output
        minimum_subset_size = 10_000
        maximum_subset_size = max_bytes_per_file / max_bytes_per_doc
        maximum_subset_size = max(minimum_subset_size, maximum_subset_size)
        self.possible_subset_sizes = [int(maximum_subset_size)]

        self.print_main("#> Local args.bsize =", args.bsize)
        self.print_main("#> args.index_root =", args.index_root)
        self.print_main(f"#> self.possible_subset_sizes = {self.possible_subset_sizes}")

        self._load_model()
        self.indexmgr = IndexManager(args.dim)
        self.iterator = self._initialize_iterator()

    def _initialize_iterator(self):
        return open(self.collection)

    def _saver_thread(self):
        for args in iter(self.saver_queue.get, None):
            self._save_batch(*args)

    def _load_model(self):
        self.colbert, self.checkpoint = load_colbert(self.args, do_print=(self.process_idx == 0))
        self.colbert = self.colbert.cuda()
        self.colbert.eval()

        self.inference = ModelInference(self.colbert, amp=self.args.amp)

    def encode(self):
        self.saver_queue = queue.Queue(maxsize=3)
        thread = threading.Thread(target=self._saver_thread)
        thread.start()

        t0 = time.time()
        local_docs_processed = 0

        for batch_idx, (offset, lines, owner) in enumerate(self._batch_passages(self.iterator)):
            if owner != self.process_idx:
                continue

            t1 = time.time()
            batch = self._preprocess_batch(offset, lines)
            embs, doclens = self._encode_batch(batch_idx, batch)

            t2 = time.time()
            self.saver_queue.put((batch_idx, embs, offset, doclens))

            t3 = time.time()
            local_docs_processed += len(lines)
            overall_throughput = compute_throughput(local_docs_processed, t0, t3)
            this_encoding_throughput = compute_throughput(len(lines), t1, t2)
            this_saving_throughput = compute_throughput(len(lines), t2, t3)

            self.print(f'#> Completed batch #{batch_idx} (starting at passage #{offset}) \t\t'
                          f'Passages/min: {overall_throughput} (overall), ',
                          f'{this_encoding_throughput} (this encoding), ',
                          f'{this_saving_throughput} (this saving)')
        self.saver_queue.put(None)

        self.print("#> Joining saver thread.")
        thread.join()

    def _batch_passages(self, fi):
        """
        Must use the same seed across processes!
        """
        np.random.seed(0)

        offset = 0
        for owner in itertools.cycle(range(self.num_processes)):
            batch_size = np.random.choice(self.possible_subset_sizes)

            L = [line for _, line in zip(range(batch_size), fi)]

            if len(L) == 0:
                break  # EOF

            yield (offset, L, owner)
            offset += len(L)

            if len(L) < batch_size:
                break  # EOF

        self.print("[NOTE] Done with local share.")

        return

    def _preprocess_batch(self, offset, lines):
        endpos = offset + len(lines)

        batch = []

        for line_idx, line in zip(range(offset, endpos), lines):
            line_parts = line.strip().split('\t')

            pid, passage, *other = line_parts

            assert len(passage) >= 1

            if len(other) >= 1:
                title, *_ = other
                passage = title + ' | ' + passage

            batch.append(passage)

            assert pid == 'id' or int(pid) == line_idx

        return batch

    def _encode_batch(self, batch_idx, batch):
        with torch.no_grad():
            embs = self.inference.docFromText(batch, bsize=self.args.bsize, keep_dims=False)
            assert type(embs) is list
            assert len(embs) == len(batch)

            local_doclens = [d.size(0) for d in embs]
            embs = torch.cat(embs)

        return embs, local_doclens

    def _save_batch(self, batch_idx, embs, offset, doclens):
        start_time = time.time()

        output_path = os.path.join(self.args.index_path, "{}.pt".format(batch_idx))
        output_sample_path = os.path.join(self.args.index_path, "{}.sample".format(batch_idx))
        doclens_path = os.path.join(self.args.index_path, 'doclens.{}.json'.format(batch_idx))

        # Save the embeddings.
        self.indexmgr.save(embs, output_path)
        self.indexmgr.save(embs[torch.randint(0, high=embs.size(0), size=(embs.size(0) // 20,))], output_sample_path)

        # Save the doclens.
        with open(doclens_path, 'w') as output_doclens:
            ujson.dump(doclens, output_doclens)

        throughput = compute_throughput(len(doclens), start_time, time.time())
        self.print_main("#> Saved batch #{} to {} \t\t".format(batch_idx, output_path),
                        "Saving Throughput =", throughput, "passages per minute.\n")

    def print(self, *args):
        print_message("[" + str(self.process_idx) + "]", "\t\t", *args)

    def print_main(self, *args):
        if self.process_idx == 0:
            self.print(*args)


def compute_throughput(size, t0, t1):
    throughput = size / (t1 - t0) * 60

    if throughput > 1000 * 1000:
        throughput = throughput / (1000*1000)
        throughput = round(throughput, 1)
        return '{}M'.format(throughput)

    throughput = throughput / (1000)
    throughput = round(throughput, 1)
    return '{}k'.format(throughput)
