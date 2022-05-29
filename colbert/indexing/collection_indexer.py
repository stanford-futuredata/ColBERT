import os
import tqdm
import time
import ujson
import torch
import random
try:
    import faiss
except ImportError as e:
    print("WARNING: faiss must be imported for indexing")

import numpy as np
import torch.multiprocessing as mp
from colbert.infra.config.config import ColBERTConfig

import colbert.utils.distributed as distributed

from colbert.infra.run import Run
from colbert.infra.launcher import print_memory_stats
from colbert.modeling.checkpoint import Checkpoint
from colbert.data.collection import Collection

from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.indexing.index_saver import IndexSaver
from colbert.indexing.utils import optimize_ivf
from colbert.utils.utils import flatten, print_message

from colbert.indexing.codecs.residual import ResidualCodec


def encode(config, collection, shared_lists, shared_queues):
    encoder = CollectionIndexer(config=config, collection=collection)
    encoder.run(shared_lists)


class CollectionIndexer():
    def __init__(self, config: ColBERTConfig, collection):
        self.config = config
        self.rank, self.nranks = self.config.rank, self.config.nranks

        if self.config.rank == 0:
            self.config.help()

        self.collection = Collection.cast(collection)
        self.checkpoint = Checkpoint(self.config.checkpoint, colbert_config=self.config).cuda()

        self.encoder = CollectionEncoder(config, self.checkpoint)
        self.saver = IndexSaver(config)

        print_memory_stats(f'RANK:{self.rank}')

    def run(self, shared_lists):
        with torch.inference_mode():
            self.setup()
            distributed.barrier(self.rank)
            print_memory_stats(f'RANK:{self.rank}')

            self.train(shared_lists)
            distributed.barrier(self.rank)
            print_memory_stats(f'RANK:{self.rank}')

            self.index()
            distributed.barrier(self.rank)
            print_memory_stats(f'RANK:{self.rank}')

            self.finalize()
            distributed.barrier(self.rank)
            print_memory_stats(f'RANK:{self.rank}')

    def setup(self):
        self.num_chunks = int(np.ceil(len(self.collection) / self.collection.get_chunksize()))

        sampled_pids = self._sample_pids()
        avg_doclen_est = self._sample_embeddings(sampled_pids)

        # Select the number of partitions
        num_passages = len(self.collection)
        self.num_embeddings_est = num_passages * avg_doclen_est
        self.num_partitions = int(2 ** np.floor(np.log2(16 * np.sqrt(self.num_embeddings_est))))

        Run().print_main(f'Creaing {self.num_partitions:,} partitions.')
        Run().print_main(f'*Estimated* {int(self.num_embeddings_est):,} embeddings.')

        self._save_plan()

    def _sample_pids(self):
        num_passages = len(self.collection)

        # Simple alternative: < 100k: 100%, < 1M: 15%, < 10M: 7%, < 100M: 3%, > 100M: 1%
        # Keep in mind that, say, 15% still means at least 100k.
        # So the formula is max(100% * min(total, 100k), 15% * min(total, 1M), ...)
        # Then we subsample the vectors to 100 * num_partitions

        typical_doclen = 120  # let's keep sampling independent of the actual doc_maxlen
        sampled_pids = 16 * np.sqrt(typical_doclen * num_passages)
        # sampled_pids = int(2 ** np.floor(np.log2(1 + sampled_pids)))
        sampled_pids = min(1 + int(sampled_pids), num_passages)

        sampled_pids = random.sample(range(num_passages), sampled_pids)
        Run().print_main(f"# of sampled PIDs = {len(sampled_pids)} \t sampled_pids[:3] = {sampled_pids[:3]}")

        return set(sampled_pids)

    def _sample_embeddings(self, sampled_pids):
        local_pids = self.collection.enumerate(rank=self.rank)
        local_sample = [passage for pid, passage in local_pids if pid in sampled_pids]

        local_sample_embs, doclens = self.encoder.encode_passages(local_sample)

        self.num_sample_embs = torch.tensor([local_sample_embs.size(0)]).cuda()
        torch.distributed.all_reduce(self.num_sample_embs)

        avg_doclen_est = sum(doclens) / len(doclens) if doclens else 0
        avg_doclen_est = torch.tensor([avg_doclen_est]).cuda()
        torch.distributed.all_reduce(avg_doclen_est)

        nonzero_ranks = torch.tensor([float(len(local_sample) > 0)]).cuda()
        torch.distributed.all_reduce(nonzero_ranks)

        avg_doclen_est = avg_doclen_est.item() / nonzero_ranks.item()
        self.avg_doclen_est = avg_doclen_est

        Run().print(f'avg_doclen_est = {avg_doclen_est} \t len(local_sample) = {len(local_sample):,}')

        torch.save(local_sample_embs, os.path.join(self.config.index_path_, f'sample.{self.rank}.pt'))

        return avg_doclen_est

    def _save_plan(self):
        if self.rank < 1:
            config = self.config
            self.plan_path = os.path.join(config.index_path_, 'plan.json')
            Run().print("#> Saving the indexing plan to", self.plan_path, "..")

            with open(self.plan_path, 'w') as f:
                d = {'config': config.export()}
                d['num_chunks'] = self.num_chunks
                d['num_partitions'] = self.num_partitions
                d['num_embeddings_est'] = self.num_embeddings_est
                d['avg_doclen_est'] = self.avg_doclen_est

                f.write(ujson.dumps(d, indent=4) + '\n')

    def train(self, shared_lists):
        if self.rank > 0:
            return

        sample, heldout = self._concatenate_and_split_sample()

        centroids = self._train_kmeans(sample, shared_lists)

        print_memory_stats(f'RANK:{self.rank}')
        del sample

        bucket_cutoffs, bucket_weights, avg_residual = self._compute_avg_residual(centroids, heldout)

        print_message(f'avg_residual = {avg_residual}')

        codec = ResidualCodec(config=self.config, centroids=centroids, avg_residual=avg_residual,
                              bucket_cutoffs=bucket_cutoffs, bucket_weights=bucket_weights)
        self.saver.save_codec(codec)

    def _concatenate_and_split_sample(self):
        print_memory_stats(f'***1*** \t RANK:{self.rank}')

        # TODO: Allocate a float16 array. Load the samples from disk, copy to array.
        sample = torch.empty(self.num_sample_embs, self.config.dim, dtype=torch.float16)

        offset = 0
        for r in range(self.nranks):
            sub_sample_path = os.path.join(self.config.index_path_, f'sample.{r}.pt')
            sub_sample = torch.load(sub_sample_path)
            os.remove(sub_sample_path)

            endpos = offset + sub_sample.size(0)
            sample[offset:endpos] = sub_sample
            offset = endpos
        
        assert endpos == sample.size(0), (endpos, sample.size())

        print_memory_stats(f'***2*** \t RANK:{self.rank}')

        # Shuffle and split out a 5% "heldout" sub-sample [up to 50k elements]
        sample = sample[torch.randperm(sample.size(0))]

        print_memory_stats(f'***3*** \t RANK:{self.rank}')

        sample, sample_heldout = sample.split(int(sample.size(0) - min(.5 * sample.size(0), 50_000)), dim=0)

        print_memory_stats(f'***4*** \t RANK:{self.rank}')

        return sample, sample_heldout

    def _train_kmeans(self, sample, shared_lists):
        torch.cuda.empty_cache()

        do_fork_for_faiss = False  # set to True to free faiss GPU-0 memory at the cost of one more copy of `sample`.

        args_ = [self.config.dim, self.num_partitions, self.config.kmeans_niters]

        if do_fork_for_faiss:
            # For this to work reliably, write the sample to disk. Pickle may not handle >4GB of data.
            # Delete the sample file after work is done.

            shared_lists[0][0] = sample
            return_value_queue = mp.Queue()

            args_ = args_ + [shared_lists, return_value_queue]
            proc = mp.Process(target=compute_faiss_kmeans, args=args_)

            proc.start()
            centroids = return_value_queue.get()
            proc.join()

        else:
            args_ = args_ + [[[sample]]]
            centroids = compute_faiss_kmeans(*args_)

        centroids = torch.nn.functional.normalize(centroids, dim=-1).half()

        return centroids

    def _compute_avg_residual(self, centroids, heldout):
        compressor = ResidualCodec(config=self.config, centroids=centroids, avg_residual=None)

        heldout_reconstruct = compressor.compress_into_codes(heldout, out_device='cuda')
        heldout_reconstruct = compressor.lookup_centroids(heldout_reconstruct, out_device='cuda')
        heldout_avg_residual = heldout.cuda() - heldout_reconstruct

        avg_residual = torch.abs(heldout_avg_residual).mean(dim=0).cpu()
        print([round(x, 3) for x in avg_residual.squeeze().tolist()])

        num_options = 2 ** self.config.nbits
        quantiles = torch.arange(0, num_options, device=heldout_avg_residual.device) * (1 / num_options)
        bucket_cutoffs_quantiles, bucket_weights_quantiles = quantiles[1:], quantiles + (0.5 / num_options)

        bucket_cutoffs = heldout_avg_residual.float().quantile(bucket_cutoffs_quantiles)
        bucket_weights = heldout_avg_residual.float().quantile(bucket_weights_quantiles)

        print_message(
            f"#> Got bucket_cutoffs_quantiles = {bucket_cutoffs_quantiles} and bucket_weights_quantiles = {bucket_weights_quantiles}")
        print_message(f"#> Got bucket_cutoffs = {bucket_cutoffs} and bucket_weights = {bucket_weights}")

        return bucket_cutoffs, bucket_weights, avg_residual.mean()

        # EVENTAULLY: Compare the above with non-heldout sample. If too different, we can do better!
        # sample = sample[subsample_idxs]
        # sample_reconstruct = get_centroids_for(centroids, sample)
        # sample_avg_residual = (sample - sample_reconstruct).mean(dim=0)

    def index(self):
        with self.saver.thread():
            batches = self.collection.enumerate_batches(rank=self.rank)
            for chunk_idx, offset, passages in tqdm.tqdm(batches, disable=self.rank > 0):
                embs, doclens = self.encoder.encode_passages(passages)
                assert embs.dtype == torch.float16

                Run().print_main(f"#> Saving chunk {chunk_idx}: \t {len(passages):,} passages "
                                 f"and {embs.size(0):,} embeddings. From #{offset:,} onward.")

                self.saver.save_chunk(chunk_idx, offset, embs, doclens)
                del embs, doclens

    def finalize(self):
        if self.rank > 0:
            return

        self._check_all_files_are_saved()
        self._collect_embedding_id_offset()

        self._build_ivf()
        self._update_metadata()

    def _check_all_files_are_saved(self):
        for chunk_idx in range(self.num_chunks):
            # EVENTUALLY: Check those files!
            pass

    def _collect_embedding_id_offset(self):
        passage_offset = 0
        embedding_offset = 0

        self.embedding_offsets = []

        for chunk_idx in range(self.num_chunks):
            metadata_path = os.path.join(self.config.index_path_, f'{chunk_idx}.metadata.json')

            with open(metadata_path) as f:
                chunk_metadata = ujson.load(f)

                chunk_metadata['embedding_offset'] = embedding_offset
                self.embedding_offsets.append(embedding_offset)

                assert chunk_metadata['passage_offset'] == passage_offset, (chunk_idx, passage_offset, chunk_metadata)

                passage_offset += chunk_metadata['num_passages']
                embedding_offset += chunk_metadata['num_embeddings']

            with open(metadata_path, 'w') as f:
                f.write(ujson.dumps(chunk_metadata, indent=4) + '\n')

        self.num_embeddings = embedding_offset
        assert len(self.embedding_offsets) == self.num_chunks

    def _build_ivf(self):
        # Maybe we should several small IVFs? Every 250M embeddings, so that's every 1 GB.
        # It would save *memory* here and *disk space* regarding the int64.
        # But we'd have to decide how many IVFs to use during retrieval: many (loop) or one?
        # A loop seems nice if we can find a size that's large enough for speed yet small enough to fit on GPU!
        # Then it would help nicely for batching later: 1GB.

        codes = torch.empty(self.num_embeddings,)
        print_memory_stats(f'RANK:{self.rank}')

        for chunk_idx in range(self.num_chunks):
            offset = self.embedding_offsets[chunk_idx]
            chunk_codes = ResidualCodec.Embeddings.load_codes(self.config.index_path_, chunk_idx, mmap_index=self.config.mmap_index)

            codes[offset:offset+chunk_codes.size(0)] = chunk_codes
        
        assert offset+chunk_codes.size(0) == codes.size(0), (offset, chunk_codes.size(0), codes.size()) 


        print_memory_stats(f'RANK:{self.rank}')

        codes = codes.sort()
        ivf, values = codes.indices, codes.values

        print_memory_stats(f'RANK:{self.rank}')

        partitions, ivf_lengths = values.unique_consecutive(return_counts=True)

        # All partitions should be non-empty. (We can use torch.histc otherwise.)
        assert partitions.size(0) == self.num_partitions, (partitions.size(), self.num_partitions)

        # ivf = [[] for _ in range(self.num_partitions)]
        # for chunk_idx in range(self.num_chunks):
        #     embedding_offset = self.embedding_offsets[chunk_idx]
        #     chunk_codes = ResidualCodec.Embeddings.load_codes(self.config.index_path_, chunk_idx)

        #     for local_embedding_idx, partition_idx in enumerate(chunk_codes.tolist()):
        #         embedding_idx = embedding_offset + local_embedding_idx
        #         ivf[partition_idx].append(embedding_idx)
        
        # ivf_lengths = [len(partition) for partition in ivf]
        # ivf = torch.tensor(flatten(ivf), dtype=torch.long)  # FIXME: If num_embeddings > 2B, use torch.long!

        print_memory_stats(f'RANK:{self.rank}')

        _, _ = optimize_ivf(ivf, ivf_lengths, self.config.index_path_)

    def _update_metadata(self):
        config = self.config
        self.metadata_path = os.path.join(config.index_path_, 'metadata.json')
        Run().print("#> Saving the indexing metadata to", self.metadata_path, "..")

        with open(self.metadata_path, 'w') as f:
            d = {'config': config.export()}
            d['num_chunks'] = self.num_chunks
            d['num_partitions'] = self.num_partitions
            d['num_embeddings'] = self.num_embeddings
            d['avg_doclen'] = self.num_embeddings / len(self.collection)

            f.write(ujson.dumps(d, indent=4) + '\n')


def compute_faiss_kmeans(dim, num_partitions, kmeans_niters, shared_lists, return_value_queue=None):
    kmeans = faiss.Kmeans(dim, num_partitions, niter=kmeans_niters, gpu=True, verbose=True, seed=123)

    sample = shared_lists[0][0]
    sample = sample.float().numpy()
    kmeans.train(sample)

    centroids = torch.from_numpy(kmeans.centroids)

    print_memory_stats(f'RANK:0*')

    if return_value_queue is not None:
        return_value_queue.put(centroids)

    return centroids


"""
TODOs:

1. Notice we're using self.config.bsize.

2. Consider saving/using heldout_avg_residual as a vector --- that is, using 128 averages!

3. Consider the operations with .cuda() tensors. Are all of them good for OOM?
"""
