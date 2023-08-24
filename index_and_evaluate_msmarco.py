import csv
import numpy as np
import os
import torch
import tqdm
import time
import gc
from multiprocess import Pool

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries
from collections import defaultdict
from colbert import Indexer, Searcher

CKPT = "/home/ubuntu/msmarco.psg.kldR2.nway64.ib__colbert-400000"
# STACKEXCHANGE_DATA_PATH = "/future/u/keshav2/naacl22_lotte_datasets_final"
STACKEXCHANGE_DATA_PATH = "/future/u/keshav2/lotte"

STACKEXCHANGE_DATASETS = [
    # "writing",
    # "recreation",
    # "science",
    # "technology",
    #"lifestyle",
    "pooled",
]

def run(i, qvals_i_1, gold_ranks, searcher, ranker, config, times):
    gc.collect()
    t2 = time.time()
    print("Start time of {}: {}".format(i, t2))
    gr = torch.tensor(gold_ranks, dtype=torch.int)
    Q = searcher.encode([qvals_i_1])
    # score = ranker.score_raw_pids(config, Q[i:i+1], torch.tensor(list(docs)[:10000], dtype=torch.int))
    scores_, pids_ = ranker.score_raw_pids(config, Q, gr)
    scores_sorter = scores_.sort(descending=True)
    pids_, scores_ = pids_[scores_sorter.indices].tolist(), scores_sorter.values.tolist()
    print("Reranking time of {}: {}".format(i, time.time() - t2))
    times.append(time.time() - t2)
    del gr
    del scores_
    del pids_
    del scores_sorter
    del Q

    gc.collect()
    t2 = time.time()
    pids, ranks, scores = searcher.search(qvals_i_1, k=100)
    print("Search time of {}: {}".format(i, time.time() - t2))
    del pids
    del ranks
    del scores

def retrieve(searcher, dataset, dataset_path, split):
    for query_type in ["search"]:  # , "forum"]:
        queries = os.path.join(
            dataset_path, dataset, split, f"questions.{query_type}.tsv"
        )
        queries = Queries(path=queries)
        ranking = []
        elapsed = []
        encode_times = []
        candidate_generation_times = []
        decompression_times = []
        scoring_times = []

        use_gpu = torch.cuda.device_count() > 0

        for qid, query in tqdm.tqdm(list(queries.items())):
            # for qid, query in list(queries.items()):
            if qid not in splade_queries:
                continue

            if use_gpu:
                torch.cuda.synchronize()
            t0 = time.time()

            (
                #(
                    pids,
                    ranks,
                    scores,
                    #candidate_generation_time,
                    #decompression_time,
                    #scoring_time,
                #),
                #encode_time,
            ) = searcher.search(query, k=100)
            R = [
                (qid, pid, rank, score) for pid, rank, score in zip(pids, ranks, scores)
            ]
            ranking.extend(R)

            if use_gpu:
                torch.cuda.synchronize()
            elapsed.append(time.time() - t0)
            #encode_times.append(encode_time)
            # candidate_generation_times.append(candidate_generation_time)
            # decompression_times.append(decompression_time)
            # scoring_times.append(scoring_time)

        """
        print(
            f"query_type={query_type}, "
            f"candidate generation_time={np.mean(candidate_generation_times)}, "
            f"decompression time={np.mean(decompression_times)}, "
            f"scoring time={np.mean(scoring_times)}"
        )
        """
        #print(f"encode_time={np.mean(encode_time)}")
        print(f"query_type={query_type}, latency={np.mean(elapsed)}, stddev={np.std(elapsed)}")
        with open(f"{dataset}.{split}.ranking.tsv", "w") as f:
            for (qid, pid, rank, score) in ranking:
                f.write(f"{qid}\t{pid}\t{rank}\t{score}\n")


def evaluate(index=True):
    split = "dev"
    nbits = 2
    k = 1000
    nprobe = 4
    ncandidates = 2 ** 16 
    experiment = (f"msmarco.nbits={nbits}.latest.mmap",)
    collection = "/home/ubuntu/collection.tsv"

    gold_rankings_files = "/home/ubuntu/msmarco.k=1000.device=gpu.ranking.tsv"
    gold_ranks = defaultdict(list)
    docs = []

    with open(gold_rankings_files, newline='', encoding='utf-8') as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter='\t')
        for line in tsv_reader:
            gold_ranks[int(line[0])].append(int(line[1]))
            docs.append(int(line[1]))
    docs = set(docs)

    print(len(docs))
    
    if not os.path.exists(collection):
        print(f"No data found for {dataset} at {collection}, skipping...")
        return
    with Run().context(RunConfig(nranks=4)):
        INDEX_NAME = f"msmarco.nbits={nbits}.latest.mmap"

        if index:
            config = ColBERTConfig(
                doc_maxlen=300,
                nbits=nbits,
                kmeans_niters=4,
                root="/home/ubuntu/experiments",
                experiment=experiment,
            )
            indexer = Indexer(CKPT, config=config)
            indexer.index(name=INDEX_NAME, collection=collection, overwrite=True)

        config = ColBERTConfig(
            root="/home/ubuntu/experiments",
            experiment=experiment,
            #nprobe=nprobe,
            #ncandidates=ncandidates,
        )
        searcher = Searcher(
            index=INDEX_NAME,
            config=config,

        )
        ranker = searcher.ranker
        queries = "/home/ubuntu/queries.dev.small.tsv"
        #queries = "/lfs/1/keshav2/msmarco.queries.sample.tsv"
        #queries = "/lfs/1/kesxhav2/colbert/splade_queries.tsv"
        queries = Queries(path=queries)
        t1 = time.time()
        qvals = list(queries.items())
        print("Queries encoded in", time.time() - t1, "sec.")
        if torch.cuda.device_count() < 1:
            print(f"No GPU detected, setting num_threads to 1...")
            torch.set_num_threads(1)
            device = "cpu"
        else:
            device = "gpu"
        times = []
        args = []
        for i in range(len(qvals)):
            # args.append((i, qvals[i][1], gold_ranks[qvals[i][0]][:100], searcher, ranker, config, times))
            print(qvals[i])
            run(i, qvals[i][1], gold_ranks[qvals[i][0]][:100], searcher, ranker, config, times)
        # with Pool(processes=4) as pool:
        #     result = pool.starmap(run, args)
        # result.get()
        times = list(sorted(times))
        print(sum(times) / len(times), times[50], times[90], times[98])
        ranking = searcher.search_all(queries, k=k)
        ranking.save(f"msmarco.k={k}.device={device}.ranking.tsv")

def main():
    evaluate(index=False)


if __name__ == "__main__":
    main()
