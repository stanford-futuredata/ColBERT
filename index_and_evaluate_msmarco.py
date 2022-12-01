import numpy as np
import os
import torch
import tqdm
import time

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries

from colbert import Indexer, Searcher

CKPT = "/dfs/scratch0/okhattab/share/2021/checkpoints/msmarco.psg.kldR2.nway64.ib__colbert-400000"
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

def evaluate(index=True):
    split = "dev"
    nbits = 2
    k = 1000
    experiment = (f"msmarco.nbits={nbits}",)
    collection = "/dfs/scratch1/okhattab/data/MSMARCO/collection.tsv"
    
    if not os.path.exists(collection):
        print(f"No data found for {dataset} at {collection}, skipping...")
        return
    with Run().context(RunConfig(nranks=4)):
        INDEX_NAME = f"msmarco.nbits={nbits}.latest"

        if index:
            config = ColBERTConfig(
                doc_maxlen=128,
                nbits=nbits,
                kmeans_niters=4,
                root="/future/u/xrsong/ColBERT/experiments",
                experiment=experiment,
            )
            indexer = Indexer(CKPT, config=config)
            indexer.index(name=INDEX_NAME, collection=collection, overwrite=True)

        config = ColBERTConfig(
            root="/future/u/xrsong/ColBERT/experiments",
            experiment=experiment,
        )
        searcher = Searcher(
            index=INDEX_NAME,
            config=config,

        )
        queries = "/dfs/scratch1/okhattab/data/MSMARCO/queries.dev.small.tsv"
        queries = Queries(path=queries)
        if torch.cuda.device_count() < 1:
            print(f"No GPU detected, setting num_threads to 1...")
            torch.set_num_threads(1)
            device = "cpu"
        else:
            device = "gpu"
        ranking = searcher.search_all(queries, k=k)
        ranking.save(f"msmarco.k={k}.device={device}.ranking.tsv")

def main():
    evaluate(index=True)


if __name__ == "__main__":
    main()
