import numpy as np
import os
import torch
import tqdm
import time
import random

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries

from colbert import Indexer, Searcher, IndexUpdater

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
    experiment = (f"msmarco_1M.nbits={nbits}",)
    # collection = "/dfs/scratch1/okhattab/data/MSMARCO/collection.tsv"
    collection = "/future/u/xrsong/data/MSMARCO/collection.tsv"
    
    if not os.path.exists(collection):
        print(f"No data found for {dataset} at {collection}, skipping...")
        return
    with Run().context(RunConfig(nranks=4)):
        INDEX_NAME = f"msmarco_1M.nbits={nbits}.latest"

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
        questions = list(queries.values())[:1]
        
        if torch.cuda.device_count() < 1:
            print(f"No GPU detected, setting num_threads to 1...")
            torch.set_num_threads(1)
            device = "cpu"
        else:
            device = "gpu"
            
#         ranking = searcher.search_all(queries, k=k)
#         ranking.save(f"msmarco.k={k}.device={device}.ranking.tsv")

# Testing
# TODO:
    #0. Both test seems to be working -- need to find a way to enlarge the scale
    #1. Removal from ivf is SLOW (like 5min per pid...)
    #2. Code for removal break when I try to make pids into a set (line 270 index updater)
    #3. Add passages back to see if they are retrieved again
    
        # REMOVAL
        for question in questions:
            results = searcher.search(question, k=k)
            top_k_ids = []
            for passage_id, passage_rank, passage_score in zip(*results):
                top_k_ids.append(passage_id)
            
            # remove m passages from top_k passages
            removed_pids = random.choices(top_k_ids, k=100)
            config = ColBERTConfig(
                doc_maxlen=128,
                nbits=nbits,
                kmeans_niters=4,
                root="/future/u/xrsong/ColBERT/experiments",
                experiment=experiment,
            )
            index_updater = IndexUpdater(config, searcher, CKPT)
            index_updater.remove(removed_pids)
            # search for the top_k passages again
            results = searcher.search(question, k=k)
            top_k_ids_after_remove = []
            for passage_id, passage_rank, passage_score in zip(*results):
                top_k_ids_after_remove.append(passage_id)
            # check that the removed passages do not appear in new results
            print(f'Pids to be removed are: {removed_pids}')
            intersection = list(set(removed_pids) & set(top_k_ids_after_remove))
            print(f'Intersection between new results and removed pids is: {intersection}')
            assert len(intersection) == 0
            print('REMOVAL SUCCEEDED')
            
        # ADD
        new_queries = "/future/u/xrsong/data/MSMARCO/queries.tsv"
        new_queries = Queries(path=new_queries)
        new_queries = list(new_queries.values())
        index_updater = IndexUpdater(config, searcher, CKPT)
        pids = index_updater.add(new_queries)
        print(f'Added new passages with the following pids: {pids}')
    
        for i in range(len(new_queries)):
            pid = pids[i]
            print(f'Searching with passage {pid} as query...')
            question = new_queries[i]
            results = searcher.search(question, k=100)
            top_k_ids = []
            for passage_id, passage_rank, passage_score in zip(*results):
                top_k_ids.append(passage_id)
            print(f'Top 5 results for query {pid} are {top_k_ids[:5]}.')
            assert pid in top_k_ids
        print('ADD SUCCEEDED')
            
def main():
    evaluate(index=False)


if __name__ == "__main__":
    main()
