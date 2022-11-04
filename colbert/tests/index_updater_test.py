# Current command to run test: 
# python -m colbert.tests.index_updater_test --checkpoint "/dfs/scratch0/okhattab/share/2021/checkpoints/msmarco.psg.kldR2.nway64.ib__colbert-400000" --expdir ./experiments --collection /future/u/xrsong/ColBERT/docs/data/collection.small.tsv


import os
import argparse
from collections import namedtuple
from datasets import load_dataset
import tqdm

from colbert import Indexer, Searcher, PassageRemover, PassageAppender, IndexUpdater
from colbert.infra import ColBERTConfig, RunConfig, Run

SquadExample = namedtuple("SquadExample", "id title context question answers")

nbits = 1  # encode each dimension with 1 bits
doc_maxlen = 180  # truncate passages at 180 tokens
experiment = f"removetest.nbits={nbits}"

def build_index_and_init_searcher(checkpoint, collection, experiment_dir):
#     nbits = 1  # encode each dimension with 1 bits
#     doc_maxlen = 180  # truncate passages at 180 tokens
#     experiment = f"removetest.nbits={nbits}"

    with Run().context(RunConfig(nranks=1)):
        config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            nbits=nbits,
            root=experiment_dir,
            experiment=experiment,
        )
        indexer = Indexer(checkpoint, config=config)
        indexer.index(name=experiment, collection=collection, overwrite=True)
        index_path = indexer.get_index()

        config = ColBERTConfig(
            root=experiment_dir,
            experiment=experiment,
        )
        searcher = Searcher(
            index=experiment,
            config=config,
        )

        return searcher, index_path

    
def get_squad_split(squad, split="validation"):
    fields = squad[split].features
    data = zip(*[squad[split][field] for field in fields])
    return [
        SquadExample(eid, title, context, question, answers["text"])
        for eid, title, context, question, answers in data
    ]

def main(args):
    
    '''
    new api
    
    searcher, index_path = build_index_and_init_searcher(checkpoint, collection, experiment_dir)
    results = searcher.search(question, k=k)
    top_k_ids = []
    for passage_id, passage_rank, passage_score in zip(*results):
        top_k_ids.append(passage_id)
        print(f"\t [{passage_id}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")
        
    updater = IndexUpdater(searcher, checkpoint)
    updater.remove(top_k_ids)
    
    results = searcher.search(question, k=k)
    TODO: perform checks compare results
    
    updater.add(passages)
    TODO: perform checks compare results
    
    '''
    checkpoint = args.checkpoint
    collection = args.collection
    experiment_dir = args.expdir

# Set up the test
    k = 5
    searcher, index_path = build_index_and_init_searcher(checkpoint, collection, experiment_dir)
    squad = load_dataset("squad")
    squad_dev = get_squad_split(squad)
    question = squad_dev[10].question
    
# Search full index
    results = searcher.search(question, k=k)
    top_k_ids = []
    for passage_id, passage_rank, passage_score in zip(*results):
        top_k_ids.append(passage_id)
        print(f"\t [{passage_id}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")
        
# Initialize IndexUpdater
    config = ColBERTConfig(
        doc_maxlen=doc_maxlen,
        nbits=nbits,
        root=experiment_dir,
        experiment=experiment,
    )
    
    index_updater = IndexUpdater(config, searcher)
    
# Remove n passages from top-k results (no persisting to disk)
    n = 3
    index_updater.remove(top_k_ids[:n], persist_to_disk=False)
    
# Search again without reloading the searcher
    results = searcher.search(question, k=k)
    top_k_ids_after = []
    for passage_id, passage_rank, passage_score in zip(*results):
        top_k_ids_after.append(passage_id)

# Test that top passage is removed
    if top_k_ids[n:] == top_k_ids_after[:-n] and top_k_ids_after[-n:] != top_k_ids[:n]:
        print("Removal SUCCEEDED")
    else:
        print(top_k_ids)
        print(top_k_ids_after)
        
        print("Removal FAILED!!!")
        
# Reload the searcher and search again
    config = ColBERTConfig(
            root=experiment_dir,
            experiment=experiment,
        )
    searcher = Searcher(
            index=experiment,
            config=config,
        )
    results = searcher.search(question, k=k)
    top_k_ids_after = []
    for passage_id, passage_rank, passage_score in zip(*results):
        top_k_ids_after.append(passage_id)

# Now we expect that nothing is removed from disk
    if top_k_ids == top_k_ids_after:
        print("Disk data INTACT")
    else:
        print("Disk data CHANGED !!!")
        
# Remove passages again with persisting to disk   
    index_updater.remove(top_k_ids[:n], persist_to_disk=True)
    
# Reload the searcher and search again
    config = ColBERTConfig(
            root=experiment_dir,
            experiment=experiment,
        )
    searcher = Searcher(
            index=experiment,
            config=config,
        )
    results = searcher.search(question, k=k)
    top_k_ids_after = []
    for passage_id, passage_rank, passage_score in zip(*results):
        top_k_ids_after.append(passage_id)
        
# Now passages should also have been removed in disk
    if top_k_ids[n:] == top_k_ids_after[:-n] and top_k_ids_after[-n:] != top_k_ids[:n]:
        print("Removal from disk SUCCEEDED")
    else:
        print(top_k_ids)
        print(top_k_ids_after)
        
        print("Removal from disk FAILED!!!")
        

# TODO: Add testing for re-appending passages

    print("THE END")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start end-to-end test.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint"
    )
    parser.add_argument(
        "--collection", type=str, required=True, help="Path to collection"
    )
    parser.add_argument(
        "--expdir", type=str, required=True, help="Experiment directory"
    )
    args = parser.parse_args()
    main(args)
