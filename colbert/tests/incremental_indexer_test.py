# Current command to run test: 
# python -m colbert.tests.incremental_indexer_test --checkpoint "/dfs/scratch0/okhattab/share/2021/checkpoints/msmarco.psg.kldR2.nway64.ib__colbert-400000" --expdir ./experiments --collection /future/u/xrsong/ColBERT/docs/data/collection.small.tsv


import os
import argparse
from collections import namedtuple
from datasets import load_dataset
import tqdm

from colbert import Indexer, Searcher, PassageRemover, PassageAppender
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


def remove_passage(index_path, pid):
    incremental_indexer = PassageAppender(index_path)
    incremental_indexer.remove_passage(pid)
    return

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

#     Set up the test
    k = 5
    searcher, index_path = build_index_and_init_searcher(checkpoint, collection, experiment_dir)
    squad = load_dataset("squad")
    squad_dev = get_squad_split(squad)
    question = squad_dev[10].question
#     question = "Which NFL team represented the AFC at Super Bowl 50?" # first question in squad's train set
    
    # Search full index
    results = searcher.search(question, k=k)
    top_k_ids = []
    for passage_id, passage_rank, passage_score in zip(*results):
        top_k_ids.append(passage_id)
        print(f"\t [{passage_id}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")
    
#     Remove top passage
    passage_remover = PassageRemover(index_path)
    passage_remover.remove_passage(top_k_ids[0])
    
    # Error input check: remove invalid pid
#     remove_passage(index_path, 10000)
    
    # Search after removal
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
#         print(f"\t [{passage_id}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")
    
    # Test that top passage is removed
    if top_k_ids[1:] == top_k_ids_after[:-1] and top_k_ids_after[-1] != top_k_ids[0]:
        print("Removal SUCCEEDED")
    else:
        print("Removal FAILED!!!")
        
        
    # Re-append the removed passage
    config = ColBERTConfig(
        doc_maxlen=doc_maxlen,
        nbits=nbits,
        root=experiment_dir,
        experiment=experiment,
    )
    index_path = "/future/u/xrsong/ColBERT/experiments/default/indexes/removetest.nbits=1"
    passage_appender = PassageAppender(checkpoint, index_path, config)
    
    passage_removed = "Diego on May 24, 1984 during their May 23–25, 1984 meetings in Washington, D.C. This was the first Super Bowl to be played at Jack Murphy Stadium (now currently known as Qualcomm Stadium) in San Diego, California. Fourteen cities were part of the bidding process, which was scheduled to award four Super Bowls (XXI, XXII, XXIII, and XXIV). The bidding cities included: Anaheim, Detroit, Houston, Jacksonville, Miami, Minneapolis, New Orleans, Pasadena, Philadelphia, San Francisco, San Diego, Seattle, Tampa, and Tempe. The Philadelphia host committee assembled what was considered a strong, but long-shot bid, hoping to win the first outdoor Super"
    new_pid = passage_appender.append_passage(passage_removed)
    
    # search after re-append
    # Search after removal
    config = ColBERTConfig(
            root=experiment_dir,
            experiment=experiment,
        )
    searcher = Searcher(
            index=experiment,
            config=config,
        )
    results = searcher.search(question, k=k)
    top_k_ids_reappend = []
    for passage_id, passage_rank, passage_score in zip(*results):
        top_k_ids_reappend.append(passage_id)
#         print(f"\t [{passage_id}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")
    
    # Test that top passage is reappended
    print(top_k_ids_reappend)
    
    if top_k_ids[1:] == top_k_ids_reappend[1:] and top_k_ids_reappend[0] == new_pid:
        print("Re-append SUCCEEDED")
    else:
        print("Re-append FAILED!!!")

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
