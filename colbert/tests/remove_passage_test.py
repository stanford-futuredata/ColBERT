import os
import argparse
from collections import namedtuple
from datasets import load_dataset
import tqdm

from colbert import Indexer, Searcher, IncrementalIndexer
from colbert.infra import ColBERTConfig, RunConfig, Run

SquadExample = namedtuple("SquadExample", "id title context question answers")


def build_index_and_init_searcher(checkpoint, collection, experiment_dir):
    nbits = 1  # encode each dimension with 1 bits
    doc_maxlen = 180  # truncate passages at 180 tokens
    experiment = f"removetest.nbits={nbits}"

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
    incremental_indexer = IncrementalIndexer(index_path)
    incremental_indexer.remove_passage(pid)
    return

def main(args):
    checkpoint = args.checkpoint
    collection = args.collection
    experiment_dir = args.expdir

    # Set up the test
    k = 5
    searcher, index_path = build_index_and_init_searcher(checkpoint, collection, experiment_dir)
#     squad = load_dataset("squad")
#     squad_dev = get_squad_split(squad)
#     question = squad_dev[0].question
    question = "Which NFL team represented the AFC at Super Bowl 50?" # first question in squad's train set
    
    # Search full index
    results = searcher.search(question, k=k)
    top_k_ids = []
    for passage_id, passage_rank, passage_score in zip(*results):
        top_k_ids.append(passage_id)
        print(f"\t [{passage_id}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")
    
    # Remove top passage
    remove_passage(index_path, top_k_ids[0])
    
    # Search after removal
    results = searcher.search(question, k=k)
    top_k_ids_after = []
    for passage_id, passage_rank, passage_score in zip(*results):
        top_k_ids.append(passage_id)
#         print(f"\t [{passage_id}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")
    
    # Test that top passage is removed
    if top_k_ids[1:] == top_k_ids_after[:-1] and top_k_ids_after[-1] != top_k_ids[0]:
        print("Removal SUCCEEDED")
    else:
        print("Removal FAILED!!!")

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
