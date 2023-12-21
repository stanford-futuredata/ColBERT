import os
import argparse
from collections import namedtuple
from datasets import load_dataset
import tqdm

from colbert import Indexer, Searcher, IndexUpdater
from colbert.infra import ColBERTConfig, RunConfig, Run

SquadExample = namedtuple("SquadExample", "id title context question answers")

nbits = 1  # encode each dimension with 1 bits
doc_maxlen = 180  # truncate passages at 180 tokens
experiment = f"removetest.nbits={nbits}"


def build_index_and_init_searcher(checkpoint, collection, experiment_dir):
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
    checkpoint = args.checkpoint
    collection = args.collection
    experiment_dir = args.expdir

    # Set up the test
    k = 5
    searcher, index_path = build_index_and_init_searcher(
        checkpoint, collection, experiment_dir
    )
    squad = load_dataset("squad")
    squad_dev = get_squad_split(squad)
    question = squad_dev[10].question
    all_tests_passed = True

    # Search full index
    results = searcher.search(question, k=k)
    top_k_ids = []
    for passage_id, passage_rank, passage_score in zip(*results):
        top_k_ids.append(passage_id)
    print(f"Initial search top results: {top_k_ids}")

    # Initialize IndexUpdater
    config = ColBERTConfig(
        doc_maxlen=doc_maxlen,
        nbits=nbits,
        root=experiment_dir,
        experiment=experiment,
    )

    index_updater = IndexUpdater(config, searcher, checkpoint)


    # Try removing an invalid passage
    raised = False
    try:
        index_updater.remove([len(searcher.ranker.doclens)])
    except ValueError as e:
        raised = True
    assert raised

    # Remove the first passage from top-k results (no persisting to disk)
    n = 1
    index_updater.remove(top_k_ids[:n])

    # Search again without reloading the searcher
    results = searcher.search(question, k=k)
    top_k_ids_after = []
    for passage_id, passage_rank, passage_score in zip(*results):
        top_k_ids_after.append(passage_id)

    # Test that top passage is removed
    print(f"Top results after removing the top passage: {top_k_ids_after}")
    if top_k_ids[n:] == top_k_ids_after[:-n] and top_k_ids_after[-n:] != top_k_ids[:n]:
        print("REMOVAL SUCCEEDED")
    else:
        print(top_k_ids)
        print(top_k_ids_after)

        print("REMOVAL FAILED!!!")
        all_tests_passed = False

    # Add the removed passage back
    passage_removed = "Diego on May 24, 1984 during their May 23–25, 1984 meetings in Washington, D.C. This was the first Super Bowl to be played at Jack Murphy Stadium (now currently known as Qualcomm Stadium) in San Diego, California. Fourteen cities were part of the bidding process, which was scheduled to award four Super Bowls (XXI, XXII, XXIII, and XXIV). The bidding cities included: Anaheim, Detroit, Houston, Jacksonville, Miami, Minneapolis, New Orleans, Pasadena, Philadelphia, San Francisco, San Diego, Seattle, Tampa, and Tempe. The Philadelphia host committee assembled what was considered a strong, but long-shot bid, hoping to win the first outdoor Super"
    new_pids = index_updater.add([passage_removed])

    # Search again without reloading the searcher
    results = searcher.search(question, k=k)
    top_k_ids_after_append = []
    for passage_id, passage_rank, passage_score in zip(*results):
        top_k_ids_after_append.append(passage_id)

    print(f"Top results after appending back the top passage: {top_k_ids_after_append}")

    if (
        top_k_ids[1:] == top_k_ids_after_append[1:]
        and top_k_ids_after_append[0] == new_pids[0]
    ):
        print("REAPPEND SUCCEEDED")
    else:
        print("REAPPEND FAILED!!!")
        all_tests_passed = False

    # Add 4 more copies of the top passages
    new_pids.extend(
        index_updater.add(
            [passage_removed, passage_removed, passage_removed, passage_removed]
        )
    )

    # Search again without reloading the searcher
    # Now we expect the pids of the 5 newly added identical top passages to be returned as seearch results
    results = searcher.search(question, k=k)
    top_k_ids_after_append = []
    for passage_id, passage_rank, passage_score in zip(*results):
        top_k_ids_after_append.append(passage_id)

    print(
        f"Top results after appending 4 more copies of the top passage: {top_k_ids_after_append}"
    )

    if set(top_k_ids_after_append) == set(
        new_pids
    ):  # all results should now be the newly appended passages
        print("REAPPEND 4 more SUCCEEDED")
    else:
        print("REAPPEND 4 more FAILED!!!")
        all_tests_passed = False

    # Persist to disk, reload the searcher and search again

    index_updater.persist_to_disk()

    config = ColBERTConfig(
        root=experiment_dir,
        experiment=experiment,
    )
    searcher = Searcher(
        index=experiment,
        config=config,
    )
    results = searcher.search(question, k=k)
    top_k_ids_reload = []
    for passage_id, passage_rank, passage_score in zip(*results):
        top_k_ids_reload.append(passage_id)

    # Still we expect the new 5 identical ids of the top passage to be returned as seearch results

    print(
        f"Top results after appending 4 more copies of the top passage and reloading: {top_k_ids_reload}"
    )

    if set(top_k_ids_reload) == set(
        new_pids
    ):  # all results should now be the newly appended passages
        print("RELOAD SUCCEEDED")
    else:
        print("RELOAD FAILED!!!")
        all_tests_passed = False

    if all_tests_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED!!!")


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
