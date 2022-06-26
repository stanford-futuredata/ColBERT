import os
from collections import defaultdict
import jsonlines
import sys

from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig, RunConfig, Run
from colbert.data import Queries

CKPT = "/dfs/scratch0/okhattab/share/2021/checkpoints/msmarco.psg.kldR2.nway64.ib__colbert-400000"
STACKEXCHANGE_DATA_PATH = "/future/u/xmgui/testdata"
EXPERIMENT_DIR = "/future/u/xmgui/ColBERT/experiments"


def build_index(dataset_path):
    nbits = 2  # encode each dimension with 2 bits
    doc_maxlen = 300  # truncate passages at 300 tokens
    collection = os.path.join(dataset_path, 'cs224u.collection.tsv')
    index_name = f'e2etest.{nbits}bits.latest'
    experiment = (f"e2etest.nbits={nbits}",)

    with Run().context(RunConfig(nranks=4)):
        config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            nbits=nbits,
            root=EXPERIMENT_DIR,
            experiment=experiment,
        )
        indexer = Indexer(CKPT, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)

        config = ColBERTConfig(
            root=EXPERIMENT_DIR,
            experiment=experiment,
        )
        searcher = Searcher(
            index=index_name,
            config=config,
        )

        return searcher


def search(searcher, dataset_path, query_type, k):
    queries = os.path.join(
        dataset_path, f"questions.{query_type}.tsv"
    )
    queries = Queries(path=queries)
    ranking = searcher.search_all(queries, k=k)
    output_path = ranking.save(f"e2etest.{query_type}.ranking.tsv")
    return output_path


def evaluate_dataset(dataset_path, query_type, k, rankings_path):
    if not os.path.exists(rankings_path):
        print(f"[query_type={query_type}] Success@{k}: ???")
        return
    rankings = defaultdict(list)
    with open(rankings_path, "r") as f:
        for line in f:
            items = line.strip().split("\t")
            qid, pid, rank = items[:3]

            qid = int(qid)
            pid = int(pid)  # post id
            rank = int(rank)
            rankings[qid].append(pid)
            assert rank == len(rankings[qid])

    success = 0
    qas_path = os.path.join(dataset_path, f"qas.{query_type}.jsonl")

    # check if the generated ranking's top k intersects with the provided answer
    num_total_qids = 0
    with jsonlines.open(qas_path, mode="r") as f:
        for line in f:
            qid = int(line["qid"])
            num_total_qids += 1
            if qid not in rankings:
                print(f"WARNING: qid {qid} not found in {rankings_path}!", file=sys.stderr)
                continue
            answer_pids = set(line["answer_pids"])
            if len(set(rankings[qid][:k]).intersection(answer_pids)) > 0:
                success += 1
    print(
        f"[query_type={query_type}] "
        f"Success@{k}: {success / num_total_qids * 100:.1f}"
    )


def main():
    k = 5
    searcher = build_index(STACKEXCHANGE_DATA_PATH)
    for query_type in ["search", "forum"]:
        rankings_path = search(searcher, STACKEXCHANGE_DATA_PATH, query_type, k)
        evaluate_dataset(STACKEXCHANGE_DATA_PATH, query_type, k, rankings_path)


if __name__ == "__main__":
    main()
