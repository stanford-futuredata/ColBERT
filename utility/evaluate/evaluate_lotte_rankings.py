import argparse
from collections import defaultdict
import jsonlines
import os
import sys


def evaluate_dataset(query_type, dataset, split, k, data_rootdir, rankings_rootdir):
    data_path = os.path.join(data_rootdir, dataset, split)
    rankings_path = os.path.join(
        rankings_rootdir, split, f"{dataset}.{query_type}.ranking.tsv"
    )
    if not os.path.exists(rankings_path):
        print(f"[query_type={query_type}, dataset={dataset}] Success@{k}: ???")
        return
    rankings = defaultdict(list)
    with open(rankings_path, "r") as f:
        for line in f:
            items = line.strip().split("\t")
            qid, pid, rank = items[:3]
            qid = int(qid)
            pid = int(pid)
            rank = int(rank)
            rankings[qid].append(pid)
            assert rank == len(rankings[qid])

    success = 0
    qas_path = os.path.join(data_path, f"qas.{query_type}.jsonl")

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
        f"[query_type={query_type}, dataset={dataset}] "
        f"Success@{k}: {success / num_total_qids * 100:.1f}"
    )


def main(args):
    for query_type in ["search", "forum"]:
        for dataset in [
            "writing",
            "recreation",
            "science",
            "technology",
            "lifestyle",
            "pooled",
        ]:
            evaluate_dataset(
                query_type,
                dataset,
                args.split,
                args.k,
                args.data_dir,
                args.rankings_dir,
            )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoTTE evaluation script")
    parser.add_argument("--k", type=int, default=5, help="Success@k")
    parser.add_argument(
        "-s", "--split", choices=["dev", "test"], required=True, help="Split"
    )
    parser.add_argument(
        "-d", "--data_dir", type=str, required=True, help="Path to LoTTE data directory"
    )
    parser.add_argument(
        "-r",
        "--rankings_dir",
        type=str,
        required=True,
        help="Path to LoTTE rankings directory",
    )
    args = parser.parse_args()
    main(args)
