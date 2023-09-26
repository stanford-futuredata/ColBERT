"""
    Evaluate Stack Overflow Passages ranking.
"""

import json
import pytrec_eval
from argparse import ArgumentParser
from colbert.utils.utils import print_message

def main(args):
    print_message(f"Stating evaluation for {args.distillation_score}..")
    # Load qrels which is a json file
    qrels = json.load(open(args.qrels, 'r'))
    print_message("Loaded qrels ..")

    score = {}
    if args.distillation_score is not None:
        # Load the file line-by-line and attempt to parse each line as a JSON object
        data_list = []
        with open(args.distillation_score, "r") as f:
            for line in f:
                data_list.append(json.loads(line))

        # Process the loaded data to the desired format
        for item in data_list:
            qid = str(item[0])
            scores = item[1]
            pid_score_dict = {str(score[1]): score[0] for score in scores}
            if qid in qrels.keys():
                score[qid] = pid_score_dict

        print_message("Loaded distillation scores ..")
    
    elif args.ranking:
        with open(args.ranking, "r") as f:
            for line in f:
                qid, pid, _, points = line.strip().split("\t")

                qid = str(qid)
                pid = str(pid)
                points = float(points)

                if qid in qrels.keys():
                    if qid not in score.keys():
                        score[qid] = {}

                    score[qid][str(pid)] = points
    
    elif args.ranx_score is not None:
        # No need to load the file line-by-line and directly parse it as a JSON object
        score = json.load(open(args.ranx_score, 'r'))

        print_message("Loaded ranx scores ..")
    
    else:
        raise ValueError("No score file provided. Make sure to provide either distillation score or ranx score in json format.")

    # assert that the qrels keys is a subset of distillation scores keys
    qrels_keys = set(qrels.keys())
    score_keys = set(score.keys())

    assert qrels_keys.issubset(score_keys), "qrels keys is not a subset of distillation scores keys"

    print_message("#> Evaluating Success@k ..")
    depth = [5, 10, 20, 100]

    for k in depth:
        success_at_k = 0
        total_pids = 0
        for qid in qrels.keys():
            qrel_pids = set(qrels[qid].keys())
            top_k_pids = sorted(score[qid].items(), key=lambda x: x[1], reverse=True)[:k]
            top_k_pids = set([pid for pid, _ in top_k_pids])
            
            common_pids = qrel_pids.intersection(top_k_pids)
            success_at_k += len(common_pids) > 0
            total_pids += len(qrel_pids)
        success_at_k = success_at_k / len(qrels.keys()) if total_pids > 0 else 0
        print_message(f"#> Success@{k}: {success_at_k}")    
    
    print_message("#> Evaluating Recall ..")
    depth = [10, 20, 30, 100]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recall'})
    scores = evaluator.evaluate(score)

    for d in depth:
        avg_score = sum([scores[qid][f'recall_{d}'] for qid in scores.keys()])/len(scores.keys())
        print_message(f"#> Recall@{d}: {avg_score}")

if __name__ == "__main__":
    parser = ArgumentParser(description="trec.")

    # Input Arguments.
    parser.add_argument('--qrels', dest='qrels', required=True, type=str)
    parser.add_argument('--ranking', dest='ranking', required=False, type=str)
    parser.add_argument('--distillation_score', dest='distillation_score', required=False, type=str)
    parser.add_argument('--ranx_score', dest='ranx_score', required=False, type=str, default=None)

    args = parser.parse_args()

    main(args)