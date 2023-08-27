import os
import random

from colbert.utils.parser import Arguments
from colbert.utils.runs import Run

from colbert.evaluation.loaders import load_colbert, load_topK, load_qrels
from colbert.evaluation.loaders import load_queries, load_topK_pids, load_collection
from colbert.evaluation.ranking import evaluate
from colbert.evaluation.metrics import evaluate_recall


def main():
    random.seed(12345)

    parser = Arguments(description='Exhaustive (slow, not index-based) evaluation of re-ranking with ColBERT.')

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_reranking_input()

    parser.add_argument('--depth', dest='depth', required=False, default=None, type=int)

    args = parser.parse()

    with Run.context():
        args.colbert, args.checkpoint = load_colbert(args)
        args.qrels = load_qrels(args.qrels)

        if args.collection or args.queries:
            assert args.collection and args.queries

            args.queries = load_queries(args.queries)
            args.collection = load_collection(args.collection)
            args.topK_pids, args.qrels = load_topK_pids(args.topK, args.qrels)

        else:
            args.queries, args.topK_docs, args.topK_pids = load_topK(args.topK)

        assert (not args.shortcircuit) or args.qrels, \
            "Short-circuiting (i.e., applying minimal computation to queries with no positives in the re-ranked set) " \
            "can only be applied if qrels is provided."

        evaluate_recall(args.qrels, args.queries, args.topK_pids)
        evaluate(args)


if __name__ == "__main__":
    main()
