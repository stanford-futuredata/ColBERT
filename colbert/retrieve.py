import os
import random

from colbert.utils.parser import Arguments
from colbert.utils.runs import Run

from colbert.evaluation.loaders import load_colbert, load_qrels, load_queries
from colbert.indexing.faiss import get_faiss_index_name
from colbert.ranking.retrieval import retrieve
from colbert.ranking.batch_retrieval import batch_retrieve


def main():
    random.seed(12345)

    parser = Arguments(description='End-to-end retrieval and ranking with ColBERT.')

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_ranking_input()
    parser.add_retrieval_input()

    parser.add_argument('--faiss_name', dest='faiss_name', default=None, type=str)
    parser.add_argument('--faiss_depth', dest='faiss_depth', default=1024, type=int)
    parser.add_argument('--part-range', dest='part_range', default=None, type=str)
    parser.add_argument('--batch', dest='batch', default=False, action='store_true')
    parser.add_argument('--depth', dest='depth', default=1000, type=int)

    args = parser.parse()

    args.depth = args.depth if args.depth > 0 else None

    if args.part_range:
        part_offset, part_endpos = map(int, args.part_range.split('..'))
        args.part_range = range(part_offset, part_endpos)

    with Run.context():
        args.colbert, args.checkpoint = load_colbert(args)
        args.qrels = load_qrels(args.qrels)
        args.queries = load_queries(args.queries)

        args.index_path = os.path.join(args.index_root, args.index_name)

        if args.faiss_name is not None:
            args.faiss_index_path = os.path.join(args.index_path, args.faiss_name)
        else:
            args.faiss_index_path = os.path.join(args.index_path, get_faiss_index_name(args))

        if args.batch:
            batch_retrieve(args)
        else:
            retrieve(args)


if __name__ == "__main__":
    main()
