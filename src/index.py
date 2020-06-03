import os
import random

from argparse import ArgumentParser
from multiprocessing import Pool

from src.parameters import DEFAULT_DATA_DIR, DEVICE
from src.utils import print_message, create_directory

from src.evaluation.loaders import load_colbert
from src.indexing.encoder import encode


def main():
    random.seed(123456)

    parser = ArgumentParser(description='Exhaustive (non-index-based) evaluation of re-ranking with ColBERT.')

    parser.add_argument('--index', dest='index', required=True)
    parser.add_argument('--checkpoint', dest='checkpoint', required=True)
    parser.add_argument('--collection', dest='collection', default='collection.tsv')

    parser.add_argument('--data_dir', dest='data_dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--output_dir', dest='output_dir', default='outputs.index/')

    parser.add_argument('--bsize', dest='bsize', default=128, type=int)
    parser.add_argument('--bytes', dest='bytes', default=2, choices=[2, 4], type=int)
    parser.add_argument('--subsample', dest='subsample', default=None)  # TODO: Add this

    # TODO: For the following four arguments, default should be None. If None, they should be loaded from checkpoint.
    parser.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])
    parser.add_argument('--dim', dest='dim', default=128, type=int)
    parser.add_argument('--query_maxlen', dest='query_maxlen', default=32, type=int)
    parser.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)

    # TODO: Add resume functionality

    args = parser.parse_args()
    args.input_arguments = args
    args.pool = Pool(10)

    create_directory(args.output_dir)

    args.index = os.path.join(args.output_dir, args.index)
    args.collection = os.path.join(args.data_dir, args.collection)

    args.colbert, args.checkpoint = load_colbert(args)

    encode(args)


if __name__ == "__main__":
    main()
