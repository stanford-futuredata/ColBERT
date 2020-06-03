import os
import random
import torch

from argparse import ArgumentParser

from src.parameters import DEFAULT_DATA_DIR
from src.training.data_reader import train
from src.utils import print_message, create_directory


def main():
    random.seed(12345)
    torch.manual_seed(1)

    parser = ArgumentParser(description='Training ColBERT with <query, positive passage, negative passage> triples.')

    parser.add_argument('--lr', dest='lr', default=3e-06, type=float)
    parser.add_argument('--maxsteps', dest='maxsteps', default=400000, type=int)
    parser.add_argument('--bsize', dest='bsize', default=32, type=int)
    parser.add_argument('--accum', dest='accumsteps', default=2, type=int)

    parser.add_argument('--data_dir', dest='data_dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--triples', dest='triples', default='triples.train.small.tsv')
    parser.add_argument('--output_dir', dest='output_dir', default='outputs.train/')

    parser.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])
    parser.add_argument('--dim', dest='dim', default=128, type=int)
    parser.add_argument('--query_maxlen', dest='query_maxlen', default=32, type=int)
    parser.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)

    # TODO: Add resume functionality
    # TODO: Save the configuration to the checkpoint.
    # TODO: Extract common parser arguments/behavior into a class.

    args = parser.parse_args()
    args.input_arguments = args

    create_directory(args.output_dir)

    assert args.bsize % args.accumsteps == 0, ((args.bsize, args.accumsteps),
                                               "The batch size must be divisible by the number of gradient accumulation steps.")
    assert args.query_maxlen <= 512
    assert args.doc_maxlen <= 512

    args.triples = os.path.join(args.data_dir, args.triples)

    train(args)


if __name__ == "__main__":
    main()
