import os
import random
import math

from colbert.utils.runs import Run
from colbert.utils.parser import Arguments
from colbert.indexing.faiss import index_faiss
from colbert.indexing.loaders import load_doclens


def main():
    random.seed(12345)

    parser = Arguments(description='Faiss indexing for end-to-end retrieval with ColBERT.')
    parser.add_index_use_input()

    parser.add_argument('--sample', dest='sample', default=None, type=float)
    parser.add_argument('--slices', dest='slices', default=1, type=int)

    args = parser.parse()
    assert args.slices >= 1
    assert args.sample is None or (0.0 < args.sample < 1.0), args.sample

    with Run.context():
        args.index_path = os.path.join(args.index_root, args.index_name)
        assert os.path.exists(args.index_path), args.index_path

        num_embeddings = sum(load_doclens(args.index_path))
        print("#> num_embeddings =", num_embeddings)

        if args.partitions is None:
            args.partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))
            print('\n\n')
            Run.warn("You did not specify --partitions!")
            Run.warn("Default computation chooses", args.partitions,
                     "partitions (for {} embeddings)".format(num_embeddings))
            print('\n\n')

        index_faiss(args)


if __name__ == "__main__":
    main()
