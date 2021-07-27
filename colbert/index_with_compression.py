import argparse
import copy
import os
import ujson
from queue import PriorityQueue
import random
import sys
import threading
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from colbert.utils.runs import Run
from colbert.utils.parser import Arguments
import colbert.utils.distributed as distributed

from colbert.utils.utils import print_message, create_directory
from colbert.indexing.encoder import CollectionEncoder
from colbert.indexing.faiss_index import FaissIndex
from colbert.indexing.faiss import get_faiss_index_name

STOP = "STOP"

def index_faiss(index, faiss_index_queues):
    while True:
        done = False
        for i in range(len(faiss_index_queues)):
            print_message(f"Retrieving embeddings from process {i} for FAISS indexing...")
            (part_id, part) = faiss_index_queues[i].get()
            if isinstance(part, str) and part == STOP:
                print_message(f"Process {i} finished encoding...")
                if i == len(faiss_index_queues) - 1:
                    done = True
                    break
                else:
                    continue
            print_message(f"Adding part {part_id} to FAISS index...")
            part = part.float().numpy()
            index.add(part)
        if done:
            break


def encode(rank, nranks, faiss_train_sample, faiss_index_queues):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = str(nranks)
    os.environ["RANK"] = str(rank)

    random.seed(12345)

    parser = Arguments(description='Precomputing document representations with ColBERT.')

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_indexing_input()
    parser.add_compressed_index_input()

    parser.add_argument('--chunksize', dest='chunksize', default=6.0, required=False, type=float)   # in GiBs
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', required=False)
    parser.add_argument("--partitions", dest="partitions", type=int)
    parser.add_argument('--sample', dest='sample', default=0.05, type=float)

    sys.argv += ["--local_rank", str(rank)]

    args = parser.parse()

    if args.sample <= 0 or args.sample > 1:
        raise ValueError("FAISS index training sample fraction must be in the range (0, 1]")

    with Run.context():
        args.index_path = os.path.join(args.index_root, args.index_name)
        args.faiss_index_path = os.path.join(args.index_path, get_faiss_index_name(args))
        assert not os.path.exists(args.index_path), args.index_path

        distributed.barrier(args.rank)

        if args.rank < 1:
            create_directory(args.index_root)
            create_directory(args.index_path)
            faiss_index = FaissIndex(args.dim, args.partitions)

        distributed.barrier(args.rank)

        process_idx = max(0, args.rank)
        encoder = CollectionEncoder(args, process_idx=process_idx, num_processes=args.nranks)

        if args.rank < 1:
            print_message(f"Encoding FAISS training sample data...")
        encoder.prepare_train_sample(faiss_train_sample, args.sample)

        distributed.barrier(args.rank)

        if args.rank < 1:
            print_message("Training FAISS index...")
            faiss_train_sample = torch.cat(tuple(faiss_train_sample))
            faiss_train_sample = faiss_train_sample.float().numpy()
            faiss_index.train(faiss_train_sample)
            faiss_thread = threading.Thread(
                target=index_faiss, args=(faiss_index, faiss_index_queues)
            )
            faiss_thread.start()

        distributed.barrier(args.rank)

        encoder.encode(faiss_index_queues[args.rank])

        if process_idx < 1:
            faiss_thread.join()
            print_message("Saving FAISS index...")
            faiss_index.save(args.faiss_index_path)

        # Save metadata.
        if args.rank < 1:
            metadata_path = os.path.join(args.index_path, 'metadata.json')
            print_message("Saving (the following) metadata to", metadata_path, "..")
            print(args.input_arguments)

            with open(metadata_path, 'w') as output_metadata:
                ujson.dump(args.input_arguments.__dict__, output_metadata)

        distributed.barrier(args.rank)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc_per_node", dest="nproc_per_node", type=int, default=1)
    args, unknown = parser.parse_known_args()

    manager = mp.Manager()
    faiss_train_sample = manager.list()
    faiss_index_queues = [manager.Queue(maxsize=1) for _ in range(args.nproc_per_node)]

    all_procs = []
    for i in range(args.nproc_per_node):
        args_ = (
            i,
            args.nproc_per_node,
            faiss_train_sample,
            faiss_index_queues
        )
        all_procs.append(mp.Process(target=encode, args=args_))
    for proc in all_procs:
        proc.start()
    for proc in all_procs:
        proc.join()


if __name__ == "__main__":
    main()

# TODO: Add resume functionality
