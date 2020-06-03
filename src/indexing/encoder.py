import os
import torch
import torch.nn as nn

from time import time
from math import ceil
from multiprocessing import Pool
from transformers import BertTokenizer

from src.parameters import DEVICE
from src.utils import print_message, create_directory


SUPER_BATCH_SIZE = 500*1024

Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
pool = Pool(28)


def to_indexed_list(D, mask, nbytes):
    mask = torch.tensor(mask).bool()

    D = D.detach().cpu()
    if nbytes == 2:
        D = D.to(dtype=torch.float16)
    else:
        assert nbytes == 4

    return [d[mask[idx]] for idx, d in enumerate(D)]


def process_batch(args, super_batch_idx, batch_indices, super_batch):
    colbert = args.colbert

    start_time = time()
    print_message("Start process_batch()", "")

    collection = []
    collection_indices = []

    with torch.no_grad():
        super_batch = list(pool.map(Tokenizer.tokenize, super_batch))
        print_message("Done tokenizing", "")

        sorted_idxs = sorted(range(len(super_batch)), key=lambda i: len(super_batch[i]))
        print_message("Done sorting", "")

        bucketed_outputs = []

        for batch_idx in range(ceil(len(super_batch) / args.bsize)):
            D_idxs = sorted_idxs[batch_idx * args.bsize: (batch_idx + 1) * args.bsize]
            D = [super_batch[d] for d in D_idxs]
            bucketed_outputs.append(to_indexed_list(*colbert.doc(D, return_mask=True), nbytes=args.bytes))
            collection_indices += [batch_indices[d] for d in D_idxs]

        for output in bucketed_outputs:
            collection += [d for d in output]

    throughput = round(len(super_batch) / (time() - start_time) * 60, 2)
    print("This super-batch's encoding rate:", throughput, "passages per minute.")

    output_path = os.path.join(args.index, str(super_batch_idx) + ".pt")
    offset, endpos = min(collection_indices), max(collection_indices)

    print("#> Writing", offset, "to", endpos, "to", output_path, '...')

    assert len(collection) == len(collection_indices)
    assert endpos - offset + 1 == len(collection_indices), (len(collection_indices))
    assert len(collection_indices) == len(set(collection_indices))

    collectionX = [None] * len(collection_indices)

    for pos, idx in enumerate(collection_indices):
        collectionX[idx - offset] = collection[pos]

    torch.save(collectionX, output_path)

    print("#> Saved!\n\n")


def encode(args, number_of_subindexes_already_saved=0):
    # TODO: Create a metadata file; save `args.input_arguments` in there
    create_directory(args.index)

    args.bsize = args.bsize * torch.cuda.device_count()

    print("#> Starting with NUM_GPUs =", torch.cuda.device_count())
    print("#> Accordingly, setting total args.bsize =", args.bsize)

    colbert = args.colbert
    colbert.bert = nn.DataParallel(colbert.bert)
    colbert.linear = nn.DataParallel(colbert.linear)
    colbert = colbert.cuda()
    colbert.eval()

    print('\n\n\n')
    print("#> args.output_dir =", args.output_dir)
    print("#> number_of_subindexes_already_saved =", number_of_subindexes_already_saved)
    print('\n\n\n')

    super_batch_idx = 0
    super_batch, batch_indices = [], []

    with open(args.collection) as f:
        for idx, passage in enumerate(f):
            if len(super_batch) == SUPER_BATCH_SIZE:
                if super_batch_idx < number_of_subindexes_already_saved:
                    print("#> Skipping super_batch_idx =", super_batch_idx, ".......")
                else:
                    process_batch(args, super_batch_idx, batch_indices, super_batch)

                print_message("Processed", str(idx), "passages so far...\n")

                super_batch_idx += 1
                super_batch, batch_indices = [], []

            pid, passage = passage.split('\t')
            super_batch.append(passage)
            batch_indices.append(idx)

            assert int(pid) == idx

    if len(super_batch):
        process_batch(args, super_batch_idx, batch_indices, super_batch)
        super_batch_idx += 1
