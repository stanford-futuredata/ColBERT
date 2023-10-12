import os
import torch
import ujson
from tqdm import tqdm

import argparse

def main(args):
    # TODO: compare residual, codes, and doclens
    # Get the number of chunks in the multi-file index
    single_path = args.single
    multi_path = args.multi

    # Get num_chunks and num_embeddings from metadata
    filepath = os.path.join(multi_path, 'metadata.json')
    with open(filepath, 'r') as f:
        metadata = ujson.load(f)
    num_chunks = metadata['num_chunks']
    print(f"Num_chunks = {num_chunks}")

    num_embeddings = metadata['num_embeddings']
    print(f"Num_embeddings = {num_embeddings}")

    dim = metadata['config']['dim']
    nbits = metadata['config']['nbits']

    ## Load and compare doclens ##
    # load multi-file doclens
    print("Loading doclens from multi-file index")
    multi_doclens = []
    for chunk_idx in tqdm(range(num_chunks)):
        with open(os.path.join(multi_path, f"doclens.{chunk_idx}.json"), 'r') as f:
            chunk = ujson.load(f)
            multi_doclens.extend(chunk)

    # load single-file doclens
    print("Loading doclens from single-file index")
    single_doclens = []
    for _ in tqdm(range(1)):
        with open(os.path.join(single_path, "doclens.0.json"), 'r') as f:
            single_doclens = ujson.load(f)

    # compare doclens
    if (multi_doclens != single_doclens):
        print("Doclens do not match!")
        print("Multi-file doclens size = {}".format(len(multi_doclens)))
        print("Single-file doclens size = {}".format(len(single_doclens)))
    else:
        print("Doclens match")

    ## Load and compare codes ##
    # load multi-file codes
    print("Loading codes from multi-file index")
    multi_codes = torch.empty(num_embeddings, dtype=torch.int32)
    offset = 0
    for chunk_idx in tqdm(range(num_chunks)):
        chunk = torch.load(os.path.join(multi_path, f"{chunk_idx}.codes.pt"))
        endpos = offset + chunk.size(0)
        multi_codes[offset:endpos] = chunk
        offset = endpos

    # load single-file codes
    print("Loading codes from single-file index")
    single_codes = []
    for _ in tqdm(range(1)):
        single_codes = torch.load(os.path.join(single_path, "0.codes.pt"))

    if (single_codes.size(0) != num_embeddings):
        print("Codes are the wrong size!")

    # compare codes
    if torch.equal(multi_codes, single_codes):
        print("Codes match")
    else:
        print("Codes do not match!")

    ## Load and compare residuals ##
    # load multi-file residuals
    print("Loading residuals from multi-file index")
    multi_residuals = torch.empty(num_embeddings, dim // 8 * nbits, dtype=torch.uint8)
    offset = 0
    for chunk_idx in tqdm(range(num_chunks)):
        chunk = torch.load(os.path.join(multi_path, f"{chunk_idx}.residuals.pt"))
        endpos = offset + chunk.size(0)
        multi_residuals[offset:endpos] = chunk
        offset = endpos

    # load single-file residuals
    print("Loading residuals from single-file index")
    single_residuals = []
    for _ in tqdm(range(1)):
        single_residuals = torch.load(os.path.join(single_path, "0.residuals.pt"))

    # compare residuals
    if torch.equal(multi_residuals, single_residuals):
        print("Residuals match")
    else:
        print("Residuals do not match!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare single-file and multi-file indexes.")
    parser.add_argument(
        "--single", type=str, required=True, help="Path to single-file index."
    )
    parser.add_argument(
        "--multi", type=str, required=True, help="Path to multi-file index."
    )

    args = parser.parse_args()
    main(args)

    print("Exiting test")
