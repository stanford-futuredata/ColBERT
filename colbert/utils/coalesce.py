import os
import argparse
import torch
from tqdm import tqdm
import ujson


def main(args):
    in_file = args.input
    out_file = args.output

    # Get num_chunks from metadata
    filepath = os.path.join(in_file, 'metadata.json')
    with open(filepath, 'r') as f:
        metadata = ujson.load(f)
    num_chunks = metadata['num_chunks']
    print(f"Num_chunks = {num_chunks}")

    # Create output dir if not already created
    if not os.path.exists(out_file):
       os.makedirs(out_file)

    ## Coalesce doclens ##
    print("Coalescing doclens files...")

    temp = []
    # read into one large list
    for i in tqdm(range(num_chunks)):
        filepath = os.path.join(in_file, f'doclens.{i}.json')
        with open(filepath, 'r') as f:
            chunk = ujson.load(f)
        temp.extend(chunk)

    # write to output json
    filepath = os.path.join(out_file, 'doclens.0.json')
    with open(filepath, 'w') as f:
        ujson.dump(temp, f)

    ## Coalesce codes ##
    print("Coalescing codes files...")

    temp = torch.empty(0, dtype=torch.int32)
    # read into one large tensor
    for i in tqdm(range(num_chunks)):
        filepath = os.path.join(in_file, f'{i}.codes.pt')
        chunk = torch.load(filepath)
        temp = torch.cat((temp, chunk))

    # save length of index
    index_len = temp.size()[0]

    # write to output tensor
    filepath = os.path.join(out_file, '0.codes.pt')
    torch.save(temp, filepath)

    ## Coalesce residuals ##
    print("Coalescing residuals files...")

    temp = torch.empty(0, dtype=torch.uint8)
    # read into one large tensor
    for i in tqdm(range(num_chunks)):
        filepath = os.path.join(in_file, f'{i}.residuals.pt')
        chunk = torch.load(filepath)
        temp = torch.cat((temp, chunk))

    print("Saving residuals to output directory (this may take a few minutes)...")

    # write to output tensor
    filepath = os.path.join(out_file, '0.residuals.pt')
    torch.save(temp, filepath)

    print("Saved index to output directory {}.".format(out_file))
    print("Number of embeddings = {}".format(index_len))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coalesce multi-file index into a single file.")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input index directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output index directory"
    )

    args = parser.parse_args()
    main(args)
