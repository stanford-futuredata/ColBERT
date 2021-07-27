import os
import torch
import ujson

from math import ceil
from itertools import accumulate
from colbert.utils.utils import print_message


def get_parts(directory, load_compressed_index=False):
    if load_compressed_index:
        extension = '.bn'
    else:
        extension = '.pt'

    parts = sorted([int(filename[: -1 * len(extension)]) for filename in os.listdir(directory)
                    if filename.endswith(extension)])

    assert list(range(len(parts))) == parts, parts

    # Integer-sortedness matters.
    parts_paths = [os.path.join(directory, '{}{}'.format(filename, extension)) for filename in parts]
    samples_paths = [os.path.join(directory, '{}.sample'.format(filename)) for filename in parts]

    return parts, parts_paths, samples_paths


def load_doclens(directory, load_compressed_index=False, flatten=True):
    parts, _, _ = get_parts(directory, load_compressed_index=load_compressed_index)

    doclens_filenames = [os.path.join(directory, 'doclens.{}.json'.format(filename)) for filename in parts]
    all_doclens = [ujson.load(open(filename)) for filename in doclens_filenames]

    if flatten:
        all_doclens = [x for sub_doclens in all_doclens for x in sub_doclens]

    return all_doclens

def load_compression_data(level, path):
    with open(path, "r") as f:
        for line in f:
            line = line.split(',')
            bits = int(line[0])
            if bits == level:
                return [float(v) for v in line[1:]]
    raise ValueError(f"No data found for {level}-bit compression")
