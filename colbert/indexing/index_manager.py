import torch
import faiss
import numpy as np

from colbert.utils.utils import print_message


class IndexManager():
    def __init__(self, dim):
        self.dim = dim

    def save(self, tensor, path_prefix):
        torch.save(tensor, path_prefix)


def load_index_part(filename, verbose=True):
    part = torch.load(filename)

    if type(part) == list:  # for backward compatibility
        part = torch.cat(part)

    return part
