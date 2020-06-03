import os
import torch

from src.utils import print_message


def load_document_encodings(directory, prefix='', extension='.pt'):
    print_message("#> Loading the document representations from", directory, "...")

    parts = sorted([int(filename[len(prefix): -1 * len(extension)]) for filename in os.listdir(directory)
                    if filename.endswith(extension)])

    assert list(range(len(parts))) == parts, parts

    collection = []

    for filename in parts:
        filename = os.path.join(directory, prefix + str(filename) + extension)  # + '.pt')

        print_message("#> Loading", filename, "...")
        sub_collection = torch.load(filename)

        print_message("#> Extending the collection...")
        collection.extend(sub_collection)

        print_message("#> Now the collection contains", len(collection), "documents...\n\n")

    return collection
