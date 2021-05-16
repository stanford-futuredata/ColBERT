import os
import torch

from colbert.utils.runs import Run
from colbert.utils.utils import print_message, save_checkpoint
from colbert.parameters import SAVED_CHECKPOINTS


def print_progress(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    print("#>>>   ", positive_avg, negative_avg, '\t\t|\t\t', positive_avg - negative_avg)


def manage_checkpoints(args, colbert, optimizer, batch_idx):
    print("Managing checkpoints...")
    arguments = args.input_arguments.__dict__

    path = os.path.join(Run.path, 'checkpoints')

    if not os.path.exists(path):
        os.mkdir(path)

    if batch_idx % 2000 == 0:
        name = os.path.join(path, "colbert.dnn")
        save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)

    if batch_idx in SAVED_CHECKPOINTS:
        name = os.path.join(path, "colbert-{}.dnn".format(batch_idx))
        print("Saving to...", name)
        save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)


# TODO for Swetha: delete this? Is it user anywhere
def manage_checkpoints_hebbia(args, colbert, optimizer, batch_idx, model_save_path, save):
    arguments = args.input_arguments.__dict__

    folder = os.path.dirname(model_save_path)

    if not os.path.exists(folder):
        os.mkdir(folder)

    if save:
        save_checkpoint(model_save_path, 0, batch_idx, colbert, optimizer, arguments)