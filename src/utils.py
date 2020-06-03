import os
import torch
import datetime


def print_message(*s):
    s = ' '.join([str(x) for x in s])
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)


def save_checkpoint(path, epoch_idx, mb_idx, model, optimizer):
    print("#> Saving a checkpoint..")

    checkpoint = {}
    checkpoint['epoch'] = epoch_idx
    checkpoint['batch'] = mb_idx
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None):
    print_message("#> Loading checkpoint", path)

    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print_message("#> checkpoint['epoch'] =", checkpoint['epoch'])
    print_message("#> checkpoint['batch'] =", checkpoint['batch'])

    return checkpoint


def create_directory(path):
    if not os.path.exists(path):
        print_message("#> Creating", path)
        os.makedirs(path)


def batch(group, bsize):
    offset = 0
    while offset < len(group):
        L = group[offset: offset + bsize]
        yield L
        offset += len(L)
    return
