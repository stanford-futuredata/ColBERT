import time
import torch
import random
import torch.nn as nn
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints


def train(config, triples, queries=None, collection=None):
    config.help()
    assert config.distributed

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    if config.distributed:
        torch.cuda.manual_seed_all(12345)

    if config.distributed:
        assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
        assert config.accumsteps == 1
        config.bsize = config.bsize // config.nranks

        print("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    if collection is not None:
        reader = LazyBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
    else:
        raise NotImplementedError()

    colbert = ColBERT(name=config.checkpoint or 'bert-base-uncased', colbert_config=config)

    colbert = colbert.to(DEVICE)
    colbert.train()

    if config.distributed:
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
                                                            output_device=config.rank,
                                                            find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                    num_training_steps=config.maxsteps)
    
    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
    criterion = nn.CrossEntropyLoss()
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    start_batch_idx = 0

    # if config.resume:
    #     assert config.checkpoint is not None
    #     start_batch_idx = checkpoint['batch']

    #     reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        if (warmup_bert is not None) and warmup_bert <= batch_idx:
            set_bert_grad(colbert, True)
            warmup_bert = None

        this_batch_loss = 0.0

        for queries, passages in BatchSteps:
            with amp.context():
                scores = colbert(queries, passages).view(2, -1).permute(1, 0)
                loss = criterion(scores, labels[:scores.size(0)])
                loss = loss / config.accumsteps

            if config.rank < 1:
                print_progress(scores)

            amp.backward(loss)

            this_batch_loss += loss.item()

        train_loss = this_batch_loss if train_loss is None else train_loss
        train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

        amp.step(colbert, optimizer, scheduler)

        if config.rank < 1:
            # avg_loss = train_loss / (batch_idx+1)

            print_message(batch_idx, train_loss)
            manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None)

    if config.rank < 1:
        print_message("#> Done with all triples!")
        ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None, consumed_all_triples=True)

        return ckpt_path  # TODO: This should validate and return the best checkpoint, not just the last one.
    


def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
