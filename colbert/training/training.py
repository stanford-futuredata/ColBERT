import time
import torch
import random
import torch.nn as nn
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints

from schedulefree import AdamWScheduleFree


def train(config: ColBERTConfig, triples, queries=None, collection=None):
    config.checkpoint = config.checkpoint or "bert-base-uncased"

    if config.rank < 1:
        config.help()

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print(
        "Using config.bsize =",
        config.bsize,
        "(per process) and config.accumsteps =",
        config.accumsteps,
    )

    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(
                config,
                triples,
                queries,
                collection,
                (0 if config.rank == -1 else config.rank),
                config.nranks,
            )
        else:
            reader = LazyBatcher(
                config,
                triples,
                queries,
                collection,
                (0 if config.rank == -1 else config.rank),
                config.nranks,
            )
    else:
        raise NotImplementedError()

    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(DEVICE)
    colbert.train()

    colbert = torch.nn.parallel.DistributedDataParallel(
        colbert,
        device_ids=[config.rank],
        output_device=config.rank,
        find_unused_parameters=True,
    )

    if config.schedule_free is False:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, colbert.parameters()),
            lr=config.lr,
            eps=1e-8,
        )
    else:
        print("WARNING, USING SCHEDULE FREE")
        print("WARNING, USING SCHEDULE FREE")
        print("WARNING, USING SCHEDULE FREE")
        print("WARNING, USING SCHEDULE FREE")
        print("WARNING, USING SCHEDULE FREE")
        optimizer = AdamWScheduleFree(
            filter(lambda p: p.requires_grad, colbert.parameters()),
            lr=config.lr,
            warmup_steps=config.warmup,
            weight_decay=config.schedule_free_wd,
        )
    if config.schedule_free:
        optimizer.train()
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None and config.schedule_free is False:
        print(
            f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps."
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup,
            num_training_steps=config.maxsteps,
        )

    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
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

        for batch in BatchSteps:
            with amp.context():
                try:
                    queries, passages, target_scores = batch
                    encoding = [queries, passages]
                except:
                    encoding, target_scores = batch
                    encoding = [encoding.to(DEVICE)]

                if not config.quant_aware:
                    scores = colbert(*encoding)
                else:
                    raise NotImplementedError

                if config.use_ib_negatives:
                    scores, ib_loss = scores
                    ib_loss = ib_loss * config.ib_loss_weight

                scores = scores.view(-1, config.nway)
                if config.normalise_training_scores:
                    if config.normalization_method == "minmax":
                        print('norm')
                        scores = (scores - scores.min(dim=-1, keepdim=True)[0]) / (
                            scores.max(dim=-1, keepdim=True)[0]
                            - scores.min(dim=-1, keepdim=True)[0]
                            + 1e-8
                        )
                    elif config.normalization_method == "querylen":
                        scores = scores / (
                            config.query_maxlen + 1e-8
                        )  # Divide by the number of tokens in the queries

                if len(target_scores) and not config.ignore_scores:
                    target_scores = (
                        torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                    )
                    target_scores = target_scores * config.distillation_alpha

                    if config.kldiv_loss:
                        target_scores = torch.nn.functional.log_softmax(
                            target_scores, dim=-1
                        )

                        log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                        kldivloss = torch.nn.KLDivLoss(
                            reduction="batchmean", log_target=True
                        )(log_scores, target_scores)

                    if config.marginmse_loss:
                        margin = scores[:, 0].unsqueeze(1) - scores[:, 1:]
                        target_margin = target_scores[:, 0].unsqueeze(1) - target_scores[:, 1:]
                        marginmse_loss = torch.nn.MSELoss()(margin, target_margin)

                    if config.kldiv_loss and config.marginmse_loss:
                        weighted_kldiv = kldivloss * config.kldiv_weight
                        weighted_marginmse = marginmse_loss * config.marginmse_weight
                        loss = (
                            weighted_kldiv
                            + weighted_marginmse
                        )
                    elif config.kldiv_loss:
                        loss = kldivloss
                    elif config.marginmse_loss:
                        loss = marginmse_loss
                    else:
                        raise ValueError(
                            "One or both of config.kldiv_loss and config.marginmse_loss must be True if distillation is enabled!"
                        )
                else:
                    raise ValueError("crossentropy loss shouldn't be used here")
                    loss = nn.CrossEntropyLoss()(scores, labels[: scores.size(0)])

                if config.use_ib_negatives:
                    if config.rank < 1:
                        print("\t\t\t\t", loss.item(), ib_loss.item())

                    og_loss = loss
                    loss += ib_loss

                loss = loss / config.accumsteps

            if config.rank < 1:
                print_progress(scores)

            amp.backward(loss)

            this_batch_loss += loss.item()

        train_loss = this_batch_loss if train_loss is None else train_loss
        train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

        if config.schedule_free:
            assert scheduler is None

        amp.step(colbert, optimizer, scheduler)

        if config.rank < 1:
            if config.use_ib_negatives:
                print_message(f"IB Loss: {ib_loss}")
                print_message(f"KL-D loss: {og_loss}")
            if config.kldiv_loss and config.marginmse_loss:
                TOTAL = weighted_kldiv + weighted_marginmse
                kldiv_proportion = weighted_kldiv / TOTAL
                marginmse_proportion = weighted_marginmse / TOTAL
                print_message(f"Weighted KL-D loss: {weighted_kldiv:.4f}")
                print_message(f"Weighted MarginMSE loss: {weighted_marginmse:.4f}")
                print_message(f"Respective proportions: KL-D {kldiv_proportion:.2%}, MarginMSE {marginmse_proportion:.2%}")
            print_message(batch_idx, train_loss)
            manage_checkpoints(config, colbert, optimizer, batch_idx + 1, savepath=None)

    if config.rank < 1:
        print_message("#> Done with all triples!")
        ckpt_path = manage_checkpoints(
            config,
            colbert,
            optimizer,
            batch_idx + 1,
            savepath=None,
            consumed_all_triples=True,
            is_schedule_free=config.schedule_free,
        )

        return ckpt_path  # TODO: This should validate and return the best checkpoint, not just the last one.


def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
