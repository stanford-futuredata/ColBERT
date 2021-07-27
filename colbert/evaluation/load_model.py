import os
import ujson
import torch
import random

from collections import defaultdict, OrderedDict

from colbert.parameters import DEVICE
from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message, load_checkpoint


def load_model(args, do_print=True):
    # Note: you have to pass `lm` twice because the from_pretrained function won't pass the first
    # parameter on to the ColBERT class' __init__ function, but will pass the second one.
    colbert = ColBERT.from_pretrained(args.lm,
                                      lm=args.lm,
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)
    colbert = colbert.to(DEVICE)

    print_message("#> Loading model checkpoint.", condition=do_print)

    checkpoint = load_checkpoint(args.checkpoint, colbert, do_print=do_print)

    colbert.eval()

    return colbert, checkpoint
