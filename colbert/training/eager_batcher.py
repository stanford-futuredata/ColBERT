import os
import ujson
import pandas as pd

from functools import partial
from colbert.utils.utils import print_message
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples

from colbert.utils.runs import Run


class EagerBatcher():
    def __init__(self, args, rank=0, nranks=1):
        self.rank, self.nranks = rank, nranks
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen, args.base_model)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen, args.base_model)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)

        self.triples_path = args.triples
        self._reset_triples()
        
        # Load SRC-TRG parallel corpus
        self._use_gradient_reversal = args.lp_loss
        print("use_gradient_reversal : ", self._use_gradient_reversal)
        if self._use_gradient_reversal == True:
            print("pass")
            self.src_seq, self.trg_seq = self._load_parallel_corpus(args.parallel)
            #print(self.src_seq)

    def _reset_triples(self):
        print("_reset_triples")
        self.reader = open(self.triples_path, mode='r', encoding="utf-8")
        self.position = 0
        
    def _load_parallel_corpus(self, path):
        
        parallel_reader = pd.read_excel(path)
        src = list(parallel_reader['src_seq'])*4
        trg = list(parallel_reader['trg_seq'])*4
        
        return src, trg

    def __iter__(self):
        return self

    def __next__(self):
        queries, positives, negatives = [], [], []

        for line_idx, line in zip(range(self.bsize * self.nranks), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            query, pos, neg = line.strip().split('\t')

            queries.append(query)
            positives.append(pos)
            negatives.append(neg)
            
            if self._use_gradient_reversal == True:
                sources, targets = [], []
                sources.append(self.src_seq[line_idx])
                targets.append(self.trg_seq[line_idx])
            else:
                sources = None
                targets = None

        self.position += line_idx + 1

        if len(queries) < self.bsize:
            raise StopIteration

        return self.collate(queries, positives, negatives, sources, targets)

    def collate(self, queries, positives, negatives, sources=None, targets=None):
        assert len(queries) == len(positives) == len(negatives) == self.bsize

        return self.tensorize_triples(queries, positives, negatives, self.bsize // self.accumsteps, self._use_gradient_reversal, sources, targets)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        self._reset_triples()

        Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')

        _ = [self.reader.readline() for _ in range(batch_idx * intended_batch_size)]

        return None
