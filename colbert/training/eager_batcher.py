import os
import ujson

from functools import partial
from colbert.utils.utils import print_message
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples

from colbert.utils.runs import Run
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename='colbert_eager_batcher.log', filemode='a')

class EagerBatcher():
    def __init__(self, args, rank=0, nranks=1):
        self.rank, self.nranks = rank, nranks
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)

        self.triples_path = args.triples
        self._reset_triples()

    def _reset_triples(self, skip_lines=0):
        self.reader = open(self.triples_path, mode='r', encoding="utf-8")
        self.position = 0
        if skip_lines:
            for _ in range(skip_lines):
                self.reader.readline()

    def __iter__(self):
        return self

    def __next__(self):
        queries, positives, negatives = [], [], []

        for line_idx, line in zip(range(self.bsize * self.nranks), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue
            try:
                query, pos, neg = line.strip().split('\t')
                queries.append(query)
                positives.append(pos)
                negatives.append(neg)
            except:
                query, pos = line.strip().split('\t')
                neg = "aaa"
                queries.append(query)
                positives.append(pos)
                negatives.append(neg)
                
                print_message(f'Invalid line: {line_idx}')
                print_message(f'Line: {line}')
                print_message('Skipping...')

        self.position += line_idx + 1

        if len(queries) < self.bsize:
            raise StopIteration

        return self.collate(queries, positives, negatives)

    def collate(self, queries, positives, negatives):
        assert len(queries) == len(positives) == len(negatives) == self.bsize

        return self.tensorize_triples(queries, positives, negatives, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        self._reset_triples(skip_lines=batch_idx * intended_batch_size)

        Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')

        # _ = [self.reader.readline() for _ in range(batch_idx * intended_batch_size)]

        return None
