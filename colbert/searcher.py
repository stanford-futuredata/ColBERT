from colbert.infra.launcher import print_memory_stats
import os
import torch

from tqdm import tqdm
from typing import Union

from colbert.data import Collection, Queries, Ranking
from colbert.infra.provenance import Provenance
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.modeling.checkpoint import Checkpoint

from colbert.search.index_storage import IndexScorer


TextQueries = Union[str, 'list[str]', 'dict[int, str]', Queries]


class Searcher:
    def __init__(self, index, checkpoint=None, collection=None, config=None):
        assert type(index) == str, "TODO: Add support for object inputs [index, model]"

        print_memory_stats()

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_
        self.index = os.path.join(default_index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.checkpoint = checkpoint or self.index_config.checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        
        self.config = ColBERTConfig.from_existing(self.checkpoint_config, self.index_config, initial_config)

        self.collection = Collection.cast(collection or self.config.collection)
        self.configure(checkpoint=self.checkpoint, collection=self.collection)

        self.checkpoint = Checkpoint(self.checkpoint, colbert_config=self.config).cuda()  # FIXME: Devices!!

        self.ranker = IndexScorer(self.index)

        print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encode(self, text: TextQueries):
        queries = text if type(text) is list else [text]

        assert type(queries) in [list, tuple], type(queries)

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(queries, bsize=512 if len(queries) > 512 else None, to_cpu=True)

        return Q

    def search(self, text: str, k=10):
        return self.dense_search(self.encode(text), k)

    def search_all(self, queries: TextQueries, k=10):
        queries = Queries.cast(queries)

        Q = self.encode(list(queries.values()))

        all_scored_pids = [list(zip(*self.dense_search(Q[query_idx:query_idx+1], k=k)))
                        for query_idx in tqdm(range(Q.size(0)))]

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = 'Searcher::search_all'
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

    def dense_search(self, Q: torch.Tensor, k=10):
        pids, scores = self.ranker.rank(self.config, Q, k)

        # assert k <= len(pids)
        return pids[:k], list(range(1, k+1)), scores[:k]


# TODO: How are we thinking about DEVICEs here? Who moves what were?

