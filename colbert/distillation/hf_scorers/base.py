from colbert.infra import Run
from colbert.parameters import DEVICE
from colbert.utils.utils import flatten
from colbert.infra.launcher import Launcher

class BaseHFScorer:
    def __init__(self, queries, collection, model, bsize=32, maxlen=180):
        self.queries = queries
        self.collection = collection
        self.model = model

        self.device = DEVICE
        self.bsize = bsize
        self.maxlen = maxlen

    def launch(self, qids, pids):
        launcher = Launcher(self._score_pairs_process, return_all=True)
        outputs = launcher.launch(Run().config, qids, pids)

        return flatten(outputs)

    def _score_pairs_process(self, config, qids, pids):
        assert len(qids) == len(pids), (len(qids), len(pids))
        share = 1 + len(qids) // config.nranks
        offset = config.rank * share
        endpos = (1 + config.rank) * share

        return self.score(qids[offset:endpos], pids[offset:endpos], show_progress=(config.rank < 1))

    def score(self, qids, pids):
        raise NotImplementedError