from colbert.infra.run import Run
from colbert.infra.launcher import Launcher
from colbert.infra.config import ColBERTConfig, RunConfig

from colbert.training.training import train


class Trainer:
    def __init__(self, triples, queries, collection, config=None):
        self.config = ColBERTConfig.from_existing(config, Run().config)

        self.triples = triples
        self.queries = queries
        self.collection = collection

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def train(self, checkpoint='bert-base-uncased'):
        """
            Note that config.checkpoint is ignored. Only the supplied checkpoint here is used.
        """

        # Resources don't come from the config object. They come from the input parameters.
        # TODO: After the API stabilizes, make this "self.config.assign()" to emphasize this distinction.
        self.configure(triples=self.triples, queries=self.queries, collection=self.collection)
        self.configure(checkpoint=checkpoint)
        self.best_checkpoint_path = self.__launch()

    def __launch(self):
        launcher = Launcher(train)
        if self.config.nranks == 1 and self.config.avoid_fork_if_possible:
            print('AVOIDING FORK DURING TRAINING')
            return launcher.launch_without_fork(self.config, self.triples, self.queries, self.collection)

        return launcher.launch(self.config, self.triples, self.queries, self.collection)

    def best_checkpoint_path(self):
        return self._best_checkpoint_path

