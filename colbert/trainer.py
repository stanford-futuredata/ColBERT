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
        """
        Configure training argument for ColBERT retriever 

        :param kw_args: Chosen arguments
        :type kw_args: Dictionary
        """
        self.config.configure(**kw_args)

    def train(self, checkpoint='bert-base-uncased'):
        """
            Note that config.checkpoint is ignored. Only the supplied checkpoint here is used.
        """

        """
        Trains ColBERT retriever using selected checkpoints
        Note that config.checkpoint is ignored. Only the supplied checkpoint here is used.

        :param checkpoint: Checkpoint for starting training. Default is 'bert-base-uncased'
        :type checkpoint: str
        """

        # Resources don't come from the config object. They come from the input parameters.
        # TODO: After the API stabilizes, make this "self.config.assign()" to emphasize this distinction.
        self.configure(triples=self.triples, queries=self.queries, collection=self.collection)
        self.configure(checkpoint=checkpoint)

        launcher = Launcher(train)

        self._best_checkpoint_path = launcher.launch(self.config, self.triples, self.queries, self.collection)


    def best_checkpoint_path(self):
        """
        Returns best performing checkpoint of ColBERT retriever

        :return: Best checkpoint path
        :rtype: Path
        """
        return self._best_checkpoint_path

