import os
import torch

import __main__
from dataclasses import dataclass
from colbert.utils.utils import timestamp

from .core_config import DefaultVal


@dataclass
class RunSettings:
    """
    The defaults here have a special status in Run(), which initially calls assign_defaults(),
    so these aren't soft defaults in that specific context.
    """

    overwrite: bool = DefaultVal(False)

    root: str = DefaultVal(os.path.join(os.getcwd(), "experiments"))
    experiment: str = DefaultVal("default")

    index_root: str = DefaultVal(None)
    name: str = DefaultVal(timestamp(daydir=True))

    rank: int = DefaultVal(0)
    nranks: int = DefaultVal(1)
    amp: bool = DefaultVal(True)

    total_visible_gpus = torch.cuda.device_count()
    gpus: int = DefaultVal(total_visible_gpus)

    avoid_fork_if_possible: bool = DefaultVal(False)

    @property
    def gpus_(self):
        value = self.gpus

        if isinstance(value, int):
            value = list(range(value))

        if isinstance(value, str):
            value = value.split(",")

        value = list(map(int, value))
        value = sorted(list(set(value)))

        assert all(
            device_idx in range(0, self.total_visible_gpus) for device_idx in value
        ), value

        return value

    @property
    def index_root_(self):
        return self.index_root or os.path.join(self.root, self.experiment, "indexes/")

    @property
    def script_name_(self):
        if "__file__" in dir(__main__):
            cwd = os.path.abspath(os.getcwd())
            script_path = os.path.abspath(__main__.__file__)
            root_path = os.path.abspath(self.root)

            if script_path.startswith(cwd):
                script_path = script_path[len(cwd) :]

            else:
                try:
                    commonpath = os.path.commonpath([script_path, root_path])
                    script_path = script_path[len(commonpath) :]
                except:
                    pass

            assert script_path.endswith(".py")
            script_name = script_path.replace("/", ".").strip(".")[:-3]

            assert len(script_name) > 0, (script_name, script_path, cwd)

            return script_name

        return "none"

    @property
    def path_(self):
        return os.path.join(self.root, self.experiment, self.script_name_, self.name)

    @property
    def device_(self):
        return self.gpus_[self.rank % self.nranks]


@dataclass
class TokenizerSettings:
    query_token_id: str = DefaultVal("[unused0]")
    doc_token_id: str = DefaultVal("[unused1]")
    query_token: str = DefaultVal("[Q]")
    doc_token: str = DefaultVal("[D]")


@dataclass
class ResourceSettings:
    checkpoint: str = DefaultVal(None)
    triples: str = DefaultVal(None)
    collection: str = DefaultVal(None)
    queries: str = DefaultVal(None)
    index_name: str = DefaultVal(None)


@dataclass
class DocSettings:
    dim: int = DefaultVal(128)
    doc_maxlen: int = DefaultVal(220)
    mask_punctuation: bool = DefaultVal(True)


@dataclass
class QuerySettings:
    query_maxlen: int = DefaultVal(32)
    attend_to_mask_tokens: bool = DefaultVal(False)
    interaction: str = DefaultVal("colbert")
    # V2.5
    cap_padding: int = DefaultVal(0)
    dynamic_query_maxlen: bool = DefaultVal(False)
    dynamic_querylen_multiples: int = DefaultVal(32)


@dataclass
class TrainingSettings:
    similarity: str = DefaultVal("cosine")

    bsize: int = DefaultVal(32)

    accumsteps: int = DefaultVal(1)

    lr: float = DefaultVal(3e-06)

    maxsteps: int = DefaultVal(500_000)

    save_every: int = DefaultVal(None)

    resume: bool = DefaultVal(False)

    ## NEW:
    warmup: int = DefaultVal(None)

    warmup_bert: int = DefaultVal(None)

    relu: bool = DefaultVal(False)

    nway: int = DefaultVal(2)

    use_ib_negatives: bool = DefaultVal(False)

    reranker: bool = DefaultVal(False)

    distillation_alpha: float = DefaultVal(1.0)

    ignore_scores: bool = DefaultVal(False)

    model_name: str = DefaultVal(None)  # DefaultVal('bert-base-uncased')

    # V2.5

    schedule_free: bool = DefaultVal(False)

    schedule_free_wd: float = DefaultVal(0.0)

    kldiv_loss: bool = DefaultVal(True)

    marginmse_loss: bool = DefaultVal(False)

    kldiv_weight: float = DefaultVal(1.0)

    marginmse_weight: float = DefaultVal(0.05)

    ib_loss_weight: float = DefaultVal(1.0)

    normalise_training_scores: bool = DefaultVal(False)

    # Can be 'minmax', 'querylen'
    normalization_method: str = DefaultVal("minmax")

    # TODO

    quant_aware: bool = DefaultVal(False)

    highest_quant_level: int = DefaultVal(8)

    lowest_quant_level: int = DefaultVal(2)


@dataclass
class IndexingSettings:
    index_path: str = DefaultVal(None)

    index_bsize: int = DefaultVal(64)

    nbits: int = DefaultVal(1)

    kmeans_niters: int = DefaultVal(4)

    resume: bool = DefaultVal(False)

    pool_factor: int = DefaultVal(1)

    clustering_mode: str = DefaultVal("hierarchical")

    protected_tokens: int = DefaultVal(0)

    @property
    def index_path_(self):
        return self.index_path or os.path.join(self.index_root_, self.index_name)


@dataclass
class SearchSettings:
    ncells: int = DefaultVal(None)
    centroid_score_threshold: float = DefaultVal(None)
    ndocs: int = DefaultVal(None)
    load_index_with_mmap: bool = DefaultVal(False)
