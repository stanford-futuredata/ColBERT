import os
from collections import namedtuple
from datasets import load_dataset
from utility.utils.dpr import has_answer, DPR_normalize

from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig, RunConfig, Run

CKPT = "/dfs/scratch0/okhattab/share/2021/checkpoints/msmarco.psg.kldR2.nway64.ib__colbert-400000"
DATA_PATH = "/future/u/xmgui/testdata"
EXPERIMENT_DIR = "/future/u/xmgui/ColBERT/experiments"

SquadExample = namedtuple("SquadExample",  "id title context question answers")


def build_index(dataset_path):
    nbits = 1  # encode each dimension with 1 bits
    doc_maxlen = 180  # truncate passages at 180 tokens
    collection = os.path.join(dataset_path, 'cs224u.collection.tsv')
    index_name = f'e2etest.{nbits}bits.latest'
    experiment = (f"e2etest.nbits={nbits}",)

    with Run().context(RunConfig(nranks=4)):
        config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            nbits=nbits,
            root=EXPERIMENT_DIR,
            experiment=experiment,
        )
        indexer = Indexer(CKPT, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)

        config = ColBERTConfig(
            root=EXPERIMENT_DIR,
            experiment=experiment,
        )
        searcher = Searcher(
            index=index_name,
            config=config,
        )

        return searcher


def success_at_k(searcher, examples, k):
    scores = []
    for ex in examples:
        scores.append(evaluate_retrieval_example(searcher, ex, k))
    return sum(scores) / len(scores)


def evaluate_retrieval_example(searcher, ex, k):
    results = searcher.search(ex.question, k=k)
    for passage_id, passage_rank, passage_score in zip(*results):
        passage = searcher.collection[passage_id]
        score = has_answer([DPR_normalize(ans) for ans in ex.answers], passage)
        if score:
            return 1
    return 0


def get_squad_split(squad, split="validation"):
    fields = squad[split].features
    data = zip(*[squad[split][field] for field in fields])
    return [SquadExample(eid, title, context, question, answers["text"])
            for eid, title, context, question, answers in data]


def main():
    k = 5
    searcher = build_index(DATA_PATH)

    squad = load_dataset("squad")
    squad_dev = get_squad_split(squad)
    success_rate = success_at_k(searcher, squad_dev, k)
    assert success_rate > 0.7, f"success rate at {success_rate} is lower than expected"


if __name__ == "__main__":
    main()
