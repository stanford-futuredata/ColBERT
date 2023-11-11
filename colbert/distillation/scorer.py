import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from colbert.infra import Run
from colbert.utils.utils import flatten
from colbert.infra.launcher import Launcher

from colbert.modeling.reranker.electra import ElectraReranker
from colbert.modeling.reranker.codeT5p import CodeT5pReranker
from colbert.modeling.reranker.deberta import DebertaV2Reranker
from colbert.modeling.reranker.codet5p_encdec import CodeT5pEncDecReranker


DEFAULT_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


class Scorer:
    def __init__(self, queries, collection, model=DEFAULT_MODEL, maxlen=180, bsize=256, model_type="encoder_only"):
        self.queries = queries
        self.collection = collection
        self.model = model

        self.maxlen = maxlen
        self.bsize = bsize
        self.model_type = model_type

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def launch(self, qids, pids):
        launcher = Launcher(self._score_pairs_process, return_all=True)
        outputs = launcher.launch(Run().config, qids, pids)

        return flatten(outputs)

    def _score_pairs_process(self, config, qids, pids):
        assert len(qids) == len(pids), (len(qids), len(pids))
        share = 1 + len(qids) // config.nranks
        offset = config.rank * share
        endpos = (1 + config.rank) * share

        return self._score_pairs(qids[offset:endpos], pids[offset:endpos], show_progress=(config.rank < 1))

    def _score_pairs(self, qids, pids, show_progress=False):
        tokenizer = AutoTokenizer.from_pretrained(self.model)

        if self.model == DEFAULT_MODEL:
            model = AutoModelForSequenceClassification.from_pretrained(self.model).to(self.device)
        else:
            if self.model_type == "encoder-only":
                if 'codet5' in str(self.model):
                    model = CodeT5pReranker.from_pretrained(self.model).to(self.device)
                elif 'deberta' in str(self.model):
                    model = DebertaV2Reranker.from_pretrained(self.model).to(self.device)
                else:
                    model = ElectraReranker.from_pretrained(self.model).to(self.device)
            else:
                if 't5' in str(self.model):
                    model = CodeT5pEncDecReranker(self.model).to(self.device)

        assert len(qids) == len(pids), (len(qids), len(pids))

        scores = []

        model.eval()
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                for offset in tqdm.tqdm(range(0, len(qids), self.bsize), disable=(not show_progress)):
                    endpos = offset + self.bsize

                    queries_ = [self.queries[qid] for qid in qids[offset:endpos]]
                    try:
                        passages_ = [self.collection[pid] for pid in pids[offset:endpos]]
                    except:
                        print(pids[offset:endpos])
                        raise Exception

                    features = tokenizer(queries_, passages_, padding='longest', truncation=True,
                                            return_tensors='pt', max_length=self.maxlen).to(self.device)

                    if self.model == DEFAULT_MODEL:
                        scores.append(model(**features).logits.flatten())
                    else:
                        scores.append(model(features))

        scores = torch.cat(scores)
        scores = scores.tolist()

        Run().print(f'Returning with {len(scores)} scores')

        return scores


# LONG-TERM TODO: This can be sped up by sorting by length in advance.
# def _score_pairs(self, qids, pids, show_progress=False):
#     tokenizer = AutoTokenizer.from_pretrained(self.model)

#     pairs_with_lengths = [(qid, pid, len(self.queries[qid])) for qid, pid in zip(qids, pids)]
#     index_map = sorted(range(len(pairs_with_lengths)), key=pairs_with_lengths.__getitem__)
#     pairs_with_lengths.sort(key=lambda x: x[2])
#     qids, pids, _ = zip(*pairs_with_lengths)

#     [ORIGINAL CODE]

#     scores = [x for _, x in sorted(zip(index_map, scores))]

#     return scores