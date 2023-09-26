import torch
import tqdm
from transformers import AutoTokenizer, AutoModel

from colbert.infra import Run
from colbert.distillation.hf_scorers.base import BaseHFScorer

class BGELargeV15Scorer(BaseHFScorer):
    def __init__(self, queries, collection, model, bsize=32, maxlen=180, query_instruction=None):
        super().__init__(queries, collection, model, bsize=bsize, maxlen=maxlen)

        self.query_instruction = query_instruction or "Represent this sentence for searching relevant passages:"

    def score(self, qids, pids, show_progress=False):
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        model = AutoModel.from_pretrained(self.model).to(self.device)
        
        assert len(qids) == len(pids), (len(qids), len(pids))

        scores = []

        model.eval()
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                for offset in tqdm.tqdm(range(0, len(qids), self.bsize), disable=(not show_progress)):
                    endpos = offset + self.bsize

                    if self.query_instruction is None:
                        queries_ = [self.queries[qid] for qid in qids[offset:endpos]]
                    else:
                        queries_ = [self.query_instruction + self.queries[qid] for qid in qids[offset:endpos]]
                    
                    try:
                        passages_ = [self.collection[pid] for pid in pids[offset:endpos]]
                    except:
                        print(pids[offset:endpos])
                        raise Exception

                    query_features = tokenizer(queries_, padding='longest', truncation=True,
                                            return_tensors='pt', max_length=self.maxlen).to(self.device)

                    passage_features = tokenizer(passages_, padding='longest', truncation=True,
                                            return_tensors='pt', max_length=self.maxlen).to(self.device)

                    query_embeddings = model(**query_features)
                    query_embeddings = query_embeddings[0][:, 0]
                    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)

                    passage_embeddings = model(**passage_features)
                    passage_embeddings = passage_embeddings[0][:, 0]
                    passage_embeddings = torch.nn.functional.normalize(passage_embeddings, p=2, dim=1)

                    batch_scores = torch.einsum('nd,nd->n', query_embeddings, passage_embeddings)
                    
                    scores.append(batch_scores)


        scores = torch.cat(scores)
        scores = scores.tolist()

        Run().print(f'Returning with {len(scores)} scores')

        return scores