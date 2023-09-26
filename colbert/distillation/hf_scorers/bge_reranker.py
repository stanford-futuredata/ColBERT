import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from colbert.infra import Run
from colbert.distillation.hf_scorers.base import BaseHFScorer

class BGERerankerScorer(BaseHFScorer):
    def __init__(self, queries, collection, model, bsize=32, maxlen=180, query_instruction=None):
        super().__init__(queries, collection, model, bsize=bsize, maxlen=maxlen)

        self.query_instruction = query_instruction

    def score(self, qids, pids, show_progress=False):
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        model = AutoModelForSequenceClassification.from_pretrained(self.model).cuda()
        
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
                    
                    pairs = [[q,p] for q, p in zip(queries_, passages_)]

                    features = tokenizer(pairs, padding='longest', truncation=True,
                                return_tensors='pt', max_length=self.maxlen).to(self.device)

                    batch_scores = model(**features, return_dict=True).logits.view(-1, ).float()

                    scores.append(batch_scores)


        scores = torch.cat(scores)
        scores = scores.tolist()

        Run().print(f'Returning with {len(scores)} scores')

        return scores