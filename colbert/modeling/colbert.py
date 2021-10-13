from colbert.search.strided_tensor import StridedTensor
from colbert.utils.utils import print_message
from colbert.modeling.base_colbert import BaseColBERT

import torch
import string


class ColBERT(BaseColBERT):
    """
        This class handles the basic encoding and scoring operations in ColBERT. It is used for training.
    """

    def __init__(self, name='bert-base-uncased', colbert_config=None):
        super().__init__(name, colbert_config)

        if self.colbert_config.mask_punctuation:
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]}

    def forward(self, Q, D):
        return self.score(self.query(*Q), self.doc(*D))

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids), device=self.device).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2).half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D

    def score(self, Q, D):
        # TODO: Flip multiplication order and permute?
        if self.colbert_config.similarity == 'cosine':
            score = Q @ D.permute(0, 2, 1)

            if self.colbert_config.relu:
                score = torch.nn.functional.relu(score)
                # FIXME: TODO: Apply ReLU in the scoring below, irrespective of training config. And check if anything changes.

            score = score.max(2).values.sum(1)

            # if self.colbert_config.relu and self.training:
            #     score = score - self.colbert_config.query_maxlen // 4
            #     # print('BEFORE:  ', score)
            #     # score = self.score_scaler(score.unsqueeze(-1)).squeeze(-1)
            #     # print('AFTER:   ', score)

            return score

        assert self.colbert_config.similarity == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask


# TODO: In Query/DocTokenizer, use colbert.raw_tokenizer


def colbert_score(Q, D_padded, D_mask):
    """
        Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
        If Q.size(0) is 1, the matrix will be compared with all passages.
        Otherwise, each query matrix will be compared against the *aligned* passage.

        EVENTUALLY: Consider masking with -inf for the maxsim (or enforcing a ReLU).
    """

    Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()
    # print_message(f'#> colbert_score{Q.size(), D_padded.size(), D_mask.size()}')

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]

    scores = (D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1))

    # print(scores.size(), D_mask.size())

    scores = scores * D_mask.to(dtype=D_padded.dtype)  # .unsqueeze(2)
    scores = scores.max(1).values.sum(-1)#.cpu()

    return scores


def colbert_score_packed(Q, D_packed, D_lengths):
    """
        Works with a single query only.
    """

    Q, D_padded, D_lengths = Q.cuda(), D_packed.cuda(), D_lengths.cuda()

    Q = Q.squeeze(0)

    assert Q.dim() == 2, Q.size()
    assert D_packed.dim() == 2, D_packed.size()

    scores = D_packed @ Q.to(dtype=D_padded.dtype).T

    scores_padded, scores_mask = StridedTensor(scores, D_lengths).as_padded_tensor()

    scores_padded = scores_padded * scores_mask.to(dtype=scores_padded.dtype)  # .unsqueeze(2)
    scores_padded = scores_padded.max(1).values

    scores = scores_padded.sum(-1).cpu()

    return scores
