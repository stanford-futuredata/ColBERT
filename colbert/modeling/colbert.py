import string
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast, RobertaPreTrainedModel, XLMRobertaModel, XLMRobertaTokenizer
from colbert.parameters import DEVICE
from colbert.modeling.gradient_reversal_layer import GradientReversalFunction
from torchmetrics.classification import BinaryAccuracy

class ColBERT(RobertaPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation, dim=128, 
                 similarity_metric='cosine', use_gradient_reversal=False, num_of_languages=2):

        super(ColBERT, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.tokenizer.add_tokens(['[unused1]'])
        self.tokenizer.add_tokens(['[unused2]'])
        
        self.model = AutoModel.from_pretrained("xlm-roberta-base")
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        '''
        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}
        '''
        
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)
        
        # Add Gradient Reversal Layer
        self._use_gradient_reversal = use_gradient_reversal
        if self._use_gradient_reversal:
            self._gradient_reverse_lambda = 0
            self._gradient_reverse_loss = torch.nn.CrossEntropyLoss(reduction="mean")
            self._language_predictor_ff = torch.nn.Linear(in_features=768, out_features=num_of_languages)

        #self._lp_acc = CategoricalAccuracy(top_k=1)
        self.src_ff_predictions_list = []
        self.trg_ff_predictions_list = []
        self.LP_metric = BinaryAccuracy()

        self.init_weights()

    #def forward(self, Q, D):
    #    score = self.score(self.query(*Q), self.doc(*D))
    #    return score

    def forward(self, Q, D, S, T):
        ir_score = self.score(self.query(*Q), self.doc(*D))
        lp_loss = 0.0
        src_acc=0.0
        trg_acc=0.0
        total_acc=0.0
        
        if self._use_gradient_reversal:
            src_lp_loss, src_ff_predictions = self.language_prediction(*S, lang = "src")
            trg_lp_loss, trg_ff_predictions = self.language_prediction(*T, lang = "trg" )
            lp_loss = (src_lp_loss + trg_lp_loss)/2
        
            src_ff_predictions = torch.argmax(src_ff_predictions, dim=1)
            trg_ff_predictions = torch.argmax(trg_ff_predictions, dim=1)
            src_acc, trg_acc, total_acc = self.lp_accuracy(src_ff_predictions, trg_ff_predictions)
            
            #print("source language prediction : ", src_acc)
            #print("target language prediction : ", trg_acc)
            #print("avg of language prediction : ", total_acc)

        return ir_score, lp_loss, (src_acc, trg_acc, total_acc)
    
    def lp_accuracy(self, src_ff_predictions, trg_ff_predictions):
        
        self.src_ff_predictions_list.extend(src_ff_predictions)
        self.trg_ff_predictions_list.extend(trg_ff_predictions)
        
        src_acc = self.LP_metric(torch.tensor(self.src_ff_predictions_list), torch.tensor([0]*len(self.src_ff_predictions_list))).item()
        trg_acc = self.LP_metric(torch.tensor(self.trg_ff_predictions_list), torch.tensor([0]*len(self.trg_ff_predictions_list))).item()
        total_acc = (src_acc + trg_acc)/2
    
        return src_acc, trg_acc, total_acc

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.model(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        #print("query : ", Q.shape)
        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.model(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)
        
        #print("document : ", D.shape)

        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D
    
    def language_prediction(self, input_ids, attention_mask, lang, keep_dims=True):
        input_ids, attention_mask = input_ids[:32,:].to(DEVICE), attention_mask[:32,:].to(DEVICE)

        encoder_outputs = self.model(input_ids, attention_mask=attention_mask)[0]
        #print(encoder_outputs.shape)

        encoder_outputs = encoder_outputs[:,0,:].squeeze()
        #print(encoder_outputs.shape)

        
        #D = self._language_predictor_ff(D)
        #mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        #D = D * mask
        
        encoder_outputs = torch.nn.functional.normalize(encoder_outputs, p=2, dim=1)
        #print(encoder_outputs.shape)
                
        encoder_outputs = GradientReversalFunction.apply(encoder_outputs, self._gradient_reverse_lambda)
        #print(encoder_outputs.shape)
        
        ff_output = self._language_predictor_ff(encoder_outputs).squeeze(-2)
        #print(ff_output.shape)
        
        if lang == "src":
            target = torch.tensor([0]*32).to(DEVICE)
        elif lang == 'trg':
            target = torch.tensor([1]*32).to(DEVICE)
            
        #print(ff_output[:32,0,:])
        #print(ff_output[:32,0,:].shape)
        #print(ff_output[:32,0,:].view(32, -1).shape)
        #print(target)
        #print(target.shape)

        return self._gradient_reverse_loss(ff_output, target), ff_output

    def score(self, Q, D):
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        #mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        mask = [[(x not in self.skiplist) and (x != 1) for x in d] for d in input_ids.cpu().tolist()]
        
        return mask
