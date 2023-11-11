import torch.nn as nn

from transformers import DebertaV2PreTrainedModel, DebertaV2Model, AutoTokenizer

class DebertaV2Reranker(DebertaV2PreTrainedModel):
    """
        Shallow wrapper around HuggingFace transformers. All new parameters should be defined at this level.
        
        This makes sure `{from,save}_pretrained` and `init_weights` are applied to new parameters correctly.
    """
    _keys_to_ignore_on_load_unexpected = [r"cls"]

    def __init__(self, config):
        super().__init__(config)
        config.max_position_embeddings = 1024
        print(config)

        self.deberta = DebertaV2Model(config)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.raw_tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)

        self.init_weights()

    def forward(self, encoding):
        outputs = self.deberta(encoding.input_ids,
                               attention_mask=encoding.attention_mask,
                               token_type_ids=encoding.token_type_ids)[0]

        scores = self.linear(outputs[:, 0]).squeeze(-1)

        return scores
    
    def save(self, path):
        assert not path.endswith('.dnn'), f"{path}: We reserve *.dnn names for the deprecated checkpoint format."

        self.save_pretrained(path)
        self.raw_tokenizer.save_pretrained(path)