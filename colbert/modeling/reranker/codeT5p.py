import copy
import torch.nn as nn

from transformers import T5PreTrainedModel, T5EncoderModel, AutoTokenizer
from transformers.models.t5.modeling_t5 import T5Stack

class CodeT5pReranker(T5PreTrainedModel):
    """
        Shallow wrapper around HuggingFace transformers. All new parameters should be defined at this level.
        
        This makes sure `{from,save}_pretrained` and `init_weights` are applied to new parameters correctly.
    """
    _keys_to_ignore_on_load_unexpected = [r"cls"]

    def __init__(self, config):
        super().__init__(config)

        print(config)

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.linear = nn.Linear(config.hidden_size, 1)
        self.raw_tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)

        self.init_weights()

    def forward(self, encoding):
        outputs = self.encoder(encoding.input_ids,
                               attention_mask=encoding.attention_mask)[0]

        scores = self.linear(outputs[:, 0]).squeeze(-1)

        return scores
    
    def save(self, path):
        assert not path.endswith('.dnn'), f"{path}: We reserve *.dnn names for the deprecated checkpoint format."

        self.save_pretrained(path)
        self.raw_tokenizer.save_pretrained(path)