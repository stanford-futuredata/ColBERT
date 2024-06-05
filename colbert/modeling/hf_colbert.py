import importlib
import torch
from turtle import forward
from unicodedata import name
from typing import Optional
import torch.nn as nn
import transformers
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer, AutoModel, AutoConfig
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers import XLMRobertaModel, XLMRobertaConfig
from transformers import ElectraModel, ElectraPreTrainedModel
from transformers import DebertaV2Model, DebertaV2PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from colbert.utils.utils import torch_load_dnn

class XLMRobertaPreTrainedModel(RobertaPreTrainedModel):
    """
    This class overrides [`RobertaModel`]. Please check the superclass for the appropriate documentation alongside
    usage examples.
    """

    config_class = XLMRobertaConfig


base_class_mapping={
    "roberta-base": RobertaPreTrainedModel,
    "google/electra-base-discriminator": ElectraPreTrainedModel,
    "xlm-roberta-base": XLMRobertaPreTrainedModel,
    "xlm-roberta-large": XLMRobertaPreTrainedModel,
    "bert-base-uncased": BertPreTrainedModel,
    "bert-large-uncased": BertPreTrainedModel,
    "microsoft/mdeberta-v3-base": DebertaV2PreTrainedModel,
    "bert-base-multilingual-uncased": BertPreTrainedModel


}

model_object_mapping = {
    "roberta-base": RobertaModel,
    "google/electra-base-discriminator": ElectraModel,
    "xlm-roberta-base": XLMRobertaModel,
    "xlm-roberta-large": XLMRobertaModel,
    "bert-base-uncased": BertModel,
    "bert-large-uncased": BertModel,
    "microsoft/mdeberta-v3-base": DebertaV2Model,
    "bert-base-multilingual-uncased": BertModel

}



def find_class_names(model_type, class_type):
    transformers_module = dir(transformers)
    model_type = model_type.replace("-", "").lower()
    for item in transformers_module:
        if model_type + class_type == item.lower():
            return item

    return None

class PruningHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.dim = 32
        self.heads = 4
        self.query = nn.Linear(hidden_size, self.dim)
        self.key = nn.Linear(hidden_size, self.dim)
        self.value = nn.Linear(hidden_size, self.dim)
        self.attention = nn.MultiheadAttention(self.dim, self.heads, batch_first=True)
        self.linear = nn.Linear(self.dim, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None].expand(-1, attention_mask.shape[-1], -1).repeat(self.heads, 1, 1)

        out = self.attention.forward(query, key, value, attn_mask=attention_mask)

        return self.linear(out[0])


def class_factory(name_or_path):
    loadedConfig  = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)

    if getattr(loadedConfig, "auto_map", None) is None:
        model_type = loadedConfig.model_type
        pretrained_class = find_class_names(model_type, 'pretrainedmodel')
        model_class = find_class_names(model_type, 'model')
        
        if pretrained_class is not None:
            pretrained_class_object = getattr(transformers, pretrained_class)
        elif model_type == 'xlm-roberta':
            pretrained_class_object = XLMRobertaPreTrainedModel
        elif base_class_mapping.get(name_or_path) is not None:
            pretrained_class_object = base_class_mapping.get(name_or_path)
        else:
            raise ValueError("Could not find correct pretrained class for the model type {model_type} in transformers library")

        if model_class != None:
            model_class_object = getattr(transformers, model_class)
        elif model_object_mapping.get(name_or_path) is not None:
            model_class_object = model_object_mapping.get(name_or_path)
        else:
            raise ValueError("Could not find correct model class for the model type {model_type} in transformers library")
    else:
        assert "AutoModel" in loadedConfig.auto_map, "The custom model should have AutoModel class in the config.automap"
        model_class = loadedConfig.auto_map["AutoModel"]
        assert model_class.endswith("Model")
        pretrained_class = model_class.replace("Model", "PreTrainedModel")
        model_class_object = get_class_from_dynamic_module(model_class, name_or_path)
        pretrained_class_object = get_class_from_dynamic_module(pretrained_class, name_or_path)


    class HF_ColBERT(pretrained_class_object):
        """
            Shallow wrapper around HuggingFace transformers. All new parameters should be defined at this level.

            This makes sure `{from,save}_pretrained` and `init_weights` are applied to new parameters correctly.
        """
        _keys_to_ignore_on_load_unexpected = [r"cls"]

        def __init__(self, config, colbert_config):
            super().__init__(config)

            self.config = config
            self.dim = colbert_config.dim
            self.linear = nn.Linear(config.hidden_size, colbert_config.dim, bias=False)
            self.pruning_head = None
            if colbert_config.prune:
                # self.pruning_head = nn.Sequential(
                #     nn.Linear(config.hidden_size, config.hidden_size),
                #     nn.ReLU(),
                #     nn.Linear(config.hidden_size, 1)
                # )
                self.pruning_head = PruningHead(config.hidden_size)
                # self.pruning_head = nn.Linear(config.hidden_size, 1)
            setattr(self,self.base_model_prefix, model_class_object(config))

            # if colbert_config.relu:
            #     self.score_scaler = nn.Linear(1, 1)

            self.init_weights()

            # if colbert_config.relu:
            #     self.score_scaler.weight.data.fill_(1.0)
            #     self.score_scaler.bias.data.fill_(-8.0)

        @property
        def LM(self):
            base_model_prefix = getattr(self, "base_model_prefix")
            return getattr(self, base_model_prefix)


        @classmethod
        def from_pretrained(cls, name_or_path, colbert_config):
            if name_or_path.endswith('.dnn'):
                dnn = torch_load_dnn(name_or_path)
                base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')

                obj = super().from_pretrained(base, state_dict=dnn['model_state_dict'], colbert_config=colbert_config)
                obj.base = base

                return obj

            obj = super().from_pretrained(name_or_path, colbert_config=colbert_config)
            obj.base = name_or_path

            return obj

        @staticmethod
        def raw_tokenizer_from_pretrained(name_or_path):
            if name_or_path.endswith('.dnn'):
                dnn = torch_load_dnn(name_or_path)
                base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')

                obj = AutoTokenizer.from_pretrained(base)
                obj.base = base

                return obj

            obj = AutoTokenizer.from_pretrained(name_or_path)
            obj.base = name_or_path

            return obj

    return HF_ColBERT
