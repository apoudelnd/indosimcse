import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModel, BertPreTrainedModel, PreTrainedModel, AutoModelForSequenceClassification
import logging
from transformers.models.electra.modeling_electra import (
    ElectraClassificationHead,
    ElectraModel,
)

from transformers.models.bert.modeling_bert import BertModel


from sklearn.utils import class_weight


logger = logging.getLogger(__name__)


class NLIPrediction(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        base_model = AutoModel.from_config(config)
        self.base_model_prefix = base_model.base_model_prefix
        setattr(self, self.base_model_prefix, base_model)
        self.cls = ElectraClassificationHead(config)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels, **kwargs):

        model = self.base_model
        hidden = model(input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids).last_hidden_state


        logits = self.cls(hidden)

        output_dict = dict()

        if labels is not None:

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1).long())
            output_dict['loss'] = loss

        output_dict['logits'] = torch.softmax(logits, 1)

        return output_dict
     


supported_models = {"nli-bert": NLIPrediction}