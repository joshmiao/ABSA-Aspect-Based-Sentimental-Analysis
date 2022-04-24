import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BertAndLinear(nn.Module):
    def __init__(self, bert_model_type=None, bert_model_path=None, type_nums=4):
        super(BertAndLinear, self).__init__()
        if bert_model_path is not None:
            self.config = BertConfig.from_pretrained(bert_model_path + 'bert_config.json')
            self.bert = BertModel.from_pretrained(bert_model_path + 'bert_model.ckpt.index',
                                                  from_tf=True, config=self.config)
        if bert_model_type is not None:
            self.config = BertConfig.from_pretrained(bert_model_type)
            self.bert = BertModel.from_pretrained(bert_model_type)
        self.fc = nn.Linear(self.config.hidden_size, type_nums)  # 直接分类

    def forward(self, input_ids):
        embeddings = self.bert(input_ids)[0]
        weights = self.fc(embeddings)
        return weights
