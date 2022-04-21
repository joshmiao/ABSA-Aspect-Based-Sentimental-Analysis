from transformers import BertModel, BertConfig, BertTokenizer, BertForPreTraining
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup


class BertAndLinear(nn.Module):
    def __init__(self, bert_model_path, type_nums=10):
        super(BertAndLinear, self).__init__()
        self.config = BertConfig.from_pretrained(bert_model_path + 'bert_config.json')
        self.bert = BertModel.from_pretrained(bert_model_path + 'bert_model.ckpt.index',
                                              from_tf=True, config=self.config)
        self.fc = nn.Linear(self.config.hidden_size, type_nums)  # 直接分类

    def forward(self, input_ids):
        embeddings = self.bert(input_ids)
        weights = self.fc(embeddings)
        prob = fn.softmax(weights, dim=-1)
        return prob

