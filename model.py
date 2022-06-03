import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig


class BertAndLinear(nn.Module):
    def __init__(self, bert_model_type=None, bert_model_path=None, type_nums=7, noise_lambda=0):
        super(BertAndLinear, self).__init__()
        if bert_model_path is not None:
            self.bert_config = BertConfig.from_pretrained(bert_model_path)
            self.bert = BertModel.from_pretrained(bert_model_path, config=self.bert_config)
        elif bert_model_type is not None:
            self.bert_config = BertConfig.from_pretrained(bert_model_type)
            self.bert = BertModel.from_pretrained(bert_model_type)
        self.fc = nn.Linear(self.bert_config.hidden_size, type_nums)
        if noise_lambda != 0:
            for name, para in self.named_parameters():
                self.state_dict()[name][:] += (torch.rand(para.size()) - 0.5) * noise_lambda * torch.std(para)

    def forward(self, input_ids):
        embeddings = self.bert(input_ids)[0]
        weights = self.fc(embeddings)
        return weights


class BertAndTransformer(nn.Module):
    def __init__(self, bert_model_type=None, bert_model_path=None, type_nums=7, noise_lambda=0):
        super(BertAndTransformer, self).__init__()
        if bert_model_path is not None:
            self.bert_config = BertConfig.from_pretrained(bert_model_path)
            self.bert = BertModel.from_pretrained(bert_model_path,
                                                  config=self.bert_config)
        elif bert_model_type is not None:
            self.bert_config = BertConfig.from_pretrained(bert_model_type)
            self.bert = BertModel.from_pretrained(bert_model_type)
        self.tfm = nn.TransformerEncoderLayer(self.bert_config.hidden_size, nhead=8)
        self.fc = nn.Linear(self.bert_config.hidden_size, type_nums)
        if noise_lambda != 0:
            for name, para in self.named_parameters():
                self.state_dict()[name][:] += (torch.rand(para.size()) - 0.5) * noise_lambda * torch.std(para)

    def forward(self, input_ids):
        embeddings = self.bert(input_ids)[0]
        encoded = self.tfm(embeddings)
        weights = self.fc(encoded)
        return weights


class RoBERTaAndLinear(nn.Module):
    def __init__(self, roberta_model_type=None, roberta_model_path=None, type_nums=7, noise_lambda=0):
        super(RoBERTaAndLinear, self).__init__()
        if roberta_model_path is not None:
            self.roberta_config = RobertaConfig.from_pretrained(roberta_model_path)
            self.roberta = RobertaModel.from_pretrained(roberta_model_path,
                                                        config=self.roberta_config)
        elif roberta_model_type is not None:
            self.roberta_config = RobertaConfig.from_pretrained(roberta_model_type)
            self.roberta = RobertaModel.from_pretrained(roberta_model_type)
        self.fc = nn.Linear(self.roberta_config.hidden_size, type_nums)
        if noise_lambda != 0:
            for name, para in self.named_parameters():
                self.state_dict()[name][:] += (torch.rand(para.size()) - 0.5) * noise_lambda * torch.std(para)

    def forward(self, input_ids):
        embeddings = self.roberta(input_ids)[0]
        weights = self.fc(embeddings)
        return weights


class RoBERTaAndTransformer(nn.Module):
    def __init__(self, roberta_model_type=None, roberta_model_path=None, type_nums=7, noise_lambda=0):
        super(RoBERTaAndTransformer, self).__init__()
        if roberta_model_path is not None:
            self.roberta_config = RobertaConfig.from_pretrained(roberta_model_path)
            self.roberta = RobertaModel.from_pretrained(roberta_model_path,
                                                        config=self.roberta_config)
        elif roberta_model_type is not None:
            self.roberta_config = RobertaConfig.from_pretrained(roberta_model_type)
            self.roberta = RobertaModel.from_pretrained(roberta_model_type)
        self.tfm = nn.TransformerEncoderLayer(self.roberta_config.hidden_size, nhead=8)
        self.fc = nn.Linear(self.roberta_config.hidden_size, type_nums)
        if noise_lambda != 0:
            for name, para in self.named_parameters():
                self.state_dict()[name][:] += (torch.rand(para.size()) - 0.5) * noise_lambda * torch.std(para)

    def forward(self, input_ids):
        embeddings = self.roberta(input_ids)[0]
        encoded = self.tfm(embeddings)
        weights = self.fc(encoded)
        return weights
