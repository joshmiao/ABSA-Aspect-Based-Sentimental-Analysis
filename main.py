from transformers import BertModel, BertConfig, BertTokenizer, BertForPreTraining
import data_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

device = torch.device('cuda:0')

model_path = './BERT_model/uncased_L-12_H-768_A-12/'
config = BertConfig.from_json_file(model_path + 'bert_config.json')
tokenizer = BertTokenizer.from_pretrained(model_path + 'vocab.txt')
model = BertForPreTraining.from_pretrained(model_path + 'bert_model.ckpt.index', from_tf=True, config=config)

# data_paths = ['./data/laptop14/', './data/rest14/', './data/rest15/', './data/rest16/']
data_paths = ['./data/Semeval&Twitter/semeval14/Laptops_Train.xml.seg']
train_dataset = data_utils.load_data(file_path=data_paths[0], tokenizer=tokenizer, batch_size=5,
                                    device=device)

for step, data in enumerate(train_dataset):
    if step >= 1:
        break
    print(step, data)
