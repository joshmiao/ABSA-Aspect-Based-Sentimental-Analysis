import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time
from transformers import BertModel, BertConfig, BertTokenizer, BertForPreTraining
import data_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.cuda.is_available())
print(torch.cuda.device_count())
device = torch.device('cuda:0')

# prepare model and corresponding tokenizer
model_path = './BERT_model/uncased_L-12_H-768_A-12/'
model_type = 'bert-base-uncased'
model = model.BertAndLinear(bert_model_type=model_type).to(device)
tokenizer = BertTokenizer.from_pretrained(model_path + 'vocab.txt')

# define hyperparameters and prepare data
epoch = 5
batch_size = 5
data_paths = ['./data/Semeval&Twitter/semeval14/Laptops_Train.xml.seg',
              './data/Semeval&Twitter/semeval14/Laptops_Test_Gold.xml.seg']
train_dataset = data_utils.load_data(file_path=data_paths[0], tokenizer=tokenizer, batch_size=batch_size,
                                     device=device, max_length=500)
test_dataset = data_utils.load_data(file_path=data_paths[1], tokenizer=tokenizer, batch_size=batch_size,
                                    device=device, max_length=500)
# view data
print(len(train_dataset))
for idx, (x, y) in enumerate(train_dataset):
    if idx >= 1:
        break
    print(idx, x, y)

# define optimizer
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)


for _ in range(epoch):
    model.train()
    st = time.time()
    for idx, (x, y) in enumerate(train_dataset):
        y_pred = model(x)
        # print(y_pred)
        # print(y_pred.size())
        # print(y)
        # print(y.size())
        loss = func.cross_entropy(y_pred.permute(0, 2, 1), y)
        print('loss =', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('used time = {0:}'.format(time.time() - st))
    model.eval()
