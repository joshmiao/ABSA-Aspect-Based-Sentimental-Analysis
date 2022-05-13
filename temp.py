from transformers import BertModel, BertConfig, BertTokenizer, BertForPreTraining
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def f1():
    # model_name = 'bert-base-uncased'
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # model = BertModel.from_pretrained(model_name)

    model_path = './BERT_model/uncased_L-12_H-768_A-12/'
    config = BertConfig.from_json_file(model_path + 'bert_config.json')
    tokenizer = BertTokenizer.from_pretrained(model_path + 'vocab.txt')
    model = BertForPreTraining.from_pretrained(model_path + 'bert_model.ckpt.index', from_tf=True, config=config)

    s1 = '[CLS] The [MASK] went to the store.[SEP]'
    s2 = 'Penguins [MASK] flightless birds.[SEP]'
    # s1 = '[CLS]My dog is cute.[SEP]'
    # s2 = 'He likes playing.[SEP]'
    # s1 = '[CLS]My dog is cute.[SEP]'
    # s2 = 'The man went to the store.[SEP]'
    s1 = tokenizer.encode(s1)
    s2 = tokenizer.encode(s2)
    token_type_ids = [0] * len(s1) + [1] * len(s2)
    token_type_ids = torch.tensor([token_type_ids])
    print(token_type_ids)
    input_ids = torch.tensor(s1 + s2).unsqueeze(0)
    print(input_ids)
    outputs = model(input_ids, token_type_ids=token_type_ids)
    print(outputs)
    # last_hidden_states = outputs[0]
    # print(last_hidden_states.size())
    # print(last_hidden_states)


def f2():
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data
            self.len = len(data)

        def __getitem__(self, index):
            return self.data[index][0], self.data[index][1]

        def __len__(self):
            return self.len

    x = torch.tensor([[1, 2, 3], [4, 5, 6], [-1, -2, -3], [-4, -5, -6]])
    y = torch.tensor([[7, 8], [10, 11], [-7, -8], [-10, -11]])
    ds = TensorDataset(x, y)
    dl = DataLoader(dataset=ds, batch_size=2, shuffle=True)

    for step, data in enumerate(dl):
        x, label = data
        print(step, x, label)


if __name__ == '__main__':
    f2()
