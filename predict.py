import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from transformers import BertTokenizer
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.cuda.is_available())
print(torch.cuda.device_count())
device = torch.device('cuda:0')

# prepare model and corresponding tokenizer
bert_model_path = './pretrained_model/BERT_model/'
bert_model_type = 'bert-base-uncased'
model = torch.load('model.pt').to(device)
tokenizer = BertTokenizer.from_pretrained(bert_model_path)

polarity = {1: 'Neutral', 3: 'Positive', 5: 'Negative'}
cata = {1: 1, 2: 1, 3: 3, 4: 3, 5: 5, 6: 5}
while True:
    s = input()
    s = s.replace(',', ' , ')
    s = s.replace('.', ' . ')
    s = s.replace('!', ' ! ')
    s = s.replace('?', ' ? ')
    s = s.replace('(', ' ( ')
    s = s.replace(')', ' ) ')
    s = s.replace(';', ' ; ')
    s = s.replace(':', ' : ')
    x = torch.tensor([tokenizer.encode(s)], device=device)
    y_pred = model(x)
    y_pred = torch.max(y_pred, dim=2)[1]
    idx, tag = 0, 0
    aspect = ''
    print(y_pred)
    for word in s.split(' '):
        if word == '':
            continue
        if tag != 0:
            if y_pred[0][idx] == tag + 1:
                aspect += word + ' '
            elif y_pred[0][idx] == 0:
                print('Aspect: {0:} Polarity: {1:}'.format(aspect, polarity[tag]))
                tag = 0
                aspect = ''
            else:
                print('Aspect: {0:} Polarity: {1:}'.format(aspect, polarity[tag]))
                tag = cata[y_pred[0][idx].item()]
                aspect = word + ' '
        else:
            if y_pred[0][idx] != 0:
                tag = cata[y_pred[0][idx].item()]
                aspect += word + ' '
        idx += len(tokenizer.encode(word))
    if tag != 0:
        print('Aspect: {0:} Polarity: {1:}'.format(aspect, polarity[tag]))
        tag = 0
        aspect = ''
