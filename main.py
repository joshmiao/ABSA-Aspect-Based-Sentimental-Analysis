# Disable CUDA devices to prevent Tensorflow from allocating memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time
from transformers import BertTokenizer, RobertaTokenizerFast
import data_utils
import torch
import torch.nn.functional as func
import torch.optim as optim
import model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.cuda.is_available())
print('There are {0:} CUDA device(s) on this computer.'.format(torch.cuda.device_count()))
device = torch.device('cuda:0')

# prepare model and corresponding tokenizer
model_path = './BERT_model/'
model_type = 'bert-base-uncased'
# model_path = './Roberta_model/'
# model_type = 'roberta-base'

model = model.BertAndLinear(bert_model_path=model_path).to(device)
# model = model.BertAndTransformer(bert_model_type=model_path).to(device)
# model = model.RoBERTaAndLinear(roberta_model_path=model_path).to(device)

tokenizer = BertTokenizer.from_pretrained(model_path)
# tokenizer = RobertaTokenizerFast.from_pretrained(model_path)

# define hyperparameters and prepare data
epoch = 30
batch_size = 16
data_paths = ['./data/Semeval&Twitter/semeval14/Laptops_Train.xml.seg',
              './data/Semeval&Twitter/semeval14/Laptops_Test_Gold.xml.seg',
              './data/Semeval&Twitter/semeval14/Restaurants_Train.xml.seg',
              './data/Semeval&Twitter/semeval14/Restaurants_Test_Gold.xml.seg',
              './data/Semeval&Twitter/acl-14-short-data/train.raw',
              './data/Semeval&Twitter/acl-14-short-data/test.raw',
              './data/Semeval&Twitter/all.txt']
train_dataset = data_utils.load_data(file_path=data_paths[0], tokenizer=tokenizer, batch_size=batch_size,
                                     device=device, max_length=110)
test_dataset = data_utils.load_data(file_path=data_paths[1], tokenizer=tokenizer, batch_size=1,
                                    device=device, max_length=110)
# train_dataset = data_utils.load_data_with_offsets_mapping(file_path=data_paths[0], tokenizer=tokenizer,
#                                                           batch_size=batch_size, device=device, max_length=200)
# test_dataset = data_utils.load_data_with_offsets_mapping(file_path=data_paths[1], tokenizer=tokenizer,
#                                                          batch_size=1, device=device, max_length=200)
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
    tot_loss = torch.tensor(0, device=device, dtype=torch.float32)
    for idx, (x, y) in enumerate(train_dataset):
        y_pred = model(x)
        loss = func.cross_entropy(y_pred.permute(0, 2, 1), y)
        # print('loss =', loss)
        tot_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch{0:} :'.format(_))
    print('avg_training_loss = ', tot_loss / len(train_dataset))
    print('used time = {0:}'.format(time.time() - st))

    model.eval()
    tot_cnt, acc_cnt, true_cnt, pred_cnt, true_pred_cnt = 0, 0, 0, 0, 0
    for idx, (x, y) in enumerate(test_dataset):
        y_pred = model(x)
        y_pred = torch.max(y_pred, dim=2)[1]
        tot_cnt += len(y_pred[0])
        for i in range(len(y_pred[0])):
            if y_pred[0][i] == y[0][i]:
                acc_cnt += 1
            if y_pred[0][i] != 0:
                pred_cnt += 1
                if y_pred[0][i] == y[0][i]:
                    true_pred_cnt += 1
            if y[0][i] != 0:
                true_cnt += 1
        # print(y_pred, y)
    print('tot_cnt={0:} acc_cnt={1:} true_cnt={2:} pred_cnt={3:} true_pred_cnt={4:}'
          .format(tot_cnt, acc_cnt, true_cnt, pred_cnt, true_pred_cnt))
    eps = 1e-8
    acc = acc_cnt / tot_cnt
    pre = true_pred_cnt / (pred_cnt + eps)
    rec = true_pred_cnt / (true_cnt + eps)
    f1 = 2 * pre * rec / (pre + rec + eps)
    print('acc.={0:.6f} pre.={1:.6f} rec.={2:.6f} f1={3:.6f}'.format(acc, pre, rec, f1))
    print('----------------------------------------------------------------------------------------')
torch.save(model, 'model.pt')
