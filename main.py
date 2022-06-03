import os
# Disable CUDA devices to prevent Tensorflow from allocating GPU memory
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from transformers import BertTokenizer, RobertaTokenizerFast
import data_utils
import torch
import matplotlib.pyplot as plt

import model
import train

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('CUDA is' + (' ' if torch.cuda.is_available() else ' not ') + 'available.')
print('There are {0:} CUDA device(s) on this computer.'.format(torch.cuda.device_count()))
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

bert_model_path = './pretrained_model/BERT_model/'
bert_model_type = 'bert-base-uncased'
roberta_model_path = './pretrained_model/Roberta_model/'
roberta_model_type = 'roberta-base'

# define hyperparameters
epoch = 40
batch_size = 16

# data_paths
data_paths = ['./data/Semeval&Twitter/semeval14/Laptops_Train.xml.seg',
              './data/Semeval&Twitter/semeval14/Laptops_Test_Gold.xml.seg',
              './data/Semeval&Twitter/semeval14/Restaurants_Train.xml.seg',
              './data/Semeval&Twitter/semeval14/Restaurants_Test_Gold.xml.seg',
              './data/Semeval&Twitter/acl-14-short-data/train.raw',
              './data/Semeval&Twitter/acl-14-short-data/test.raw',
              './data/Semeval&Twitter/all.txt']

# prepare model and corresponding tokenizer
BL_model = model.BertAndLinear(bert_model_path=bert_model_path, noise_lambda=0.2).to(device)
BT_model = model.BertAndTransformer(bert_model_type=bert_model_path, noise_lambda=0.2).to(device)
RL_model = model.RoBERTaAndLinear(roberta_model_path=roberta_model_path, noise_lambda=0.2).to(device)
RT_model = model.RoBERTaAndTransformer(roberta_model_path=roberta_model_path, noise_lambda=0.2).to(device)

bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
roberta_tokenizer = RobertaTokenizerFast.from_pretrained(roberta_model_path)

bert_train_dataset = data_utils.load_data(file_path=data_paths[2], tokenizer=bert_tokenizer, batch_size=batch_size,
                                          device=device, max_length=110)
bert_test_dataset = data_utils.load_data(file_path=data_paths[3], tokenizer=bert_tokenizer, batch_size=1,
                                         device=device, max_length=110)
roberta_train_dataset = data_utils.load_data_with_offsets_mapping(file_path=data_paths[2], tokenizer=roberta_tokenizer,
                                                                  batch_size=batch_size, device=device, max_length=110)
roberta_test_dataset = data_utils.load_data_with_offsets_mapping(file_path=data_paths[3], tokenizer=roberta_tokenizer,
                                                                 batch_size=1, device=device, max_length=110)
# view data
print(len(bert_train_dataset))
for idx, (x, y) in enumerate(bert_train_dataset):
    if idx >= 1:
        break
    print(idx, x, y)

BL_loss, BL_f1 = train.train_model(model=BL_model, device=device, train_dataset=bert_train_dataset,
                                   test_dataset=bert_test_dataset, epoch=epoch)
BT_loss, BT_f1 = train.train_model(model=BT_model, device=device, train_dataset=bert_train_dataset,
                                   test_dataset=bert_test_dataset, epoch=epoch)
RL_loss, RL_f1 = train.train_model(model=RL_model, device=device, train_dataset=roberta_train_dataset,
                                   test_dataset=roberta_test_dataset, epoch=epoch)
RT_loss, RT_f1 = train.train_model(model=RT_model, device=device, train_dataset=roberta_train_dataset,
                                   test_dataset=roberta_test_dataset, epoch=epoch)

print('max F1_measure for BertAndLinear = {0:.4f}'.format(max(BL_f1)))
print('max F1_measure for BertAndTransformer = {0:.4f}'.format(max(BT_f1)))
print('max F1_measure for RoBERTaAndLinear = {0:.4f}'.format(max(RL_f1)))
print('max F1_measure for RoBERTaAndTransformer = {0:.4f}'.format(max(RT_f1)))

fig, axs = plt.subplots(2, 1, figsize=(15, 10))
axs[0].set_title('Training Loss', fontsize=24)
axs[0].set_xlabel('Number of epoch')
axs[0].set_ylabel('Training Loss')
axs[0].set_xticks(range(epoch))
axs[0].plot(range(epoch), BL_loss, label='bert_linear_loss')
axs[0].plot(range(epoch), BT_loss, label='bert_tf_loss')
axs[0].plot(range(epoch), RL_loss, label='roberta_linear_loss')
axs[0].plot(range(epoch), RT_loss, label='roberta_tf_loss')
axs[0].legend()

axs[1].set_title('F1_measure', fontsize=24)
axs[1].set_xlabel('Number of epoch')
axs[1].set_ylabel('F1_measure')
axs[1].set_xticks(range(epoch))
axs[1].plot(range(epoch), BL_f1, label='bert_linear_f1')
axs[1].plot(range(epoch), BT_f1, label='bert_tf_f1')
axs[1].plot(range(epoch), RL_f1, label='roberta_linear_f1')
axs[1].plot(range(epoch), RT_f1, label='roberta_tf_f1')
axs[1].legend()
fig.tight_layout(pad=3, h_pad=8.0)

plt.show()
fig.savefig('fig.png', format='png')
