import torch
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return self.len


def load_data(tokenizer, file_path,  batch_size, device, max_length=500):
    data = []
    label_dict = {'0': 1, '1': 2, '-1': 3}
    data_file = open(file_path, 'r')
    last_text = ''
    while True:
        # get every 3 lines and truncate the last '\n'
        text = data_file.readline()[:-1]
        target_word = data_file.readline()[:-1]
        target_label = data_file.readline()[:-1]
        # print(text)
        # print(word)
        # print(word_label)
        if not text:
            break
        assert target_label in ['0', '-1', '1']
        labels = []
        for word in text.split(' '):
            if word == '$T$':
                tokenized_word = tokenizer.encode(target_word)
                labels += [label_dict[target_label]] * len(tokenized_word)
            else:
                tokenized_word = tokenizer.encode(word)
                labels += [0] * len(tokenized_word)
        text = text.replace('$T$', target_word)
        sentence = tokenizer.encode(text)

        # print(len(sentence), len(labels))
        assert len(sentence) == len(labels) and len(sentence) <= max_length and len(labels) <= max_length
        sentence += [0] * (max_length - len(sentence))
        labels += [0] * (max_length - len(labels))

        sentence = torch.tensor(sentence, device=device)
        labels = torch.tensor(labels, device=device)
        # print(target_word)
        # print(text.replace('$T$', target_word))
        # print(last_text)
        if text == last_text:
            data[-1][1] += labels
        else:
            data.append([sentence, labels])
        last_text = text
    return DataLoader(MyDataset(data), batch_size=batch_size, shuffle=True)


def load_data_old(tokenizer, file_path, batch_size, device):
    labels_dic = {'O': 0, 'T-POS': 1, 'T-NEU': 2, 'T-NEG': 3}
    data = []
    cnt = 0
    data_file = open(file_path, 'r')
    for _, line in enumerate(data_file):
        line = line.split('####')
        sentence = tokenizer.encode(line[0])
        # sentence = tokenizer.encode(line[0])
        line[1] = line[1][0:-1].split(' ')  # delete the '\n' at the end of each sentence
        labels = []
        for word in line[1]:
            if word.count('=') != 1:
                continue
            if not word.split('=')[1] in labels_dic:
                continue
            length = len(tokenizer.encode(word.split('=')[0]))
            labels += [labels_dic[word.split('=')[1]]] * length

        # print(sentence)
        # print(labels)
        print(len(sentence), len(labels))
        assert len(sentence) <= 500 and len(labels) <= 500
        if len(sentence) != len(labels):
            print('error at line', _)
            print(sentence)
            print(labels)
            cnt += 1
        else:
            sentence += [0] * (500 - len(sentence))
            labels += [0] * (500 - len(labels))
            labels = torch.tensor(labels, device=device)
            sentence = torch.tensor(sentence, device=device)
            data.append([sentence, labels])
    print(cnt)
    # print(data)
    return DataLoader(MyDataset(data), batch_size=batch_size)
