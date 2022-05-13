import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


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
    label_dict = {'0': 1, '1': 3, '-1': 5}
    data_file = open(file_path, 'r', encoding='UTF-8')
    last_text = ''
    m = -1
    while True:
        # get every 3 lines and truncate the last '\n'
        text = data_file.readline()[:-1]
        target_word = data_file.readline()[:-1]
        target_label = data_file.readline()[:-1]
        # print(text)
        # print(target_word)
        # print(target_label)
        if not text:
            break
        # print('now w = ', target_word, 't = ', target_label)
        assert target_label in ['0', '-1', '1']
        label = [0]
        for word in text.split(' '):
            if word == '$T$':
                tokenized_word = tokenizer.encode(target_word, add_special_tokens=False)
                label += [label_dict[target_label]] + [label_dict[target_label] + 1] * (len(tokenized_word) - 1)
            else:
                tokenized_word = tokenizer.encode(word, add_special_tokens=False)
                label += [0] * len(tokenized_word)
        text = text.replace('$T$', target_word)
        label.append(0)
        sentence = tokenizer.encode(text, add_special_tokens=True)

        # print(len(sentence), len(labels))
        if len(sentence) > m:
            m = len(sentence)
        print(len(sentence))
        print(len(label))
        assert len(sentence) == len(label) and len(sentence) <= max_length and len(label) <= max_length
        sentence += [0] * (max_length - len(sentence))
        label += [0] * (max_length - len(label))

        sentence = torch.tensor(sentence, device=device)
        label = torch.tensor(label, device=device)
        # print(target_word)
        # print(text.replace('$T$', target_word))
        # print(last_text)
        if text == last_text:
            data[-1][1] += label
        else:
            data.append([sentence, label])
        last_text = text
    print('max length = ', m)
    return DataLoader(MyDataset(data), batch_size=batch_size, shuffle=True)


def load_data_with_offsets_mapping(tokenizer, file_path, batch_size, device, max_length=500):
    encoded, labels = [], []
    label_dict = {'0': 1, '1': 3, '-1': 5}
    data_file = open(file_path, 'r', encoding='UTF-8')
    last_text = ''
    m = -1
    while True:
        # get every 3 lines and truncate the last '\n'
        text = data_file.readline()[:-1]
        target_word = data_file.readline()[:-1]
        target_label = data_file.readline()[:-1]
        if not text:
            break
        assert target_label in ['0', '-1', '1']
        target_offsets_begin = text.find('$T$')
        target_offsets_end = target_offsets_begin + len(target_word)
        text = text.replace('$T$', target_word)
        encoded_text = tokenizer(text, return_offsets_mapping=True)['input_ids']
        mapping = tokenizer(text, return_offsets_mapping=True)['offset_mapping']
        label = []
        begin = 1
        for idx in range(len(encoded_text)):
            if mapping[idx][0] >= target_offsets_begin and mapping[idx][1] <= target_offsets_end:
                if begin == 1:
                    label.append(label_dict[target_label])
                    begin = 0
                else:
                    label.append(label_dict[target_label] + 1)
            else:
                label.append(0)
        assert len(encoded_text) == len(label) and len(encoded_text) <= max_length and len(label) <= max_length
        if len(encoded_text) > m:
            m = len(encoded_text)
        encoded_text += [0] * (max_length - len(encoded_text))
        label += [0] * (max_length - len(label))
        encoded_text = torch.tensor(encoded_text, device=device)
        label = torch.tensor(label, device=device)
        if text == last_text:
            assert labels[-1].size() == label.size()
            labels[-1] += label
        else:
            encoded.append(encoded_text)
            labels.append(label)
        last_text = text
    print('max length = ', m)
    return DataLoader(TensorDataset(torch.stack(encoded), torch.stack(labels)),
                      batch_size=batch_size, shuffle=True)
