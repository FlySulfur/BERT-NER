import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from transformers import AdamW
from seqeval.metrics import f1_score


def read_data(text_path, tag_path):
    """

    :param text_path: path of the text
    :param tag_path: path of the tag, according to the text
    :return: two lists
    """
    all_data = []
    all_tag_ = []
    text_f = open(text_path, "r", encoding="utf-8")
    tag_f = open(tag_path, "r", encoding="utf-8")
    line = text_f.readline().replace(" ", "").replace("\n", "")
    while line:
        line_ls = list(line)
        line = text_f.readline().replace(" ", "").replace("\n", "")
        all_data.append(line_ls)
    line = tag_f.readline()
    while line:
        line = line.strip().split(" ")
        line_ls = list(line)
        line = tag_f.readline()
        all_tag_.append(line_ls)
    return all_data, all_tag_


def get_tag_index(tags):
    """

    :param tags: the list of tags
    :return: dictionary and list of tags with index
    """
    tag_index = {"PAD": 0, "UNK": 1}
    for tag in tags:
        for item in tag:
            if item not in tag_index:
                tag_index[item] = len(tag_index)
    return tag_index, list(tag_index)


class BertDataset(Dataset):
    def __init__(self, all_data, all_tag_, tag_index, max_len_, tokenizer_):
        self.all_data = all_data
        self.all_tag = all_tag_
        self.tag_index = tag_index
        self.max_len = max_len_
        self.tokenizer = tokenizer_

    def __getitem__(self, item):
        data = self.all_data[item]
        tag_ = self.all_tag[item][:self.max_len]
        data_code = self.tokenizer.encode(data, add_special_tokens=True, max_length=self.max_len+2, padding="max_length"
                                          , truncation=True, return_tensors="pt")
        tag_code = [0] + [self.tag_index.get(i, 1) for i in tag_] + [0] + [0]*(self.max_len-len(data))
        tag_code = torch.tensor(tag_code)
        return data_code.reshape(-1), tag_code, len(tag_)

    def __len__(self):
        return self.all_data.__len__()


class Model(nn.Module):
    def __init__(self, num):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert_chinese")
        self.classifier = nn.Linear(768, num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, b_code, b_tag=None):
        """

        :param b_code: a batch of the encoded text
        :param b_tag: corresponding tags
        :return: loss or the result of prediction
        """
        bert_out = self.bert(b_code)[0]
        pre_ = self.classifier(bert_out)
        if b_tag is not None:
            loss_ = self.loss(pre_.reshape(-1, pre_.shape[-1]), b_tag.reshape(-1))
            return loss_
        else:
            return torch.argmax(pre_, dim=-1)


if __name__ == "__main__":
    train_text, train_tag = read_data("data/train.txt", "data/train_TAG.txt")
    dev_text, dev_tag = read_data("data/dev.txt", "data/dev_TAG.txt")
    test_text, test_tag = read_data("data/test.txt", "data/test_TAG.txt")

    tag_index_di, tag_index_li = get_tag_index(train_tag)
    # tag_index_di_dev, tag_index_li_dev = get_tag_index(dev_tag)
    tokenizer = BertTokenizer.from_pretrained("bert_chinese")

    batch_size = 10
    epoch = 2
    max_len = 100
    lr = 0.00001
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = BertDataset(train_text, train_tag, tag_index_di, max_len, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    dev_dataset = BertDataset(dev_text, dev_tag, tag_index_di, max_len, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = BertDataset(test_text, test_tag, tag_index_di, max_len, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Model(len(tag_index_di)).to(device)
    opt = AdamW(model.parameters(), lr)
    for e in range(epoch):
        model.train()
        for b_data_code, b_tag_code, b_len in train_dataloader:
            b_data_code = b_data_code.to(device)
            b_tag_code = b_tag_code.to(device)
            loss = model.forward(b_data_code, b_tag_code)
            loss.backward()
            opt.step()
            opt.zero_grad()
            print("loss:", loss)

    model.eval()
    all_pre = []
    all_tag = []
    for b_data_code1, b_tag_code1, b_len in dev_dataloader:
        b_data_code1 = b_data_code1.to(device)
        b_tag_code1 = b_tag_code1.to(device)
        pre = model.forward(b_data_code1)

        pre = pre.numpy().tolist()
        # pre = [[tag_index_li[j]for j in i]for i in pre]

        tag = b_tag_code1.numpy().tolist()
        for p, t, l in zip(pre, tag, b_len):
            p = p[1:1 + l]
            t = t[1:1 + l]
            p = [tag_index_li[i] for i in p]
            t = [tag_index_li[i] for i in t]
            all_pre.append(p)
            all_tag.append(t)
            # b_tag_code = [[tag_index_li[j]for j in i]for i in b_tag_code]
    output_f = open("output.txt", "a", encoding="utf-8")
    output_f.truncate(0)
    for item in all_pre:
        string_t = str(item).replace("'", "").replace("[", "").replace("]", "")
        string_t = string_t.replace(",", "")
        ret = ""
        for i in range(len(string_t)):
            ret += string_t[i]
        output_f.write(ret + "\n")
    score = f1_score(all_tag, all_pre)
    print("score:", score)
