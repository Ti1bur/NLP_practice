import re
from random import sample
import jieba
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from gensim.models import KeyedVectors, Word2Vec
import codecs
import os

class LSTM_CNN_ResNet(nn.Module):
    def __init__(self, embed_matrix, vocab_size, kernel_sizes, num_filters, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)  #embeding_dim = 128
        self.embedding.weight.data.copy_(embed_matrix)
        self.embedding.weight.requires_grad = True
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        #embeding = 128, hidden_dim = 256
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=n_layers, bidirectional = bidirectional,dropout=dropout) #input(128) output(256)
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=hidden_dim * 2, out_channels=num_filters, kernel_size=k, stride=1) for k in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_dim)   #输入512, output=1
        self.dropout = nn.Dropout(dropout)

    def forward(self, text): #input x(32, 66)
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        #hidden (2, 32, 256)  output(32,71,512) cell(2,32,256)
        output = output.transpose(1, 2)  #(32, 512, 99)
        #残差
        res = output
        res = F.avg_pool1d(res, res.size(-1)).squeeze(dim = -1)
        output = [F.relu(conv(output)) for conv in self.conv]   #[(32, 80, 68), (32, 80, 67), (32, 80,66)]
        output = [F.max_pool1d(c, c.size(-1)).squeeze(dim = -1) for c in output]  #[(32, 80), (32,80), (32,80)]
        output = torch.cat(output, dim = 1)       #(32,240)
        output = res + output
        output = self.fc(self.dropout(output))   #(32,1)
        return torch.sigmoid(output).squeeze()

def model_word2vec(vc = 128, window=2, min_conunt=1, workers=5, sg=0, epochs=20):
    train = pd.read_csv('../o2o商铺食品安全/train.csv', sep='\t')
    test = pd.read_csv('../o2o商铺食品安全/test_new.csv')
    train = train[['comment', 'label']]

    # 将所有句子提取
    comment = train['comment'].tolist() + test['comment'].tolist()
    comments = pd.DataFrame(comment)
    comments.columns = ['comment']
    comments['comment'] = comments['comment'].map(lambda x: [w for w in list(jieba.cut(x)) if len(w) != 1])
    #训练
    model = Word2Vec(size = vc, window = window, min_count=min_conunt, workers = workers, sg = sg, iter = epochs)
    model.build_vocab(comments['comment'])
    model.train(comments['comment'], total_examples=model.corpus_count, epochs=model.epochs)
    model.save('./word2vector.model')
    model.wv.save_word2vec_format('embed.txt')
    print('Word2Vec训练完毕')

def set_valid():
    train = pd.read_csv('../o2o商铺食品安全/train.csv', sep = '\t')
    train = train[['label', 'comment']]

    shuffled = train.sample(frac=1)
    result = np.array_split(shuffled, 5)
    print('第一份训练集')
    new_train = pd.concat([result[0], result[1], result[2], result[3]], axis = 0)
    new_valid = result[4]
    new_train.to_csv('../o2o商铺食品安全/new_train1.csv', sep = '\t', index=None)
    new_valid.to_csv('../o2o商铺食品安全/new_valid1.csv', sep = '\t', index=None)

    print('第二份训练集')
    new_train = pd.concat([result[0], result[1], result[2], result[4]], axis=0)
    new_valid = result[3]

    new_train.to_csv('../o2o商铺食品安全/new_train2.csv', sep='\t', index=None)
    new_valid.to_csv('../o2o商铺食品安全/new_valid2.csv', sep='\t', index=None)

    print('第三份训练集')
    new_train = pd.concat([result[0], result[1], result[4], result[3]], axis=0)
    new_valid = result[2]

    new_train.to_csv('../o2o商铺食品安全/new_train3.csv', sep='\t', index=None)
    new_valid.to_csv('../o2o商铺食品安全/new_valid3.csv', sep='\t', index=None)

    print('第四份训练集')
    new_train = pd.concat([result[0], result[4], result[2], result[3]], axis=0)
    new_valid = result[1]

    new_train.to_csv('../o2o商铺食品安全/new_train4.csv', sep='\t', index=None)
    new_valid.to_csv('../o2o商铺食品安全/new_valid4.csv', sep='\t', index=None)

    print('第五份训练集')
    new_train = pd.concat([result[4], result[1], result[2], result[3]], axis=0)
    new_valid = result[0]

    new_train.to_csv('../o2o商铺食品安全/new_train5.csv', sep='\t', index=None)
    new_valid.to_csv('../o2o商铺食品安全/new_valid5.csv', sep='\t', index=None)

    print('成功划分验证集')

def cutcut(text):
    regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9，。？：！；“”]')
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]

def get_data(path, sep, size=64):
    iterator = pd.read_csv(path, encoding='utf-8', delimiter=sep, iterator=True)
    while True:
        try:
            data = iterator.get_chunk(size)
            label = data.iloc[:, 0]
            comment = data.iloc[:, 1]
            comment = comment.apply(cutcut)
            yield comment, label
        except StopIteration:
            break

def set_of_word():
    train_iter = get_data('../o2o商铺食品安全/train.csv', sep='\t', size=32)
    test_iter = get_data('../o2o商铺食品安全/test_new.csv', sep=',', size=32)

    res = set()
    for comment, label in train_iter:
        for x in comment:
            for y in x:
                res.add(y)

    for comment, label in test_iter:
        for x in comment:
            for y in x:
                res.add(y)
    return res

def get_word_dict():
    word = set_of_word()
    model = KeyedVectors.load_word2vec_format('embed.txt')
    id_word = model.index2word
    without = list(word - set(id_word))
    without.insert(0, '<pad>')

    id_word = without + id_word
    word_dict = dict(zip(id_word, [w for w in range(len(id_word))]))
    return model, word_dict, without

def std(comment):
    res = []
    max_len = 0
    for x in comment:
        max_len = max(max_len, len(x))
    for x in comment:
        a = []
        for w in x:
            try:
                id = word_dict[w]
            except KeyError:
                id = word_dict['<unk>']
            a.append(id)
        while len(a) < max_len:
            a.append(word_dict['<pad>'])
        res.append(a)
    return res

def calc(train_x, train_y, opt, criterion, model):
    model.train()
    opt.zero_grad()
    y_pred = model(train_x)
    loss = criterion(y_pred, train_y)
    loss.backward()
    opt.step()
    return loss

def train(fold, epoch=20):
    torch.cuda.empty_cache()

    model = LSTM_CNN_ResNet(embed_matrix=embed_matrix, vocab_size = vs, embedding_dim = embed_dim, hidden_dim=120,
            output_dim=1, n_layers=2, bidirectional=True, dropout=0.3, pad_idx = pad_idx, kernel_sizes=[5, 6, 7], num_filters=80)
    model.to('cuda')  # 开启GPU加速模式
    train_path = '../o2o商铺食品安全/new_train'+ascii(fold)+'.csv'
    valid_path = '../o2o商铺食品安全/new_valid'+ascii(fold)+'.csv'
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = nn.BCELoss()
    for i in range(epoch):
        #训练
        iterator = get_data(train_path, sep='\t', size = 32)
        for train_x, train_y in iterator:
            train_x = torch.from_numpy(np.array(std(train_x), dtype=np.int64)).to('cuda')
            train_y = np.array(train_y.tolist(), dtype=np.float32)
            train_y = torch.from_numpy(train_y).to('cuda')
            loss = calc(train_x, train_y, opt, criterion, model)

        y_pred = []
        ys = []
        iterator = get_data(valid_path, sep = '\t', size = 32)
        for valid_x, valid_y in iterator:
            valid_x = torch.from_numpy(np.array(std(valid_x), dtype=np.int64)).to('cuda')
            model.eval()
            with torch.no_grad():
                y_p = model(valid_x)
            y_p = torch.round(y_p).cpu().numpy().tolist()
            y_pred += y_p
            ys += list(valid_y)
        f1 = f1_score(ys, y_pred)
        print('epoch: {} F1 : {:.2f}'.format(i + 1, f1))


    # 预测
    output = []
    model.eval()
    iterator = get_data('../o2o商铺食品安全/test_new.csv', sep=',', size=32)
    with torch.no_grad():
        for comment, id in iterator:
            test_x = torch.from_numpy(np.array(std(comment), dtype=np.int64)).to('cuda')
            pred = model(test_x)
            pred = pred.cpu().numpy().tolist()
            output += pred
    return output


if __name__ == '__main__':
    #训练词向量
    # if not os.path.exists('./word2vector.model'):
    model_word2vec(vc = 128, window = 2, min_conunt=1, workers=5, sg = 0, epochs=20)

    #划分验证集
    # set_valid()

    #获得embed词典
    model, word_dict, without = get_word_dict()

    #获得需要的参数
    embed_matrix = model.vectors
    embed_dim = embed_matrix.shape[1]
    without_embed = np.zeros((len(without), embed_dim))
    embed_matrix = torch.from_numpy(np.vstack((without_embed, embed_matrix)))
    vs = embed_matrix.shape[0]
    pad_idx = word_dict['<pad>']

    pred = []
    #五折交叉训练
    for i in range(1, 6):
        print('第%d折训练'%i)
        output = train(i, epoch = 50)
        if len(pred) == 0:
            pred = [x / 5 for x in output]
        else:
            for j in range(len(pred)):
                pred[j] = pred[j] + output[j] / 5

    label = []
    for i in pred:
        if i >= 0.5:
            label.append(1)
        else:
            label.append(0)

    result = pd.read_csv('../o2o商铺食品安全/sample.csv')
    result['label'] = label
    result.to_csv('../o2o商铺食品安全/陈浩如-DL_lay2'
                  '_lstm_dropout.csv', index = None)