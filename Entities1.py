import functools
import os
import re
import gc
import json
import jieba
import codecs
import keras.layers
from langconv import *
import numpy as np
import pandas as pd
from heapq import nsmallest
from sklearn.model_selection import KFold
from gensim.models import FastText, Word2Vec
import heapq
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.initializers import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from bert_serving.client import BertClient
from keras import backend as K
import time
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 设置GPU动态分配内存
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


def get_data(path):
    f = codecs.open(path, 'r', 'utf-8')
    res = []
    for x in f.readlines():
        res.append(json.loads(x.strip()))
    res = pd.DataFrame(res)
    return res

def f1(data):
    dict = {}
    for i in data:
        dict[i['entity']] = i['emotion']
    return dict

def data_clear(data):
    f = re.compile('<[^>]+>', re.S)
    data['text'] = data['text'].apply(lambda x : f.sub('', x))   #去掉HTML标签
    data['text'] = data['text'].apply(lambda x : x.lower())      #大写转小写
    data['text'] = data['text'].apply(lambda x : Converter('zh-hans').convert(x))  #繁体转简体
    data['text'] = data['text'].apply(lambda x : [w for w in jieba.cut(x) if len(w) != 1])
    return data

def get_embeding(model, size):
    w = np.zeros((len(index_word) + 1, size))
    for word, idx in word_index.items():
        if word in model:
            w[idx] = model[word]
    return w

def get_label(data, sequence, index_word):
    res = []
    for i, seq in enumerate(sequence):
        tmp = []
        dict = data.loc[i, 'coreEntityEmotions']
        for word in seq:
            if word == 0:
                tmp.append(0)
            elif index_word[word] not in dict.keys():
                tmp.append(0)
            else:
                tmp.append(1)
        res.append(tmp)
    return np.array(res)

def get_model(embedding_matrix):
    K.clear_session()
    sequence = Input(shape=(max_len,), dtype='float32')
    embedding = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],input_length=max_len, trainable=False, mask_zero=False)(sequence)
    spat = SpatialDropout1D(dropout)(embedding)
    blstm1 = Bidirectional(LSTM(128, kernel_initializer=Orthogonal(seed=2022), return_sequences=True),merge_mode='sum')(spat)
    dropout1 = Dropout(dropout, seed=2022)(blstm1)
    blstm2 = Bidirectional(LSTM(128, kernel_initializer=Orthogonal(seed=2022), return_sequences=True),merge_mode='sum')(dropout1)
    dropout2 = Dropout(dropout, seed=2022)(blstm2)

    res1 = add([blstm1, dropout2])

    blstm3 = Bidirectional(LSTM(128, kernel_initializer=Orthogonal(seed=2022), return_sequences=True), merge_mode='sum')(dropout2)
    dropout3 = Dropout(dropout, seed=2022)(blstm3)

    res2 = add([blstm2, dropout3])

    res = concatenate([res1, res2], axis = 2)

    output = TimeDistributed(Dense(1, kernel_initializer=Orthogonal(seed=2022), activation='sigmoid'))(res)
    output = Reshape((max_len,))(output)
    model = Model(sequence, output)
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])

    return model

def calc_result(pred, data, index_word):
    entities_list = []
    probs = pd.DataFrame(pred)

    for i in range(len(probs)):
        entity_list = []
        prob = probs.loc[i]
        text = data[i]
        mn = 0
        idx = -1
        s = set()
        for j in range(len(prob)):
            if text[j] == 0:
                continue
            if prob[j] >= prob_entities and index_word[text[j]] not in s:
                entity_list.append((prob[j], j))
                s.add(index_word[text[j]])
            elif prob[j] > mn:
                mn = prob[j]
                idx = j

        if len(entity_list) == 0:
            entity_list.append((mn, idx))
        sorted(entity_list, key=lambda x : x[0])
        while len(entity_list) > 3:
            entity_list.pop()
        real_entity = []
        for (p, j) in entity_list:
            if text[j] == 0:
                continue
            real_entity.append(index_word[text[j]])
        entities_list.append(','.join(real_entity))

    return entities_list

def get_score(real, pred_data):
    entity_score = 0
    for i in real.index:
        pred = pred_data.iloc[i]
        true = real.iloc[i]

        pred_entity = pred['entity']
        true_entity = true['entity']
        n = len(list(set(pred_entity) & set(true_entity)))  # 单个样本正确预测的实体个数
        if n != 0:
            precision = n / len(pred_entity)  # 单个样本的实体准确率
            recall = n / len(true_entity)  # 单个样本的实体召回率
            f1 = 2 * precision * recall / (precision + recall)  # 单个样本的实体F1值
            entity_score += f1

    entity_score /= len(real)  # 实体分数

    print('entity score =', entity_score)

if __name__ == '__main__':
    st = time.time()
    bert_size = 768
    word2vec_size=100
    fasttext_size=100
    max_len=100
    nflod = 5
    lr = 0.0003
    batch_size=64
    epochs=4
    dropout = 0.3
    prob_entities = 0.75 #划分为实体的阈值
    jieba.load_userdict('./dict/entities.txt')
    print('读取数据')
    train_data = get_data('./data/train.txt')
    test_data = get_data('./data/test.txt')
    print('数据预处理')
    train_data['text'] = train_data['title'] + train_data['content']
    test_data['text'] = test_data['title'] + test_data['content']
    train_data.drop(['title', 'content'], axis=1,inplace=True)
    test_data.drop(['title', 'content'], axis=1,inplace=True)

    train_data['coreEntityEmotions'] = train_data['coreEntityEmotions'].apply(f1)
    test_data['coreEntityEmotions'] = ''

    train_data = data_clear(train_data)
    test_data = data_clear(test_data)
    text = pd.concat([train_data['text'], test_data['text']], ignore_index=True)

    print('文本序列化')
    token = Tokenizer(num_words=None)
    token.fit_on_texts(text)
    index_word = token.index_word
    word_index = token.word_index
    sequences = token.texts_to_sequences(text)

    print('bert词向量')
    model = BertClient()
    words = list(index_word.values())
    w1 = model.encode([w.replace(' ', '') for w in words])
    w1 = np.concatenate((np.zeros((1,bert_size)), w1), axis=0)
    del words
    gc.collect()

    print('word2vec词向量')
    model = Word2Vec(text, seed=2022,size=word2vec_size, min_count=1,workers=20,iter=20,window=8)
    model.save('./vector/word2vec/word2vec.model')
    w2 = get_embeding(model,word2vec_size)

    print('fasttext词向量')
    model = FastText(text, seed=2022,size=fasttext_size, min_count=1,workers=20,iter=20,window=8)
    model.save('./vector/fasttext/fasttext.model')
    w3 = get_embeding(model,fasttext_size)
    del model
    gc.collect()

    print('词向量拼接')
    embeding = np.concatenate((w1, w2, w3), axis=1)
    del w1, w2, w3
    gc.collect()

    print('文本长度补齐')
    data = pad_sequences(sequences, maxlen=max_len,padding='post', truncating='post')
    X_train = data[:len(train_data)]
    X_test = data[len(train_data):]
    del data
    gc.collect()

    print('数据打标')
    Y_train = get_label(train_data,X_train, index_word)

    print('模型训练及预测')
    pred_train = np.zeros((len(X_train), max_len))
    pred_test = np.zeros((len(X_test), max_len))

    kf = KFold(n_splits=nflod, shuffle=True, random_state=2022).split(X_train)
    for i, (train_idx, valid_idx) in enumerate(kf):
        print('第%d折训练'%(i + 1))
        x_train, y_train = X_train[train_idx], Y_train[train_idx]
        x_valid, y_valid = X_train[valid_idx], Y_train[valid_idx]

        model = get_model(embeding)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))
        pred_train[valid_idx] = model.predict(x_valid)
        pred_test += model.predict(X_test) / nflod

    print('预测结果处理')
    entity_list = calc_result(pred_train, X_train, index_word)
    pred = pd.DataFrame({'newsId': train_data['newsId'], 'entity': entity_list})

    print('打分')
    real_data = train_data[['newsId', 'coreEntityEmotions']]
    real_data['entity'] = train_data['coreEntityEmotions'].apply(lambda x: list(x.keys()))
    real_data = real_data[['newsId', 'entity']]
    pred['entity'] = pred['entity'].apply(lambda x: x.split(','))
    get_score(real_data, pred)

    pred.to_pickle('pred_entity.pkl')
    ed = time.time()
    print('共用{}小时'.format((ed - st) / 3600))

