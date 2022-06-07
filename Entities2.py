import functools
import os
import re
import gc
import json
import jieba
import codecs
import keras.layers
from langconv import *
import pandas as pd
from sklearn.model_selection import KFold
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

def get_entity_comment(data):
    res = []
    for i in range(len(data)):
        data1 = data.loc[i]
        dict = {}
        for j in data1['coreEntityEmotions'].keys():
            tmp = []
            for k in data1['text']:
                if j in k:
                    tmp.append(k)
            if len(tmp) >= 3:
                xx = []
                xx.append(tmp[0])
                xx.append(tmp[-1])
                xx.append(tmp[len(tmp) // 2 + 1])
                dict[j] = xx
            else:
                tmp = []
                for k in range(len(data1['text'])):
                    if j in data1['text'][k]:
                        if k >= 1:
                            tmp.append(data1['text'][k - 1])
                            if len(tmp) == 3:
                                break
                        tmp.append(data1['text'][k])
                        if len(tmp) == 3:
                            break
                        if k < len(data1['text']) - 1:
                            tmp.append(data1['text'][k + 1])
                            if len(tmp) == 3:
                                break
                while len(tmp) < 3:
                    tmp.append([])
                dict[j] = tmp
            xxx = []
            for k in dict[j]:
                xxxx = []
                for z in k:
                    if z in data1['coreEntityEmotions'].keys() and z != j:
                        xxxx.append('<unk>')
                    else:
                        xxxx.append(z)

                xxx.append(xxxx)
            dict[j] = xxx
            t = ''
            for x in dict[j]:
                t = t + ''.join(x) + '。'
            dict[j] = t
        res.append(dict)

    return res

def merge(data):
    res = []

    for i in range(len(data)):
        t = data.loc[i, 'title']
        t = t + '。' + data.loc[i, 'content']
        res.append(t)

    return res

def f1(t):
    t = t.replace('!', '。')
    t = t.replace('?', '。')
    t = t.replace('！', '。')
    t = t.replace('？', '。')
    t = t.replace('\n', '')
    res = []
    for j in t.split('。'):
        if len(j) != 0:
            tmp = []
            for k in jieba.cut(j):
                if len(k) != 0:
                    tmp.append(k)
            if len(tmp) != 0:
                res.append(tmp)
    return res

def f2(data):
    dict = {}
    for i in data:
        dict[i['entity']] = i['emotion']
    return dict

def f3(data):
    res = {}
    for i in data:
        res[i] = ''
    return res

def f4(t):
    t = t.replace(',', '')
    t = t.replace('“', '')
    t = t.replace('”', '')
    t = t.replace('‘', '')
    t = t.replace('’', '')
    t = t.replace('@', '')
    t = t.replace('#', '')
    t = t.replace('$', '')
    t = t.replace('.', '')
    t = t.replace('、', '')
    t = t.replace(':', '')
    t = t.replace('：', '')
    t = t.replace('。', '')
    t = t.replace('，', '')
    t = t.replace('|', '')
    t = t.replace('（', '')
    t = t.replace('）', '')
    if t == '':
        t = '<unk>'
    return t

def get_train_label(data):
    train = []
    entity = []
    label = []
    for i in range(len(data)):
        data1 = data.loc[i]
        for j in data1['coreEntityEmotions'].keys():
            entity.append(j)
            train.append(data1['entity_comment'][j])
            if data1['coreEntityEmotions'][j] == 'POS':
                label.append([[1, 0, 0]])
            elif data1['coreEntityEmotions'][j] == 'NORM':
                label.append([[0, 1, 0]])
            else:
                label.append([[0, 0, 1]])
    return train, label, entity

def get_test_data(data):
    comment = []
    entity = []
    for i in range(len(data)):
        data1 = data.loc[i]
        for j in data1['entity_comment'].keys():
            entity.append(j)
            comment.append(data1['entity_comment'][j])
    return entity, comment

def get_tokenizer(data):
    s = set()
    i_c = {}
    c_i = {}
    for x in data:
        s.add(x)

    for i, content in enumerate(s):
        i_c[i + 1] = content
        c_i[content] = i + 1
    seq = []
    for c in data:
        seq.append([c_i[c]])
    return i_c, c_i, seq

def get_model(embedding_matrix):
    K.clear_session()
    sequence = Input(shape=(max_len,), dtype='float32')
    embedding = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],input_length=max_len, trainable=False, mask_zero=True)(sequence)
    mask = Masking(mask_value=0.)(embedding)
    spat = SpatialDropout1D(0.2)(mask)
    blstm1 = Bidirectional(LSTM(256, kernel_initializer=Orthogonal(seed=2022), return_sequences=True),merge_mode='sum')(spat)
    dropout1 = Dropout(0.2, seed=2022)(blstm1)
    blstm2 = Bidirectional(LSTM(128, kernel_initializer=Orthogonal(seed=2022), return_sequences=True),merge_mode='sum')(dropout1)
    dropout2 = Dropout(0.2, seed=2022)(blstm2)
    blstm3 = Bidirectional(LSTM(64, kernel_initializer=Orthogonal(seed=2022), return_sequences=True), merge_mode='sum')(dropout2)
    dropout3 = Dropout(0.2, seed=2022)(blstm3)
    blstm4 = Bidirectional(LSTM(32, kernel_initializer=Orthogonal(seed=2022), return_sequences=True), merge_mode='sum')(dropout3)
    dropout4 = Dropout(0.2, seed=2022)(blstm4)
    output = Dense(3, kernel_initializer=Orthogonal(seed=2022), activation='softmax')(dropout4)
    model = Model(sequence, output)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
    return model

def get_label(data):
    res = []
    for i in range(len(data)):
        data1 = data.loc[i]
        res.append(data1['label'])
    return np.array(res)

def get_score(real, pred_data):
    entity_score = 0
    emotions_score = 0
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

        pred_emotion = [entity + '_' + emotions for entity, emotions in zip(pred['entity'], pred['emotion'])]
        true_emotion = [entity + '_' + emotions for entity, emotions in zip(true['entity'], true['emotion'])]
        n = len(list(set(pred_emotion) & set(true_emotion)))  # 单个样本正确预测的实体情感个数
        if n != 0:
            precision = n / len(pred_emotion)  # 单个样本的实体情感准确率
            recall = n / len(true_emotion)  # 单个样本的实体情感召回率
            f1 = 2 * precision * recall / (precision + recall)  # 单个样本的实体情感F1值
            emotions_score += f1
    entity_score /= len(real)  # 实体分数
    emotions_score /= len(real)  # 实体情感分数
    score = (entity_score + emotions_score) / 2  # 总分数

    print('entity score =', entity_score)
    print('emotions score =', emotions_score)
    print('score =', score)

if __name__ == '__main__':
    st = time.time()
    bert_size = 768
    word2vec_size = 100
    fasttext_size = 100
    max_len = 1
    nflod = 2
    lr = 0.0003
    batch_size = 64
    epochs = 4
    dropout = 0.3


    print('读取数据')
    jieba.load_userdict('./dict/entities.txt')
    train_data = get_data('./data/train.txt')
    test_data = pd.read_pickle('./pred_entity.pkl')

    print('数据预处理')
    train_data['text'] = merge(train_data)
    train_data.drop(['title', 'content'], axis=1, inplace=True)
    f = re.compile('<[^>]+>', re.S)
    train_data['text'] = train_data['text'].apply(lambda x: f.sub('', x))  # 去掉HTML标签
    train_data['text'] = train_data['text'].apply(lambda x: x.lower())  # 大写转小写
    train_data['text'] = train_data['text'].apply(lambda x: Converter('zh-hans').convert(x))  # 繁体转简体
    train_data['text'] = train_data['text'].apply(f1)
    #情感标签处理
    train_data['coreEntityEmotions'] = train_data['coreEntityEmotions'].apply(f2)
    test_data['text'] = train_data['text']
    test_data['coreEntityEmotions'] = test_data['entity'].apply(f3)


    train_data['entity_comment'] = get_entity_comment(train_data)
    test_data['entity_comment'] = get_entity_comment(test_data)

    emotion_num = {'POS' : 0, 'NORM' : 1, 'NEG' : 2}
    num_emotion = {0 : 'POS', 1 : 'NORM', 2 : 'NEG'}

    X_train, Y_train, entity = get_train_label(train_data)
    test_entity, X_test = get_test_data(test_data)
    train = pd.DataFrame({'text' : X_train,'entity': entity, 'label' : Y_train})
    train = train[train['text'] != '。。。'].reset_index()
    test = pd.DataFrame({'text' : X_test, 'entity' : test_entity})
    train['text'] = train['text'].apply(f4)
    test['text'] = test['text'].apply(f4)

    text = pd.concat([train['text'], test['text']], ignore_index=True)

    print('句子序列化')
    index_content, content_index, sequences = get_tokenizer(text)

    print('打标')
    Y_train = get_label(train)

    print('训练句向量')
    model = BertClient()
    c = list(index_content.values())
    embedding = model.encode(c)

    print('文本长度补齐')
    data = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    X_train = data[:len(train)]
    X_test = data[len(train):]
    del data
    gc.collect()

    print('模型训练及预测')
    pred_test = np.zeros((len(X_test), 3))

    kf = KFold(n_splits=nflod, shuffle=True, random_state=2022).split(X_train)
    for i, (train_idx, valid_idx) in enumerate(kf):
        print('第%d折训练' % (i + 1))
        x_train= X_train[train_idx]
        y_train = Y_train[train_idx]
        x_valid, y_valid = X_train[valid_idx], Y_train[valid_idx]

        model = get_model(embedding)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))
        x = model.predict(X_test)
        x = np.squeeze(x, axis = 1)
        pred_test += x / nflod


    print('预测结果处理')
    preds = []
    for i in range(len(pred_test)):
        pred = pred_test[i]
        if pred[0] == max(pred[0], pred[1], pred[2]):
            preds.append('POS')
        elif pred[1] == max(pred[0], pred[1], pred[2]):
            preds.append('NORM')
        else:
            preds.append('NEG')


    print('计算验证集得分')
    real_data = train_data[['newsId', 'coreEntityEmotions']]
    real_data['entity'] = train_data['coreEntityEmotions'].apply(lambda x: list(x.keys()))
    real_data['emotion'] = train_data['coreEntityEmotions'].apply(lambda x: list(x.values()))
    real_data = real_data[['newsId', 'entity', 'emotion']]
    id = 0
    res1 = []
    res2 = []
    for i in range(len(test_data)):
        data1 = test_data.loc[i]
        tmp1 = []
        tmp2 = []
        for j in range(len(data1['coreEntityEmotions'].keys())):
            tmp1.append(preds[id])
            tmp2.append(test_entity[id])
            id += 1
        res1.append(tmp1)
        res2.append(tmp2)
    test_data['entity'] = res2
    test_data['emotion'] = res1
    get_score(real_data, test_data)
    ed = time.time()
    used = ed - st
    print('共用{}小时'.format(used/3600))