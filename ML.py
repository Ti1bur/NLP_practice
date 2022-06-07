import re
import pandas as pd
import numpy as np
import jieba
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import warnings
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
warnings.filterwarnings('ignore')

def filter(text):
    text = re.sub(r"[A-Za-z0-9\!\=\？\%\[\]\,\（\）\>\<:&lt;\/#\. -----\_]", "", text)
    text = text.replace('图片', '')
    text = text.replace('\xa0', '')

    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, ' ', text)
    text = text.strip()
    return text

def get_stopwords():
    with open('stop_words.txt', 'r', encoding='utf-8') as f:
        stop_words = [word.strip('\n') for word in f.readlines()]
    return stop_words

def cut_text(s, stop_words):
    tokens = list(jieba.cut(s))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def get_TFIDF():
    # 训练tf-idf模型
    tfidf = TfidfVectorizer(min_df=5, ngram_range=[1, 3], token_pattern=r'\b\w+\b', max_features=100000)
    tfidf.fit(train_comments_cut + test_comments_cut)

    X_train_TFIDF = tfidf.transform(train_comments_cut).toarray()
    Y_train_TFIDF = np.array(train['label'].tolist())

    X_test_TFIDF = tfidf.transform(test_comments_cut).toarray()
    return X_train_TFIDF, Y_train_TFIDF, X_test_TFIDF

def TFIDF_SVM():
    kf = KFold(n_splits=5, shuffle=True, random_state=2022).split(X_train_TFIDF)
    pred = []
    for i, (train_idx, val_idx) in enumerate(kf):
        print('第%d折开始训练'%(i + 1))
        x_train, y_train = X_train_TFIDF[train_idx], Y_train_TFIDF[train_idx]
        model = SVC(C=0.9, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='linear',
                    max_iter=-1, random_state=None, shrinking=True,tol=0.001, verbose=False)
        model.fit(x_train, y_train)
        preds = model.predict(X_test_TFIDF)

        if len(pred) == 0:
            pred = [x / 5 for x in preds]
        else:
            for j in range(len(pred)):
                pred[j] = pred[j] + preds[j] / 5

    res = [1 if x >= 0.5 else 0 for x in pred]
    return res

def TFIDF_XGBOOST():
    params = {'booster': 'gbtree', 'objective': 'binary:logistic', 'eval_metric': 'auc', 'gamma': 0.1, 'min_child_weight': 1.1, 'max_depth': 5, 'lambda': 10, 'subsample': 0.7, 'colsample_bytree': 0.7, 'colsample_bylevel': 0.7,
               'eta': 0.01, 'tree_method': 'gpu_hist', 'seed': 0, 'nthread': 12, 'predictor': 'gpu_predictor'}
    kf = KFold(n_splits=5, shuffle=True, random_state=2022).split(X_train_TFIDF)
    pred = []
    for i, (train_idx, val_idx) in enumerate(kf):
        print('第%d折开始训练' % (i + 1))
        x_train, y_train = X_train_TFIDF[train_idx], Y_train_TFIDF[train_idx]
        train_data = xgb.DMatrix(x_train, label=y_train)
        test_data = xgb.DMatrix(X_test_TFIDF)
        w = [(train_data, 'train')]

        model = xgb.train(params, train_data, num_boost_round=5000, evals=w, early_stopping_rounds=50)
        predict = pd.DataFrame(model.predict(test_data, validate_features=False), columns=['prob'])
        preds = predict['prob'].tolist()

        if len(pred) == 0:
            pred = [x / 5 for x in preds]
        else:
            for j in range(len(pred)):
                pred[j] = pred[j] + preds[j] / 5

    res = [1 if x >= 0.5 else 0 for x in preds]  ##############阈值记得多试试看,换几个试试效果
    return res

def get_BOW():
    vectorizer = CountVectorizer(min_df=10, ngram_range=[1, 3], token_pattern=r'\b\w+\b')
    vectorizer.fit(train_comments_cut + test_comments_cut)

    X_train_BOW = vectorizer.transform(train_comments_cut).toarray()
    Y_train_BOW = np.array(train['label'].tolist())
    X_test_BOW = vectorizer.transform(test_comments_cut).toarray()

    return X_train_BOW, Y_train_BOW, X_test_BOW

def BOW_SVM():
    kf = KFold(n_splits=5, shuffle=True, random_state=2022).split(X_train_BOW)
    pred = []
    for i, (train_idx, val_idx) in enumerate(kf):
        print('第%d折开始训练' % (i + 1))
        x_train, y_train = X_train_BOW[train_idx], Y_train_BOW[train_idx]
        model = SVC(C=0.9, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='linear',
                    max_iter=-1, random_state=None, shrinking=True, tol=0.001, verbose=False)
        model.fit(x_train, y_train)
        preds = model.predict(X_test_BOW)

        if len(pred) == 0:
            pred = [x / 5 for x in preds]
        else:
            for j in range(len(pred)):
                pred[j] = pred[j] + preds[j] / 5

    res = [1 if x >= 0.5 else 0 for x in pred]
    return res

def save(pred, path):
    sub = test.copy()
    sub['label'] = pred
    sub[['id', 'label']].to_csv('./o2o商铺食品安全/陈浩如-ML/'+path, index = None)

if __name__ == '__main__':
    print('读取数据')
    train = pd.read_csv('./o2o商铺食品安全/train.csv', sep = '\t')
    test = pd.read_csv('./o2o商铺食品安全/test_new.csv', sep = ',')
    print('数据预处理')
    train['comment'] = train['comment'].apply(lambda x : filter(x))
    test['comment'] = test['comment'].apply(lambda x : filter(x))
    stop_words = get_stopwords()
    print('jieba分词中')
    train_comments_cut = [cut_text(sent, stop_words) for sent in train.comment.values]
    test_comments_cut = [cut_text(sent, stop_words) for sent in test.comment.values]

    print('训练TFIDF模型')
    X_train_TFIDF, Y_train_TFIDF, X_test_TFIDF = get_TFIDF()

    print('训练BOW模型')
    X_train_BOW, Y_train_BOW, X_test_BOW = get_BOW()

    print('训练TFIDF-SVM模型')
    pred1 = TFIDF_SVM()
    save(pred1, 'result_TFIDF_SVM.csv')
    print('训练TFIDF-XGBOOST模型')
    pred2 = TFIDF_XGBOOST()
    save(pred2, 'result_TFIDF_XGBOOST.csv')
    print('训练BOW-SVM模型')
    pred3 = BOW_SVM()
    save(pred3, 'result_BOW_SVM.csv')
    result = []
    for i in range(len(pred1)):
        result.append(pred1[i] + pred2[i] + pred3[i])

    y_test = [1 if x >= 2 else 0 for x in result]
    save(y_test, 'result_mix_model.csv')