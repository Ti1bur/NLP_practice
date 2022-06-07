import os
import re
import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold

train_df = pd.read_csv("./datasets/train.csv", sep='\t')
test_df = pd.read_csv("./datasets/test_new.csv")

def clean_space(text):
    """"
    处理多余的空格
    """
    match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
    should_replace_list = match_regex.findall(text)
    order_replace_list = sorted(should_replace_list,key=lambda i:len(i), reverse=True)
    for i in order_replace_list:
        if i == u' ':
            continue
        new_i = i.strip()
        text = text.replace(i, new_i)
    return text

def clean(text):
    """
    清除无用的信息
    :param text:
    :return:
    """
    if type(text) != str:
        return text
    text = clean_space(text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\{IMG:[0-9]+\}', '', text)
    text = re.sub(r'\?{2,}', '', text)
    text = re.sub(r'[0-9a-zA-Z]{100,}', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http(s?):[/a-zA-Z0-9.=&?_#]+', '', text)
    text = re.sub(r'&ldquo;', '', text)
    text = re.sub(r'&rdquo;', '', text)
    text = re.sub(r'—{5,}', '', text)

    text = re.sub(r'？{2,}', '', text)
    text = re.sub(r'●', '', text)
    text = re.sub(r'【图】', '', text)
    text = re.sub(r'[0-9]+[-|.|/|年][0-9]{2}[-|.|/|月][0-9]{2}日?', '', text)
    text = re.sub(r'&nbsp;', '', text)
    text = re.sub(r'[0-9]{15,}', '', text)
    text = re.sub(r'&quot;', '', text)
    return text

train_df['id'] = list(range(len(train_df)))
train_df['comment'] = train_df['comment'].apply(clean_space)
train_df['comment'] = train_df['comment'].apply(clean)
test_df['comment'] = test_df['comment'].apply(clean_space)
test_df['comment'] = test_df['comment'].apply(clean)

train_df[['id', 'comment', 'label']].to_csv('./datasets/process_train.csv', index=None)
test_df[['id', 'comment']].to_csv('./datasets/process_test.csv', index=None)