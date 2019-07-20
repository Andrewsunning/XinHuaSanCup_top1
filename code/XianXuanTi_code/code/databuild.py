import os
pth = os.getcwd()

import sys
sys.path.append(pth)

from config import *
from functions import *
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from functools import reduce
import warnings
warnings.filterwarnings('ignore')

'''
1. reading & making, 读取数据+构造
2. preprocessing, 预处理
3. labeling, 打标签
4. featuring, 特征工程
'''

## reading & making
print('reading & making')

# 训练集合
df_train = pd.read_csv(path_data+file_train)

# 测试集合
df_test = pd.read_csv(path_data+file_test)

# 停用词
lst_stopwords = []
with open(path_data+file_stopwords, 'r') as f:
    for line in f:
        lst_stopwords.append(line.strip())

# 标签映射
set_accusations = set(reduce(lambda x,y: x+y, [x.split(';') for x in df_train.accusation.unique()]))
dic_labels = {}
for i, accusation in enumerate(set_accusations):
    dic_labels[accusation] = str(i)
with open(path_data+file_dic_labels, 'w') as f: # 备份
    json.dump(dic_labels, f)

## preprocessing
print('preprocessing')

# 去重
df_train.drop_duplicates(inplace=True)

## labeling
print('labeling')

# 多标签构建
df_train['label'] = df_train['accusation'].apply(get_label, args=(dic_labels, )) # 标签编码
iter_label = (set(x.split(',')) for x in df_train.label)
lst_labels = sorted(set.union(*iter_label))
df_train_y = pd.DataFrame(np.zeros((len(df_train), len(lst_labels))), columns=lst_labels)
for i, label in enumerate(df_train.label):
    df_train_y.loc[i, label.split(',')] = 1
df_train_y.to_csv(path_data+file_train_y, index=None) # 备份

## featuring
print('featuring')

# 训练集
df_train['fact_words'] = df_train['fact'].apply(get_words, args=(lst_stopwords, )) # 分词
df_train.to_csv(path_data+file_train_preprocessing, index=None) # 备份
tfidfer = TfidfVectorizer(max_features=num_features, max_df=0.8) # 初始化tfidf器
tfidfer.fit(df_train.fact_words) # 训练词向量
tfidf = tfidfer.transform(df_train.fact_words) # 构建词向量
np_train_X = tfidf.toarray()
np.save(path_data+file_train_X, np_train_X)

# 测试集
df_test['fact_words'] = df_test['fact'].apply(get_words, args=(lst_stopwords, )) # 分词
df_test.to_csv(path_data+'test_preprocessing.csv', index=None) # 备份
tfidf = tfidfer.transform(df_test.fact_words) # 构建词向量
np_test_X= tfidf.toarray()
np.save(path_data+file_test_X, np_test_X)