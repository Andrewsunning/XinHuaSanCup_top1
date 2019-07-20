#!/usr/bin/python
# coding=UTF-8

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
pd.set_option('display.max_rows',200)


# 读入个特征数据集
train_label = pd.read_csv('./process_data/add_data_2_train_label.csv')

df_0 = pd.read_csv('./process_data/第一轮特征.csv')

df_1 = pd.read_csv('./process_data/学生的性别_补卡_班车次数特征.csv')

df_2 = pd.read_csv('./process_data/学生每种消费方式的花费占总消费金额的比例特征提取.csv')

df_3 = pd.read_csv('./process_data/消费总次数+消费金额在0-10，10-20，20元以上之间的次数除以消费总次数特征提取.csv')

df_4 = pd.read_csv('./process_data/寒暑假是否在校特征.csv')

df_5 = pd.read_csv('./process_data/学生24小时的每个小时的消费次数除以活跃天数特征.csv')

df_6 = pd.read_csv('./process_data/活跃天数特征+高消费地点的消费次数+消费次数除以活跃天数特征.csv')

df_7 = pd.read_csv('./process_data/周末消费次数与工作日消费次数的比值特征.csv')

df_8 = pd.read_csv('./process_data/大众消费地点的消费次数特征.csv')

print(train_label.shape)


df_0.drop(columns=['is_poor'], inplace=True)
df_1.drop(columns=['is_poor'], inplace=True)
df_8.drop(columns=['is_poor'], inplace=True)

train_label = train_label.merge(df_0, on='xh', how='left')
train_label = train_label.merge(df_1, on='xh', how='left')
train_label = train_label.merge(df_2, on='xh', how='left')
train_label = train_label.merge(df_3, on='xh', how='left')
train_label = train_label.merge(df_4, on='xh', how='left')
train_label = train_label.merge(df_5, on='xh', how='left')
train_label = train_label.merge(df_6, on='xh', how='left')
train_label = train_label.merge(df_7, on='xh', how='left')
train_label = train_label.merge(df_8, on='xh', how='left')

print(train_label.shape)    # (24690, 112)

train_label.to_csv('./process_data/第三轮特征.csv', index=None)


# 划分训练集和验证集
# 分层抽样切分得到测试集test
def split_data(df):
    print("切分前数据数为{}".format(str(df.shape)))
    cols = [x for x in df.columns if x not in ['is_poor']]
    X = df[cols]
    y = df[['is_poor']]
    dataset1_X, dataset2_X, dataset1_y, dataset2_y = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y)
    print(dataset1_X.shape, dataset1_y.shape)
    print(dataset2_X.shape, dataset2_y.shape)

    dataset1 = dataset1_X.join(dataset1_y)
    dataset2 = dataset2_X.join(dataset2_y)
    print("训练集数据数为{}".format(str(dataset1.shape)))
    print("验证集数据数为{}".format(str(dataset2.shape)))

    return dataset1, dataset2


print('generate data...')
dataset1, dataset2 = split_data(train_label)
# dataset3 = test_user_feature_data

dataset1.to_csv('./process_data/3_lun_train_data.csv', index=None)
dataset2.to_csv('./process_data/3_lun_valid_data.csv', index=None)
# dataset3.to_csv(test_data_file_path, index=None)
print('generated ok!')

# generate data...
# 切分前数据数为(24690, 112)
# (19752, 111) (19752, 1)
# (4938, 111) (4938, 1)
# 训练集数据数为(19752, 112)
# 验证集数据数为(4938, 112)
# generated ok!