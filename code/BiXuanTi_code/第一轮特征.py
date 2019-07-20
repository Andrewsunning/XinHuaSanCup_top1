# !/usr/bin/python
# coding=UTF-8

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',30)


# 学生基本数据信息（本科生） ok
bks_xsjbsjxx = pd.read_csv('./process_data/droped_bks_xsjbsjxx_out.csv')
# # 学籍基本数据信息（本科生）
# bks_xjjbsjxx = pd.read_csv('./bks_xjjbsjxx_out.csv')
# 本科生成绩信息
# bks_cjxx = pd.read_csv('./droped_bks_cjxx_out.csv')

# # 排课数据信息（本科生）  ok
# bks_pksjxx = pd.read_csv('./bks_pksjxx_out.csv')

# # 课程数据信息（本科生）    ok
# bks_kcsjxx = pd.read_csv('./bks_kcsjxx_out.csv')

# 一卡通消费日志：YKT_JYRZ  ok
ykt_jyrz = pd.read_csv('./新增数据/merge_ykt_jyrz.csv', names=['xh','jylx','jyje','jyrq','jysj','jydd','shdm','shmc','zdjh','ljykcs','jyye'])

# label标签
train_label = pd.read_csv('./process_data/add_data_2_train_label.csv')

y = train_label['is_poor']


def get_feature(ykt_jyrz, label):
    for feature in ykt_jyrz.columns[1:]:
        if ykt_jyrz[feature].dtype == 'object':
            label = label.merge(ykt_jyrz.groupby(by='xh')[feature].count().reset_index().rename(columns = {feature:'count_'+ feature}), how='left', on='xh')
            label = label.merge(ykt_jyrz.groupby(by='xh')[feature].nunique().reset_index().rename(columns = {feature:'nunique_'+ feature}), how='left', on='xh')
        else:
            label =label.merge(ykt_jyrz.groupby(['xh'])[feature].count().reset_index().rename(columns = {feature:'count_'+ feature}),on='xh',how='left')
            label =label.merge(ykt_jyrz.groupby(['xh'])[feature].nunique().reset_index().rename(columns = {feature:'nunique_'+ feature}),on='xh',how='left')
            label =label.merge(ykt_jyrz.groupby(['xh'])[feature].mean().reset_index().rename(columns = {feature:'mean_'+ feature}),on='xh',how='left')
            label =label.merge(ykt_jyrz.groupby(['xh'])[feature].std().reset_index().rename(columns = {feature:'std_'+ feature}),on='xh',how='left')
            label =label.merge(ykt_jyrz.groupby(['xh'])[feature].max().reset_index().rename(columns = {feature:'max_'+ feature}),on='xh',how='left')
            label =label.merge(ykt_jyrz.groupby(['xh'])[feature].min().reset_index().rename(columns = {feature:'min_'+ feature}),on='xh',how='left')
            label =label.merge(ykt_jyrz.groupby(['xh'])[feature].sum().reset_index().rename(columns = {feature:'sum_'+ feature}),on='xh',how='left')
            label =label.merge(ykt_jyrz.groupby(['xh'])[feature].skew().reset_index().rename(columns = {feature:'skew_'+ feature}),on='xh',how='left')
    return label

# 提取特征
train_valid_data = get_feature(ykt_jyrz, train_label)
print(train_valid_data.shape)    # (24690, 52)


train_valid_data.to_csv('./process_data/第一轮特征.csv', index=None)