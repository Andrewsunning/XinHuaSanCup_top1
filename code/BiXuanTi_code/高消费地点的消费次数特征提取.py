#!/usr/bin/python
# coding=UTF-8

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',30)


all_ykt_jyrz = pd.read_csv('./新增数据/merge_ykt_jyrz.csv',  names=['xh','jylx','jyje','jyrq','jysj','jydd','shdm','shmc','zdjh','ljykcs','jyye'])


# 只保留交易类型为'持卡人消费'
# 其他交易类型后续考虑
print(all_ykt_jyrz.shape)    # (22161110, 11)
all_ykt_jyrz = all_ykt_jyrz[all_ykt_jyrz.jylx == '持卡人消费']
print(all_ykt_jyrz.shape)    # (21902047, 11)


# 商户名称和交易地点取值全部相同，所以删除商户名称字段
all_ykt_jyrz.drop(columns=['shmc'], inplace=True)
print(all_ykt_jyrz.shape)    # (21902047, 10)


# 将学生的交易记录按照学号，交易日期，交易地点，交易小时合并求和,认为他们是一次消费，然后对交易金额求和
# tmp表示按次消费记录，all_ykt_jyrz只表示刷卡记录
print(all_ykt_jyrz.shape)    # (21902047, 10)
all_ykt_jyrz['jiaoyiyhour'] = all_ykt_jyrz.jysj.apply(lambda x : x[11:13])
tmp = all_ykt_jyrz.groupby(by=['xh', 'jyrq', 'jydd', 'jiaoyiyhour'], as_index=False)['jyje'].sum()
print(tmp.shape)    # (13660610, 5)


# 学生在上述7个地点的消费次数
gaoxiaofeididian = ['图书馆POS', '中心食堂四层', '博留咖啡厅', '建艺学院', '教育书店咖啡厅', '专招食堂']
print(tmp.shape)    # (13660610, 5)
boo_1 = tmp.jydd.apply(lambda x : True if x in gaoxiaofeididian else False)
tmp_2 = tmp[boo_1]
print(tmp_2.shape)    # (6077, 5)

tmp_2 = tmp_2.groupby(by=['xh', 'jydd'], as_index=False)['jyrq'].count().rename(columns={'jyrq':'gaoxiaofei_jiaoyicishu'})

# 读入label标签
train_label = pd.read_csv('./process_data/add_data_2_train_label.csv')

features = ['教育书店咖啡厅', '图书馆POS', '中心食堂四层', '博留咖啡厅', '建艺学院', '专招食堂']
print(train_label.shape)    # (24690, 2)
for feature in features:
    t = tmp_2[tmp_2.jydd == feature]
    train_label = train_label.merge(t[['xh', 'gaoxiaofei_jiaoyicishu']], on='xh', how='left').rename(columns={'gaoxiaofei_jiaoyicishu':'gaoxiaofeicishu_'+feature})
print(train_label.shape)    # (24690, 8)


# 用0填充缺失值
train_label.fillna(0, inplace=True)

train_label.to_csv('./process_data/高消费地点的消费次数特征.csv', index=None)

