#!/usr/bin/python
#coding=UTF-8

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


# # 按地点进行分组，并求学生在各地点的消费次数
# df1 = tmp.groupby(by=['jydd'], as_index=False).count()[['jydd', 'jyje']].rename(columns={'jyje':'zuiduo_xiaofeicishu'})
# df1.sort_values(by='zuiduo_xiaofeicishu', axis=0, ascending=False)


# 提取出在最多消费地点消费的记录tmp_2
zuiduo_xiaofeididian = ['中心食堂一层', '开发区校区食堂', '五食堂一层', '大学生超市', '五食堂二层', '中心食堂二层', '沁园餐厅']
boo1 = tmp.jydd.apply(lambda x : True if x in zuiduo_xiaofeididian else False)
print(tmp.shape)
tmp_2 = tmp[boo1]
print(tmp_2.shape)    # (7794367, 5)


# 按照学号和交易地点分组求次数count，计算每个学生在上述大众消费地点的消费次数
df_2 = tmp_2.groupby(by=['xh', 'jydd'], as_index=False).count()[['xh', 'jydd', 'jyrq']].rename(columns={'jyrq':'zuiduo_xiaofeicishu'})

# 读入label标签
train_label = pd.read_csv('./process_data/add_data_2_train_label.csv')

# 计算特征
features = ['中心食堂一层', '开发区校区食堂', '五食堂一层', '大学生超市', '五食堂二层', '中心食堂二层', '沁园餐厅']
print(train_label.shape)    # (24690, 2)
for feature in features:
    t = df_2[df_2.jydd == feature]
    train_label = train_label.merge(t[['xh', 'zuiduo_xiaofeicishu']], on='xh', how='left').rename(columns={'zuiduo_xiaofeicishu':'zuiduo_xiaofeicishu_'+feature})
print(train_label.shape)    # (24690, 9)

# 用0填充缺失值
train_label.fillna(0, inplace=True)

train_label.to_csv('./process_data/大众消费地点的消费次数特征.csv', index=None)