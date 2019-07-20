#!/usr/bin/python
# coding=UTF-8

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',30)

# 读入学生基本数据信息
bks_xsjbsjxx = pd.read_csv('./process_data/droped_bks_xsjbsjxx_out.csv')
print(bks_xsjbsjxx.shape)    # (32479, 13)

# 性别特征进行one-hot编码，并拼接到原DataFrame
xingbie = pd.get_dummies(bks_xsjbsjxx.xb)
bks_xsjbsjxx = pd.concat([bks_xsjbsjxx, xingbie], axis=1)

# 修改列名并截取性别和学号特征
bks_xsjbsjxx.rename(columns={'女':'xingbie_nv', '男':'xingbie_nan'}, inplace=True)
bks_xsjbsjxx = bks_xsjbsjxx[['xh', 'xingbie_nv','xingbie_nan']]
print(bks_xsjbsjxx.shape)    # (32479, 3)

# 读入label标签
train_label = pd.read_csv('./process_data/add_data_2_train_label.csv')

train_label = train_label.merge(bks_xsjbsjxx, on='xh', how='left')
print(train_label.shape)    # (24690, 4)


# 学生是否补过卡特征
all_ykt_jyrz = pd.read_csv('./新增数据/merge_ykt_jyrz.csv',  names=['xh','jylx','jyje','jyrq','jysj','jydd','shdm','shmc','zdjh','ljykcs','jyye'])

# 不仅考虑食堂消费数据，还要考虑其他类型的消费记录，同时我们只保留交易类型为'持卡人消费'
print(all_ykt_jyrz.shape)    # (22161110, 11)
all_ykt_jyrz = all_ykt_jyrz[all_ykt_jyrz.jylx == '持卡人消费']
print(all_ykt_jyrz.shape)    # (21902047, 11)

# 提取出学生'补卡'记录
boo_1 = all_ykt_jyrz.jydd.apply(lambda x : True if x=='自助补卡收卡成本' else False)
tmp1 = all_ykt_jyrz[boo_1]
print(tmp1.shape)    # (125, 11)


tmp1 = tmp1.groupby(by=['xh'], as_index=False).count()[['xh', 'jylx']]
tmp1.rename(columns={'jylx':'has_buka'}, inplace=True)
print(tmp1.shape)    # (124, 2)


train_label = train_label.merge(tmp1, on='xh', how='left')
train_label.fillna(0, inplace=True)
print(train_label.shape)    # (24690, 5)

# 提取出学生'班车'记录
boo_2 = all_ykt_jyrz.jydd.apply(lambda x : True if x=='班车' else False)
tmp2 = all_ykt_jyrz[boo_2]
print(tmp2.shape)    # (46326, 11)


tmp2 = tmp2.groupby(by=['xh'], as_index=False).count()[['xh', 'jylx']]
tmp2.rename(columns={'jylx':'banchecishu'}, inplace=True)
print(tmp2.shape)    # (6745, 2)

# 计算学生的活跃天数，即发生过交易记录的天数
df_4 = all_ykt_jyrz.groupby(by=['xh']).nunique().drop(columns = ['xh']).reset_index()
print(df_4.shape)    # (24662, 11)
df_4 = df_4[['xh', 'jyrq']].rename(columns={'jyrq': 'huoyuetianshu'})
print(df_4.shape)    # (24662, 2)

tmp2 = tmp2.merge(df_4, on='xh', how='left')
print(tmp2.shape)    # (6745, 3)

tmp2['bili_huoyuetianshu_banchecishu'] = tmp2.banchecishu/tmp2.huoyuetianshu
tmp2.drop(columns=['banchecishu', 'huoyuetianshu'], inplace=True)
print(tmp2.shape)

train_label = train_label.merge(tmp2, on='xh', how='left')
print(train_label.shape)    # (24690, 6)

# 用0填充缺失值
train_label.fillna(0, inplace=True)

train_label.to_csv('./process_data/学生的性别_补卡_班车次数特征.csv', index=None)