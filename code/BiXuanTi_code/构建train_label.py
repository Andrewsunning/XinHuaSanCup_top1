#!/usr/bin/python
# coding=UTF-8

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',30)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import time
import gc


# 降低内存使用
def data_downcast(data, msg='all_data'):
    print(msg+'===\n')
    print('memory_usage:', data.memory_usage(deep=True).sum()/1024**2)
    print('————————————————————————————————————————————')
    all_dataint = data.select_dtypes(include=['int64', 'int32']).apply(pd.to_numeric, downcast='integer')
    data[all_dataint.columns] = all_dataint
    print('int type downcast down')
    print('memory_usage:', data.memory_usage(deep=True).sum()/1024**2)
    print('————————————————————————————————————————————')
    all_datafloat = data.select_dtypes(include=['float64']).apply(pd.to_numeric, downcast='float')
    data[all_datafloat.columns] = all_datafloat
    print('memory_usage:', data.memory_usage(deep=True).sum()/1024**2)
    print('float type downcast down')
    del all_dataint
    del all_datafloat
    gc.collect()
    return data


ykt_jyrz = pd.read_csv('./新增数据/merge_ykt_jyrz.csv', names=['xh','jylx','jyje','jyrq','jysj','jydd','shdm','shmc','zdjh','ljykcs','jyye'])

# 降内存
ykt_jyrz = data_downcast(ykt_jyrz)

# 读入学生基本数据信息
bks_xsjbsjxx = pd.read_csv('./process_data/droped_bks_xsjbsjxx_out.csv')
print(bks_xsjbsjxx.shape)    # (32479, 13)



# 查看在ykt_jyrz和bks_xsjbsjxx中学号都出现的记录数
print(len(set(ykt_jyrz.xh) & set(bks_xsjbsjxx.xh)))    # 24690
train_label = pd.Series(list(set(ykt_jyrz.xh) & set(bks_xsjbsjxx.xh))).to_frame()
train_label.columns = ['xh']
print(train_label.shape)     # (24690, 1)

train_label.to_csv('./process_data/add_data_train_label.csv')    # 只有学号字段，没有label字段


# 1. 食堂消费占总消费的比例

# 只保留交易类型为'持卡人消费'
print(ykt_jyrz.shape)    # (22161110, 11)
ykt_jyrz = ykt_jyrz[ykt_jyrz.jylx == '持卡人消费']
print(ykt_jyrz.shape)    # (21902047, 11)

jiaoyididian = ['七食堂1楼', '七食堂2楼', '中心食堂一层','中心食堂二层', '中心食堂三层', '中心食堂四层', '中心食堂清真', '九食堂清真餐厅', '九食堂三楼', '沁园餐厅',
 '第十食堂', '五食堂一层','五食堂二层', '五食堂三层' ,'专招食堂',
 '开发区校区食堂','沁园餐厅','盘锦校区B12食堂','盘锦校区B13食堂']
boo = ykt_jyrz.jydd.apply(lambda x : True if x in jiaoyididian else False)
print(ykt_jyrz.shape)    # (21902047, 11)
# tmp表示在食堂消费的记录
tmp = ykt_jyrz[boo]
print(tmp.shape)    # (12841528, 11)

fenzi = tmp.groupby(by=['xh'], as_index=False)['jyje'].sum()
fenmu = ykt_jyrz.groupby(by=['xh'], as_index=False)['jyje'].sum()
print(fenmu.shape, fenzi.shape)     # (24662, 2) (24419, 2)

# 计算食堂消费占总消费的比例
df = fenmu.merge(fenzi, on='xh', how='left').rename(columns = {'jyje_x':'zongxiaofeijine', 'jyje_y':'shitangxiaofeijine'})
zhibiao_1 = df.shitangxiaofeijine / df.zongxiaofeijine
df['engeer_xishu'] = zhibiao_1

# 缺失值填充：将恩格尔西湖填充为0
df.fillna(0, inplace=True)

# 2. 食堂消费次数

# 对学号，日期，交易发生的地点和发生的小时为合并字段进行合并，认为他们是一次消费，然后对交易金额求和

tmp['jiaoyiyhour'] = tmp.jysj.apply(lambda x : x[11:13])
tmp_meici = tmp.groupby(by=['xh', 'jyrq', 'jydd', 'jiaoyiyhour'], as_index=False)['jyje'].sum()
print(tmp_meici.shape)    # (7517528, 5)

# 计算某一学生吃饭的次数
tmp2 = tmp_meici.groupby('xh').size().to_frame().reset_index()
tmp2.columns = ['xh', 'chifancishu']

# 3. 每次食堂消费的平均金额
# 计算某一学生吃饭的总金额和平均金额
tmp2['chifanzongjine'] = np.array(tmp_meici.groupby(by=['xh'])['jyje'].sum())
tmp2['chifanpingjunjine'] = tmp2.chifanzongjine / tmp2.chifancishu

# tmp2.shape    # (24419, 4)

# 4. 消费余额的均值

df_3 = tmp.groupby(by=['xh'], as_index=False)['jyye'].mean()
print(df_3.shape)    # (24419, 2)

# 5. 吃早餐的天数占学生活跃天数的比例

# 计算学生的活跃天数，即发生过交易记录的天数
df_4 = ykt_jyrz.groupby(by=['xh']).nunique().drop(columns = ['xh']).reset_index()
print(df_4.shape)    # (24662, 11)

df_4 = df_4[['xh', 'jyrq']].rename(columns={'jyrq': 'huoyuetianshu'})
print(df_4.shape)    # (24662, 2)

# 计算学生吃早餐的天数
zaocanshijianduan = ['05' , '06' ,'07', '08']  # 5.00 -- 9.00
t1 = tmp_meici.groupby(by=['xh', 'jiaoyiyhour'], as_index=False).count()
boo = t1.jiaoyiyhour.apply(lambda x : True if x in zaocanshijianduan else False)
t1 = t1[boo]
t1.rename(columns={'jyrq':'zaocancishu'}, inplace=True)
t2 = t1.groupby('xh', as_index=False).sum()
print(t2.shape)    # (23563, 4)

t2.drop(columns=['jydd', 'jyje'], inplace=True)
df_4 = df_4.merge(t2, on='xh', how='left')
print(df_4.shape)    # (24662, 3)

# zaocancishu的缺失值用0填充，表示其没有吃过早餐
df_4.fillna(0, inplace=True)
df_4['chizaocan_bili'] = df_4.zaocancishu / df_4.huoyuetianshu

# 6. 合并指标，为学生打标签
train_label =  pd.read_csv('./process_data/add_data_train_label.csv', index_col=0)
print(train_label.shape)    # (24690, 1)


# df---engeer_xishu
# tmp2---chifancishu,chifanpingjunjine
# df_3---jyye
# df_4---chizaocan_bili

print(df.shape)
print(tmp2.shape)
print(df_3.shape)
print(df_4.shape)

# (24662, 4)
# (24419, 4)
# (24419, 2)
# (24662, 4)

train_label = train_label.merge(df, on='xh', how='left')
train_label = train_label.merge(tmp2, on='xh', how='left')
train_label = train_label.merge(df_3, on='xh', how='left')
train_label = train_label.merge(df_4, on='xh', how='left')

# 删除train_label中没有用的字段['zongxiaofeijine', 'shitangxiaofeijine', 'huoyuetianshu','zaocancishu', 'chifanzongjine']
cols = ['xh', 'engeer_xishu', 'chifancishu', 'chifanpingjunjine', 'jyye', 'chizaocan_bili']
train_label = train_label[cols]

# 填充train_label的缺失值
values = {'engeer_xishu':0, 'chifancishu':0, 'chifanpingjunjine':train_label.chifanpingjunjine.max(), 'jyye':train_label.jyye.max(), 'chizaocan_bili':0}
train_label.fillna(value=values, inplace=True)

# 考虑到个字段均值的意义不大，所以采用归一化进行数据缩放
scaler = MinMaxScaler()
data = train_label[['engeer_xishu', 'chifancishu', 'chifanpingjunjine', 'jyye', 'chizaocan_bili']]
scaler.fit(data)
data = scaler.transform(data)

tmp3 = pd.DataFrame(data)
tmp3.columns = ['min_max_engeer_xishu', 'min_max_chifancishu', 'min_max_chifanpingjunjine', 'min_max_jyye', 'min_max_chizaocan_bili']
train_label = pd.concat([train_label, tmp3], axis=1)
print(train_label.shape)    # (24690, 11)

train_label = train_label[['xh', 'min_max_engeer_xishu', 'min_max_chifancishu', 'min_max_chifanpingjunjine', 'min_max_jyye', 'min_max_chizaocan_bili']]
print(train_label.shape)    # (24690, 6)

# 各字段权重值
# min_max_engeer_xishu  +0.4
# min_max_chifancishu +0.2
# min_max_chizaocan_bili  +0.10
# min_max_chifanpingjunjine -0.25
# min_max_jyye  -0.05

## 集成字段取值越大，越可能是贫困生
train_label['prob_is_poor'] = 0.4*train_label.min_max_engeer_xishu + 0.2*train_label.min_max_chifancishu +0.1*train_label.min_max_chizaocan_bili - 0.25*train_label.min_max_chifanpingjunjine - 0.05*train_label.min_max_jyye

# 打标签
yuzhi = train_label.prob_is_poor.quantile(0.86)
train_label['is_poor'] = train_label.prob_is_poor.apply(lambda x : 1 if x > yuzhi else 0)

train_label[['xh', 'is_poor']].to_csv('./process_data/add_data_2_train_label.csv', index=None)