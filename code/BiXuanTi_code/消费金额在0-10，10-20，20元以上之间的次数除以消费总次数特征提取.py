#!/usr/bin/python
# coding=UTF-8

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',30)

all_ykt_jyrz = pd.read_csv('./新增数据/merge_ykt_jyrz.csv',  names=['xh','jylx','jyje','jyrq','jysj','jydd','shdm','shmc','zdjh','ljykcs','jyye'])

# 不仅考虑食堂消费数据，还要考虑其他类型的消费记录，同时我们只保留交易类型为'持卡人消费'
print(all_ykt_jyrz.shape)    # (22161110, 11)
all_ykt_jyrz = all_ykt_jyrz[all_ykt_jyrz.jylx == '持卡人消费']
print(all_ykt_jyrz.shape)    # (21902047, 11)

# 对学号，日期，交易发生的地点和发生的小时为合并字段进行合并，认为他们是一次消费，然后对交易金额求和
all_ykt_jyrz['jiaoyiyhour'] = all_ykt_jyrz.jysj.apply(lambda x : x[11:13])
all_ykt_jyrz_meici = all_ykt_jyrz.groupby(by=['xh', 'jyrq', 'jydd', 'jiaoyiyhour'], as_index=False)['jyje'].sum()

# 计算每个学生的消费总次数
tmp = all_ykt_jyrz_meici.groupby(by=['xh'], as_index=False).count()[['xh', 'jyrq']].rename(columns={'jyrq':'xiaofeizongcishu'})

# 提取出消费金额在0-10元的消费记录
boo_1 = all_ykt_jyrz_meici.jyje.apply(lambda x:True if x<1000 else False)
xiaofeijine_0_10 = all_ykt_jyrz_meici[boo_1]
print(xiaofeijine_0_10.shape)    # (10099333, 5)

# 计算每名学生在0-10元之间的消费次数
xiaofeicishu_0_10 = xiaofeijine_0_10.groupby(by=['xh'], as_index=False).count()[['xh', 'jyrq']].rename(columns={'jyrq':'xiaofeicishu_0_10'})


# 提取出消费金额在10-20元的消费记录
boo_2 = all_ykt_jyrz_meici.jyje.apply(lambda x:True if ((x>=1000) and (x<=2000)) else False)
xiaofeijine_10_20 = all_ykt_jyrz_meici[boo_2]
print(xiaofeijine_10_20.shape)    # (3172943, 5)


# 计算每名学生在10-20元之间的消费次数
xiaofeicishu_10_20 = xiaofeijine_10_20.groupby(by=['xh'], as_index=False).count()[['xh', 'jyrq']].rename(columns={'jyrq':'xiaofeicishu_10_20'})


# 提取出消费金额在20元以上的消费记录
boo_3 = all_ykt_jyrz_meici.jyje.apply(lambda x:True if x>2000 else False)
xiaofeijine_dayu20 = all_ykt_jyrz_meici[boo_3]
print(xiaofeijine_dayu20.shape)


# 计算每名学生在20元以上的消费次数
xiaofeicishu_dayu20 = xiaofeijine_dayu20.groupby(by=['xh'], as_index=False).count()[['xh', 'jyrq']].rename(columns={'jyrq':'xiaofeicishu_dayu20'})


# 读入label标签
train_label = pd.read_csv('./process_data/add_data_2_train_label.csv')

print(train_label.shape)    # (24690, 2)
train_label = train_label.merge(tmp, on='xh', how='left')
train_label = train_label.merge(xiaofeicishu_0_10, on='xh', how='left')
train_label = train_label.merge(xiaofeicishu_10_20, on='xh', how='left')
train_label = train_label.merge(xiaofeicishu_dayu20, on='xh', how='left')
print(train_label.shape)    # (24690, 6)

# 删除总消费次数为nan的记录
tmp2 =  train_label[train_label.xiaofeizongcishu.notna()]

# 保留消费总次数大于75%分位数的记录
print(tmp2.shape)    # (24651, 6)
tmp2 = tmp2[tmp2.xiaofeizongcishu > 769.0]
print(tmp2.shape)    # (6145, 6)

# 用0填充xiaofeicishu_dayu20的缺失值
tmp2.fillna(0, inplace=True)

train_label = train_label[['xh', 'xiaofeizongcishu']]
print(train_label.shape)    # (24690, 2)
train_label = train_label.merge(tmp2[['xh', 'xiaofeicishu_0_10', 'xiaofeicishu_10_20', 'xiaofeicishu_dayu20']], how='left', on='xh')
print(train_label.shape)    # (24690, 5)

# 计算各区间消费次数占总消费次数的比例
features = ['xiaofeicishu_0_10', 'xiaofeicishu_10_20', 'xiaofeicishu_dayu20']
for feature in features:
    train_label['bili_zongcishu_' + feature] = train_label[feature]/train_label.xiaofeizongcishu


# 删除原始次数西段
train_label.drop(columns=features, inplace=True)

# 填充缺失值
# 设置字段和填充值的对应关系
values = {'bili_zongcishu_xiaofeicishu_0_10':train_label.bili_zongcishu_xiaofeicishu_0_10.min(), 'bili_zongcishu_xiaofeicishu_10_20':train_label.bili_zongcishu_xiaofeicishu_10_20.mean(), 'bili_zongcishu_xiaofeicishu_dayu20':train_label.bili_zongcishu_xiaofeicishu_dayu20.max()}
train_label.fillna(value=values, inplace=True)

# 消费总次数用0填充
train_label.fillna(0, inplace=True)

train_label.to_csv('./process_data/消费总次数+消费金额在0-10，10-20，20元以上之间的次数除以消费总次数特征提取.csv', index=None)