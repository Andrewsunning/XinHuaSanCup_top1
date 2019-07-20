#!/usr/bin/python
# coding=UTF-8

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',30)

import matplotlib.pyplot as plt




all_ykt_jyrz = pd.read_csv('./新增数据/merge_ykt_jyrz.csv',  names=['xh','jylx','jyje','jyrq','jysj','jydd','shdm','shmc','zdjh','ljykcs','jyye'])

# 不仅考虑食堂消费数据，还要考虑其他类型的消费记录，同时我们只保留交易类型为'持卡人消费'
print(all_ykt_jyrz.shape)    # (22161110, 11)
all_ykt_jyrz = all_ykt_jyrz[all_ykt_jyrz.jylx == '持卡人消费']
print(all_ykt_jyrz.shape)    # (21902047, 11)

# 对学号，日期，交易发生的地点和发生的小时为合并字段进行合并，认为他们是一次消费，然后对交易金额求和
all_ykt_jyrz['jiaoyiyhour'] = all_ykt_jyrz.jysj.apply(lambda x : x[11:13])
all_ykt_jyrz_meici = all_ykt_jyrz.groupby(by=['xh', 'jyrq', 'jydd', 'jiaoyiyhour'], as_index=False)['jyje'].sum()

# 按照交易日期分组，计算每天发生交易的次数，并绘制趋势折线图
tmp = all_ykt_jyrz_meici.groupby(by=['jyrq'], as_index=False).count()[['jyrq', 'xh']].rename(columns={'xh':'xiaofeicishu'})
tmp['jyrq'] = pd.to_datetime(tmp.jyrq)
tmp.set_index('jyrq', inplace=True)
plt.figure(figsize=(10,10))
tmp['xiaofeicishu'].plot()

# 将在2月1日---2月15日有过消费记录的学生视为寒假留校    02-01 --- 02-15
# 将在8月1日---8月15日有过消费记录的学生视为暑假留校    08-01 --- 08-15
hanjiafanwei = ['02-01', '02-02', '02-03', '02-04', '02-05', '02-06', '02-07', '02-08',
               '02-09', '02-10', '02-11', '02-12', '02-13', '02-14', '02-15']
shujiafanwei = ['08-01', '08-02', '08-03', '08-04','08-05','08-06','08-07', '08-08',
                '08-09', '08-10','08-11', '08-12', '08-13', '08-14', '08-15']

# 生成‘jiaoyiyueri’字段，表示消费记录发生的月份和日子
tmp2 = all_ykt_jyrz_meici.groupby(by=['xh', 'jyrq'], as_index=False).count()[['xh', 'jyrq']]
tmp2['jiaoyiyueri'] = tmp2.jyrq.apply(lambda x:x[5:])

# 计算寒假是否在校
boo1 = tmp2.jiaoyiyueri.apply(lambda x : True if x in hanjiafanwei else False)
hanjiazaixiao = tmp2[boo1]

hanjiazaixiao = pd.DataFrame(list(set(hanjiazaixiao.xh)) , columns=['xh'])
hanjiazaixiao['zaixiao_hanjia'] = 1

# 读入label标签
train_label = pd.read_csv('./process_data/add_data_2_train_label.csv')

# hanjiazaixiao 与 train_label按照学号合并
print(train_label.shape)    # (24690, 2)
train_label = train_label.merge(hanjiazaixiao, how='left', on='xh')
print(train_label.shape)    # (24690, 3)


print(train_label.shape)    # (24690, 3)
train_label.drop(columns=['is_poor'], inplace=True)
train_label.fillna(0, inplace=True)
print(train_label.shape)    # (24690, 2)

train_label.to_csv('./process_data/寒暑假是否在校特征.csv', index=None)