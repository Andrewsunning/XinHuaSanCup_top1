#!/usr/bin/python
# coding=UTF-8

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',30)

import numpy as np



all_ykt_jyrz = pd.read_csv('./新增数据/merge_ykt_jyrz.csv',  names=['xh','jylx','jyje','jyrq','jysj','jydd','shdm','shmc','zdjh','ljykcs','jyye'])


# 不仅考虑食堂消费数据，还要考虑其他类型的消费记录，同时我们只保留交易类型为'持卡人消费'
print(all_ykt_jyrz.shape)    # (22161110, 11)
all_ykt_jyrz = all_ykt_jyrz[all_ykt_jyrz.jylx == '持卡人消费']
print(all_ykt_jyrz.shape)    # (21902047, 11)


# 学生的总消费金额
tmp1 = all_ykt_jyrz.groupby(by=['xh'], as_index=False).sum()[['xh', 'jyje']].rename(columns={'jyje':'zongxiaofeijine'})
print(tmp1.shape)    # (24662, 2)

## change here
canyin_fanwei =  ['七食堂1楼', '七食堂2楼', '专招食堂', '中心食堂一层', '中心食堂三层', '中心食堂二层', '中心食堂四层', '中心食堂清真', '九食堂三楼', '九食堂清真餐厅', '五食堂一层', '五食堂三层', '五食堂二层','博留一层' ,'开发区校区食堂','沁园餐厅', '盘锦校区B12食堂','盘锦校区B13食堂','第十食堂']

boo_1 = all_ykt_jyrz.jydd.apply(lambda x : True if x in canyin_fanwei else False)
canyinxiaofei = all_ykt_jyrz[boo_1]
print(canyinxiaofei.shape)    # (12864326, 11)

# 学生的餐饮消费金额
tmp2 = canyinxiaofei.groupby(by=['xh'], as_index=False).sum()[['xh', 'jyje']].rename(columns={'jyje':'canyinxiaofeijine'})
print(tmp2.shape)    # (24447, 2)

## change here
richang_fanwei =  ['校医院','京鹤直饮水', '北山AB区热水器', '北山B区浴室', '北山C区热水器','北山浴室' ,'卢工洗衣机','圈存缴网费', '开发区开水机','开发区智能控电','开发区浴室','开发区网络计费', '梁工洗衣机','电子缴电费','西山浴室', '西山热水器', '赵工洗衣机', '郭顺发洗衣机']

boo_2 = all_ykt_jyrz.jydd.apply(lambda x : True if x in richang_fanwei else False)
richangxiaofei = all_ykt_jyrz[boo_2]
print(richangxiaofei.shape)    # (7808793, 11)

# 学生的日常消费金额
tmp3 = richangxiaofei.groupby(by=['xh'], as_index=False).sum()[['xh', 'jyje']].rename(columns={'jyje':'richangxiaofeijine'})
print(tmp3.shape)    # (24413, 2)

## change here
yudong_fanwei = ['体育馆','北山体育馆','软件学院体育馆']

boo_3 = all_ykt_jyrz.jydd.apply(lambda x : True if x in yudong_fanwei else False)
yundongxiaofei = all_ykt_jyrz[boo_3]
print(yundongxiaofei.shape)    # (99550, 11)

# 学生的运动消费金额
tmp4 = yundongxiaofei.groupby(by=['xh'], as_index=False).sum()[['xh', 'jyje']].rename(columns={'jyje':'yundongxiaofeijine'})
print(tmp4.shape)    # (11145, 2)

## change here
riyong_fanwei = ['十食堂超市','大学生超市','班车','自助补卡收卡成本']

boo_4 = all_ykt_jyrz.jydd.apply(lambda x : True if x in riyong_fanwei else False)
riyongxiaofei = all_ykt_jyrz[boo_4]
print(riyongxiaofei.shape)   # (1062707, 11)

# 学生的日用消费金额
tmp5 = riyongxiaofei.groupby(by=['xh'], as_index=False).sum()[['xh', 'jyje']].rename(columns={'jyje':'riyongxiaofeijine'})
print(tmp5.shape)    # (20262, 2)

## change here
shechi_fanwei = ['博留咖啡厅','教育书店咖啡厅']

boo_5 = all_ykt_jyrz.jydd.apply(lambda x : True if x in shechi_fanwei else False)
shechixiaofei = all_ykt_jyrz[boo_5]
print(shechixiaofei.shape)    # (5053, 11)


# 学生的奢侈消费金额
tmp6 = shechixiaofei.groupby(by=['xh'], as_index=False).sum()[['xh', 'jyje']].rename(columns={'jyje':'shechixiaofeijine'})
print(tmp6.shape)    # (1646, 2)


## change here
xuexi_fanwei = ['图书馆POS','图书馆机房管理系统','建艺学院','开发区机房系统', '机房管理商户', '汇文系统','英语四六级']

boo_6 = all_ykt_jyrz.jydd.apply(lambda x : True if x in xuexi_fanwei else False)
xuexixiaofei = all_ykt_jyrz[boo_6]
print(xuexixiaofei.shape)    # (58360, 11)

# 学生的学习消费金额
tmp7 = xuexixiaofei.groupby(by=['xh'], as_index=False).sum()[['xh', 'jyje']].rename(columns={'jyje':'xuexixiaofeijine'})
print(tmp7.shape)    # (15356, 2)

# 读入label标签
train_label = pd.read_csv('./process_data/add_data_2_train_label.csv')

# 合并各字段
tmps = [tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7]
for t in tmps:
    train_label = train_label.merge(t, on='xh', how='left')
print(train_label.shape)    # (24690, 9)

# 计算各消费类型占消费总金额的比例
features = ['canyinxiaofeijine', 'richangxiaofeijine', 'yundongxiaofeijine','riyongxiaofeijine', 'shechixiaofeijine', 'xuexixiaofeijine']

for feature in features:
    train_label['bili_zongxiaofeijine_'+feature] = train_label[feature]/train_label.zongxiaofeijine

print(train_label.shape)    # (24690, 15)

# 删除原始消费特征
train_label.drop(columns=features, inplace=True)

# 缺失值填充，总金额填充最大值，其他填充0
train_label['zongxiaofeijine'] = train_label.zongxiaofeijine.replace(np.nan, train_label.zongxiaofeijine.max())

# 0值填充
train_label.fillna(0, inplace=True)
# 删除'zongxiaofeijine', 'is_poor'两个特征
train_label.drop(columns=['zongxiaofeijine', 'is_poor'], inplace=True)
print(train_label.shape)    # (24690, 7)

train_label.to_csv('./process_data/学生每种消费方式的花费占总消费金额的比例特征提取.csv', index=None)
