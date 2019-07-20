#!/usr/bin/python
#coding=UTF-8

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',30)

import gc



# 学生基本数据信息（本科生） ok
bks_xsjbsjxx = pd.read_csv('./raw_data/bks_xsjbsjxx_out.csv')
# # 学籍基本数据信息（本科生）
# bks_xjjbsjxx = pd.read_csv('./bks_xjjbsjxx_out.csv')
# 本科生成绩信息
# bks_cjxx = pd.read_csv('./bks_cjxx_out.csv')

# # 排课数据信息（本科生）  ok
# bks_pksjxx = pd.read_csv('./bks_pksjxx_out.csv')

# # 课程数据信息（本科生）    ok
# bks_kcsjxx = pd.read_csv('./bks_kcsjxx_out.csv')

# 一卡通消费日志：YKT_JYRZ  ok
ykt_jyrz_2018 = pd.read_table('./新增数据/ykt_jyrz_2018.txt', sep=';')
ykt_jyrz_2019 = pd.read_table('./新增数据/ykt_jyrz_2019.txt', sep=';')


# def missing_data(df):
#     null_count = pd.Series(df.isnull().sum())
#     null_pct = pd.Series(df.isnull().sum() / df.shape[0])
#     column_types = pd.Series(df.dtypes)
#     missing_info = pd.concat([null_count, null_pct, column_types], axis=1, ignore_index=False)
#
#     missing_info.columns = ['null_count', 'null_pct', 'dtypes']
#
#     return missing_info.transpose()


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

# 1. 学生基本数据信息
# 一条记录未知性别可以删掉
print(bks_xsjbsjxx.shape)
bks_xsjbsjxx = bks_xsjbsjxx.loc[bks_xsjbsjxx.xb != '未知性别']
print(bks_xsjbsjxx.shape)

bks_xsjbsjxx.to_csv('./process_data/droped_bks_xsjbsjxx_out.csv', index=None)


# 2. 一卡通消费日志：YKT_JYRZ
ykt_jyrz = pd.concat([ykt_jyrz_2018, ykt_jyrz_2019], axis=0, ignore_index=True)
ykt_jyrz = data_downcast(ykt_jyrz)

# 将列名转换为小写
ykt_jyrz.columns = [col.lower() for col in ykt_jyrz.columns]

# 数据集中没有2020年的数据
print(ykt_jyrz.shape)
boo = ykt_jyrz.jyrq.apply(lambda x: True if x[:4] == '2020' else False)

# 删除一卡通消费日志中，学号异常的记录

print(ykt_jyrz.shape)    # (22176513, 11)
boo = ykt_jyrz.xh.apply(lambda x: True if x[:2] == '20' else False)
# sum(boo)    # 22161110
ykt_jyrz = ykt_jyrz[boo]
print(ykt_jyrz.shape)    # (22161110, 11)

ykt_jyrz.iloc[: 10000000].to_csv('./新增数据/merge_ykt_jyrz.csv', mode='a', index=None, header=None)
ykt_jyrz.iloc[10000000 : ].to_csv('./新增数据/merge_ykt_jyrz.csv', mode='a', index=None, header=None)