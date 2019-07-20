#!/usr/bin/python
# coding=UTF-8

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',30)


all_ykt_jyrz = pd.read_csv('./新增数据/merge_ykt_jyrz.csv',  names=['xh','jylx','jyje','jyrq','jysj','jydd','shdm','shmc','zdjh','ljykcs','jyye'])

print(all_ykt_jyrz.shape)   # (22161110, 11)


# 计算学生的活跃天数，即发生过交易记录的天数
df_4 = all_ykt_jyrz.groupby(by=['xh']).nunique().drop(columns = ['xh']).reset_index()
print(df_4.shape)    # (24702, 11)
df_4 = df_4[['xh', 'jyrq']].rename(columns={'jyrq': 'huoyuetianshu'})
print(df_4.shape)    # (24702, 2)

# 读取学生高消费地点的消费次数特征文件
gaoxiaofeicishu_tezheng = pd.read_csv('./process_data/高消费地点的消费次数特征.csv')
print(gaoxiaofeicishu_tezheng.shape)    # (24690, 8)

# 按照gaoxiaofeicishu_tezheng的学号进行merge
gaoxiaofeicishu_huoyuetianshu_tezheng = gaoxiaofeicishu_tezheng.merge(df_4, on='xh', how='left')

# 高消费次数除以学生的活跃天数
features = [x for x in gaoxiaofeicishu_huoyuetianshu_tezheng.columns if x not in ['xh', 'is_poor', 'huoyuetianshu']]
for feature in features:
    gaoxiaofeicishu_huoyuetianshu_tezheng['huoyuetianshu_'+feature] = gaoxiaofeicishu_huoyuetianshu_tezheng[feature] / gaoxiaofeicishu_huoyuetianshu_tezheng.huoyuetianshu

# 删除‘is_poor’字段
gaoxiaofeicishu_huoyuetianshu_tezheng.drop(columns=['is_poor'], inplace=True)
print(gaoxiaofeicishu_huoyuetianshu_tezheng.shape)    # (24690, 14)

gaoxiaofeicishu_huoyuetianshu_tezheng.to_csv('./process_data/活跃天数特征+高消费地点的消费次数+消费次数除以活跃天数特征.csv', index=None)
