#!/usr/bin/python
#coding=UTF-8

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',30)


all_ykt_jyrz = pd.read_csv('./新增数据/merge_ykt_jyrz.csv',  names=['xh','jylx','jyje','jyrq','jysj','jydd','shdm','shmc','zdjh','ljykcs','jyye'])

# 只保留食堂消费数据
jiaoyididian = ['七食堂1楼', '七食堂2楼', '中心食堂一层','中心食堂二层', '中心食堂三层', '中心食堂四层', '中心食堂清真', '九食堂清真餐厅', '九食堂三楼', '沁园餐厅',
 '第十食堂', '五食堂一层','五食堂二层', '五食堂三层' ,'专招食堂',
 '开发区校区食堂','沁园餐厅','盘锦校区B12食堂','盘锦校区B13食堂']
boo = all_ykt_jyrz.jydd.apply(lambda x : True if x in jiaoyididian else False)
print(all_ykt_jyrz.shape)    # (22161110, 11)
# tmp表示在食堂消费的记录
tmp = all_ykt_jyrz[boo]
print(tmp.shape)    # (12842468, 11)

# 只保留交易类型为'持卡人消费'
print(tmp.shape)    # (12842468, 11)
tmp = tmp[tmp.jylx == '持卡人消费']
print(tmp.shape)    # (12841528, 11)

# 对学号，日期，交易发生的地点和发生的小时为合并字段进行合并，认为他们是一次消费，然后对交易金额求和
tmp['jiaoyiyhour'] = tmp.jysj.apply(lambda x : x[11:13])
tmp_meici = tmp.groupby(by=['xh', 'jyrq', 'jydd', 'jiaoyiyhour'], as_index=False)['jyje'].sum()

# 将交易日期字段转换为周x
print(tmp_meici.shape)    # (7517528, 5)
tmp_meici['xingqi_x'] = pd.to_datetime(tmp_meici.jyrq).apply(lambda x:x.weekday()+1)
print(tmp_meici.shape)    # (7517528, 6)
# 生成字段is_weekends,表明消费是否为周末
weekends = [6,7]
tmp_meici['is_weekends'] = tmp_meici.xingqi_x.apply(lambda x : 1 if x in weekends else 0)
print(tmp_meici.shape)    # (7517528, 7)

# 分别计算学生在周末和非周末的消费次数
df_is_weekends = tmp_meici[tmp_meici.is_weekends == 1]
df_no_weekends = tmp_meici[tmp_meici.is_weekends == 0]
print(df_is_weekends.shape)
print(df_no_weekends.shape)

## 按学号分组求count，代表学生消费的次数
df1 = df_is_weekends.groupby(by=['xh'], as_index=False).count()[['xh', 'jyrq']].rename(columns={'jyrq':'zhoumo_xiaofeicishu'})
df2 = df_no_weekends.groupby(by=['xh'], as_index=False).count()[['xh', 'jyrq']].rename(columns={'jyrq':'gongzuori_xiaofeicishu'})

print(df1.shape)    # (24161, 2)

print(df2.shape)    # (24393, 2)

# 将周末消费次数和非周末消费次数合并
df2 =  df2.merge(df1, how='left', on='xh')
print(df2.shape)    # (24393, 3)

# 计算学生周末在食堂消费的次数与工作日在食堂消费次数的比值
df2['bili_zhoumoshitangxiaofeicishu_gongzuorixiaofeicishu'] = df2.zhoumo_xiaofeicishu/df2.gongzuori_xiaofeicishu
print(df2.shape)    # (24393, 4)

# 周末不在食堂消费的，周末消费次数也低，为异常数据
df2.dropna(inplace=True)
print(df2.shape)    # (24135, 4)

# 将食堂消费消费次数小于75%分位数的学生比例设置为0
df2['bili_zhoumoshitangxiaofeicishu_gongzuorixiaofeicishu'] = pd.Series(list(map(lambda x, y : 0.0 if x<343.0 else y, df2.gongzuori_xiaofeicishu, df2.bili_zhoumoshitangxiaofeicishu_gongzuorixiaofeicishu)))

df2 = df2[['xh', 'bili_zhoumoshitangxiaofeicishu_gongzuorixiaofeicishu']]

df2.to_csv('./process_data/周末消费次数与工作日消费次数的比值特征.csv', index=None)