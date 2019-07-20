#!/usr/bin/python
#coding=UTF-8

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',30)




all_ykt_jyrz = pd.read_csv('./新增数据/merge_ykt_jyrz.csv',  names=['xh','jylx','jyje','jyrq','jysj','jydd','shdm','shmc','zdjh','ljykcs','jyye'])


# 计算学生的活跃天数，即发生过交易记录的天数
df_4 = all_ykt_jyrz.groupby(by=['xh']).nunique().drop(columns = ['xh']).reset_index()
print(df_4.shape)    # (24702, 11)
df_4 = df_4[['xh', 'jyrq']].rename(columns={'jyrq': 'huoyuetianshu'})
print(df_4.shape)    # (24702, 2)

# 不仅考虑食堂消费数据，还要考虑其他类型的消费记录，同时我们只保留交易类型为'持卡人消费'
print(all_ykt_jyrz.shape)    # (22161110, 11)
all_ykt_jyrz = all_ykt_jyrz[all_ykt_jyrz.jylx == '持卡人消费']
print(all_ykt_jyrz.shape)    # (21902047, 11)

# 对学号，日期，交易发生的地点和发生的小时为合并字段进行合并，认为他们是一次消费，然后对交易金额求和
all_ykt_jyrz['jiaoyiyhour'] = all_ykt_jyrz.jysj.apply(lambda x : x[11:13])
all_ykt_jyrz_meici = all_ykt_jyrz.groupby(by=['xh', 'jyrq', 'jydd', 'jiaoyiyhour'], as_index=False)['jyje'].sum()

# 排除浴室、洗衣机、电费等消费记录
paichu_jydd = ['北山AB区热水器',
'北山B区浴室',
 '北山C区热水器','北山浴室','卢工洗衣机','图书馆机房管理系统',
 '圈存缴网费','开发区开水机',
 '开发区智能控电',
 '开发区机房系统',
 '开发区浴室','开发区网络计费',
 '教务注册大厅收费','服务大厅',
 '本科生图像采集',
 '机房管理商户','梁工洗衣机',
 '汇文系统','班车',
 '电子缴电费','西山浴室',
 '西山热水器',
 '赵工洗衣机','郭顺发洗衣机']

boo = all_ykt_jyrz_meici.jydd.apply(lambda x:False if x in paichu_jydd else True)
print(all_ykt_jyrz_meici.shape)    # (13660610, 5)
all_ykt_jyrz_meici = all_ykt_jyrz_meici[boo]
print(all_ykt_jyrz_meici.shape)    # (8996482, 5)


# 该计算每名学生在24个小时内的消费次数咯qq
xiaoshi_xiaofeicishu = all_ykt_jyrz_meici.groupby(by=['xh', 'jiaoyiyhour'], as_index=False).count()[['xh', 'jiaoyiyhour','jyrq']].rename(columns={'jyrq':'xiaoshi_xiefeicishu'})
print(xiaoshi_xiaofeicishu.shape)    # (380401, 3)

# 读入label标签
train_label = pd.read_csv('./process_data/add_data_2_train_label.csv')

# 计算学生在每个小时内的消费次数
xiaoshi = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15',
 '16', '17', '18', '19', '20', '21', '22','23']
print(train_label.shape)    # (24690, 2)
for feature in xiaoshi:
    t = xiaoshi_xiaofeicishu[xiaoshi_xiaofeicishu.jiaoyiyhour == feature]
    train_label = train_label.merge(t[['xh', 'xiaoshi_xiefeicishu']], on='xh', how='left').rename(columns={'xiaoshi_xiefeicishu' : feature+'_xiaoshi_xiefeicishu'})
print(train_label.shape)    # (24690, 26)

# 删除‘is_poor’字段，并用0填缺失值
print(train_label.shape)
train_label.drop(columns=['is_poor'], inplace=True)
print(train_label.shape)
train_label.fillna(0, inplace=True)

# 每小时消费次数除以活跃天数
train_label = train_label.merge(df_4, on='xh', how='left')
print(train_label.shape)    # (24690, 26)
features = [x for x in train_label.columns if x not in ['xh', 'huoyuetianshu']]
for feature in features:
    train_label[feature + '_chuyihuoyuetianshu'] = train_label[feature] / train_label.huoyuetianshu
print(train_label.shape)    # (24690, 50)

train_label.drop(columns=features, inplace=True)
# print(train_label.shape)
train_label.drop(columns=['huoyuetianshu'], inplace=True)
print(train_label.shape)    # (24690, 26)

train_label.to_csv('./process_data/学生24小时的每个小时的消费次数除以活跃天数特征.csv', index=None)