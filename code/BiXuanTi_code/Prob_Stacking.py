#!/usr/bin/python
#coding=UTF-8

import warnings
warnings.filterwarnings('ignore')

import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

# 导入模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from xgboost.sklearn import XGBClassifier

# 导入stacking包
from vecstack import stacking

model_lgb = lgb.LGBMClassifier(
    learning_rate=0.05,  # 重点，最后调
    colsample_bytree=0.8,  # 修改为最优参数=0.8
    subsample=0.7,    # 修改为最优参数=0.7
    num_leaves=90,    # 重点，修改为最优参数=90
    min_child_weight=4, # 防止过拟合
    min_data_in_leaf=4,
    gamma=0,
    n_estimators=1522,  # 以上参数需要调参
    max_depth=-1,
    is_unbalance=True,
    boosting_type = 'gbdt',
    objective='binary',
    random_state=2019,
    n_jobs=2
    )

svc1 = SVC(random_state=2019, probability=True, C=56, gamma=0.6, class_weight={0:0.16, 1:1})


xgb1 = XGBClassifier(
    learning_rate=0.1,
    n_estimators=845,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    seed=0
    )

rf4 = RandomForestClassifier(n_estimators=235, max_features = 0.85, min_samples_split=2, min_samples_leaf=5,
                             max_depth=19, random_state=2019, class_weight={0:0.16, 1:1})


# 读入训练集和验证集
dataset1 = pd.read_csv('./process_data/3_lun_train_data.csv')
dataset2 = pd.read_csv('./process_data/3_lun_valid_data.csv')

# dataset1.head()

# 构建训练集的特征和标签
features = [x for x in dataset1.columns if x not in ['xh', 'is_poor']]
train_x = dataset1[features]
train_y = dataset1['is_poor']

val_x = dataset2[features]
val_y = dataset2['is_poor']

from sklearn.preprocessing import MinMaxScaler

# 初始化缩放器
scaler = MinMaxScaler()

# 拟合，计算缩放操作需要的个字段的最大和最小值
scaler.fit(train_x)
train_x = scaler.transform(train_x)
val_x = scaler.transform(val_x)
print(train_x.shape)    # (19752, 110)
print(val_x.shape)    # (4938, 110)


# -1填充缺失值
train_x[np.isnan(train_x)] = -1
val_x[np.isnan(val_x)] = -1

# 创建基学习器集合
models = [
    svc1,xgb1,model_lgb,rf4
]


def roc_auc_score_universal(y_true, y_pred):
    """ROC AUC metric for both binary and multiclass classification.

    Parameters
    ----------
    y_true - 1d numpy array
        True class labels
    y_pred - 2d numpy array
        Predicted probabilities for each class
    """
    ohe = OneHotEncoder(sparse=False)
    y_true = ohe.fit_transform(y_true.reshape(-1, 1))
    # @@@@
    if len(y_pred.shape) == 1:
        y_pred = np.c_[y_pred, y_pred]
        y_pred[:, 0] = 1 - y_pred[:, 1]
    # @@@@
    auc_score = roc_auc_score(y_true, y_pred)
    return auc_score


# S_train:stacking操作后返回的训练集
# S_test:stacking操作后返回的验证集
'''
参数解释：
    models：进行stacking操作的模型集合
    train_x：训练集的特征
    train_y：训练集的标签
    val_x：验证集的特征
    regression：设置是否为回归任务，True表示进行回归任务，False表示进行分类任务
    mode：
    needs_proba：设置是否输出预测概率，而不是结果
    save_dir：
    metric：设置模型性能的评价指标
    n_folds：交叉验证的折数
    stratified：是否进行分层抽样
    shuffle：设置是否打乱样本顺序
    random_state：设置随机种子
'''

S_train_1, S_test_1 = stacking(models,                   # list of models
                               train_x, train_y, val_x,   # data
                               regression=False,           # classification task (if you need
                                                           #     regression - set to True)
                               mode='oof_pred',            # mode: oof for train set, fit on full
                                                           #     train and predict test set once
                               needs_proba=True,           # predict probabilities (if you need
                                                           #     class labels - set to False)
                               save_dir='./stacking',               # save result and log in current dir
                                                           #     (to disable saving - set to None)
                               metric=roc_auc_score_universal,            # metric: callable
                               n_folds=5,                  # number of folds
                               stratified=True,            # stratified split for folds
                               shuffle=True,               # shuffle the data
                               random_state=2019,             # ensure reproducibility
                               verbose=2)                  # print all info
# task:         [classification]
# n_classes:    [2]
# metric:       [roc_auc_score_universal]
# mode:         [oof_pred]
# n_models:     [4]

# model  0:     [SVC]
#     fold  0:  [0.99686559]
#     fold  1:  [0.99645416]
#     fold  2:  [0.99752121]
#     fold  3:  [0.99730695]
#     fold  4:  [0.99811955]
#     ----
#     MEAN:     [0.99725349] + [0.00056815]
#     FULL:     [0.99721956]

#     Fitting on full train set...

# model  1:     [XGBClassifier]
#     fold  0:  [0.99605510]
#     fold  1:  [0.99601446]
#     fold  2:  [0.99687670]
#     fold  3:  [0.99701364]
#     fold  4:  [0.99762741]
#     ----
#     MEAN:     [0.99671746] + [0.00061221]
#     FULL:     [0.99671388]

#     Fitting on full train set...

# model  2:     [LGBMClassifier]
#     fold  0:  [0.99507302]
#     fold  1:  [0.99554495]
#     fold  2:  [0.99667135]
#     fold  3:  [0.99653401]
#     fold  4:  [0.99742247]
#     ----
#     MEAN:     [0.99624916] + [0.00083848]
#     FULL:     [0.99621914]

#     Fitting on full train set...

# model  3:     [RandomForestClassifier]
#     fold  0:  [0.98903262]
#     fold  1:  [0.98957861]
#     fold  2:  [0.99265387]
#     fold  3:  [0.99076198]
#     fold  4:  [0.99236961]
#     ----
#     MEAN:     [0.99087934] + [0.00144816]
#     FULL:     [0.99082188]

#     Fitting on full train set...

# Result was saved to [stacking/[2019.06.01].[19.36.50].237580.5d4e19.npy]

n_classes = 2
print('We have %d classes and %d models so in resulting arrays \
we expect to see %d columns.' % (n_classes, len(models), n_classes * len(models)))
print('S_train_1 shape:', S_train_1.shape)
print('S_test_1 shape: ', S_test_1.shape)

# We have 2 classes and 4 models so in resulting arrays we expect to see 8 columns.
# S_train_1 shape: (19752, 8)
# S_test_1 shape:  (4938, 8)

from glob import glob

names = sorted(glob(pathname='./stacking/*.npy'))
npy_1_name = names[0]  # for later use

print('Arrays:')
for name in names:
    print(name)

names = sorted(glob(pathname='./stacking/*.log.txt'))
log_1_name = names[0]  # for later use

print('\nLogs:')
for name in names:
    print(name)

# Arrays:
# ./stacking/[2019.05.26].[21.05.03].971760.57a7cf.npy

# Logs:
# ./stacking/[2019.05.26].[21.05.03].971760.57a7cf.log.txt


print("Let's open this log: %s" % log_1_name)
with open(log_1_name) as f:
    lines = f.readlines()

print("Let's look what models did we build in those session.\n")
for line in lines:
    if re.search(r'model [0-9]+', line):
        print(line)

# Let's open this log: ./stacking/[2019.05.26].[21.05.03].971760.57a7cf.log.txt
# Let's look what models did we build in those session.

print('We have %d classes and %d models TOTAL so in resulting arrays \
we expect to see %d columns.' % (n_classes, len(models), n_classes * (len(models))))

# Create empty arrays
S_train_all = np.zeros((train_x.shape[0], 0))
S_test_all = np.zeros((val_x.shape[0], 0))

# Load results
for name in sorted(glob('./stacking/*.npy')):
    print('Loading: %s' % name)
    S = np.load(name, allow_pickle=True)
    S_train_all = np.c_[S_train_all, S[0]]
    S_test_all = np.c_[S_test_all, S[1]]

print('\nS_train_all shape:', S_train_all.shape)
print('S_test_all shape: ', S_test_all.shape)

# S_train_all shape: (19752, 8)
# S_test_all shape:  (4938, 8)


# Initialize 2nd level model
model = lgb.LGBMClassifier(
    learning_rate=0.05,  # 重点，最后调
    colsample_bytree=0.8,  # 修改为最优参数=0.8
    subsample=0.7,  # 修改为最优参数=0.7
    num_leaves=90,  # 重点，修改为最优参数=90
    min_child_weight=4,  # 防止过拟合
    #     min_sum_hessian_in_leaf=1e-5,   # 调参后，分数下降了
    min_data_in_leaf=4,
    gamma=0,
    n_estimators=63,  # 以上参数需要调参
    max_depth=-1,
    is_unbalance=True,
    boosting_type='gbdt',
    objective='binary',
    random_state=2019,
    n_jobs=2,

)

print('-----------Cross validation Start...-----------')
lgb_param = model.get_params()
lgtrain = lgb.Dataset(S_train_all, train_y)
lgtest = lgb.Dataset(S_test_all)
cv_result = lgb.cv(lgb_param, lgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=5, \
                   stratified=True, metrics='auc', early_stopping_rounds=500)
model.set_params(
    n_estimators=len(cv_result['auc-mean']))  # cv_result是字典类型，keys为['auc_mean', 'auc_stdv']，values的取值个数为模型的最优迭代次数
print('the ideal number of trees is : {}'.format(str(model.get_params()['n_estimators'])))

# stacking模型的最优决策树数目为
# the ideal number of trees is : 13


# Fit 2nd level model
model = model.fit(S_train_all, train_y)

# Predict
y_pred = model.predict_proba(S_test_all)[:,1]

# Final prediction score
print('Final prediction score: %.8f' % roc_auc_score(val_y, y_pred))