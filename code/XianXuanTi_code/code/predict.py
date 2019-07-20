import numpy as np
import pandas as pd
from sklearn.externals import joblib
from config import *
from functions import *
import json
import warnings
warnings.filterwarnings('ignore')

'''
1. reading file
2. predicting
3. post-processing
4. modifing
'''

# reading file
print('reading file...')
clf = joblib.load(file_lr_ovr_model) # 模型
df_train_y = pd.read_csv(path_data+file_train_y)
lst_labels = df_train_y.columns # 标签列
df_test_preprocessing = pd.read_csv(path_data+file_test_preprocessing) # 测试集ids
np_test_X = np.load(path_data+file_test_X) # 测试集特征数据
with open(path_data+file_dic_labels, 'r') as f: 
    dic_labels = json.load(f) # 标签映射

# predicting
print('predicting...')
pred_y_prob = clf.predict_proba(np_test_X)

# post-processing
print('post-processing...')
pred_y_prob_mod = (pred_y_prob >= threshold).astype(int)
pred_y = pd.DataFrame(data=pred_y_prob_mod, columns=lst_labels).astype(int)
pred_y['ids'] = df_test_preprocessing['ids']
pred_y['label'] = ''
for i in lst_labels:
    pred_y['label'] = pred_y['label'].astype(str) + ':' + pred_y[i].astype(str)
dic_labels_rev = {v:k for k,v in dic_labels.items()}
pred_y['accusation'] = pred_y['label'].apply(get_accusation, args=(lst_labels, dic_labels_rev))
result = pred_y[['ids', 'accusation']]
result.replace({'':np.nan}, inplace=True)
result = result.fillna(method='bfill')
result.to_csv(path_data+file_result, index=None)

# modifing
print('modifing...')
with open(path_data+file_result, 'r+') as f:
    content = f.read()        
    f.seek(0, 0)
    f.write('test_sample\n' + content)