import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from config import *
from functions import *
import json

import warnings
warnings.filterwarnings('ignore')

'''
1. reading
2. training
'''

# reading
print('readings...')
X = np.load(path_data+file_train_X)
y = pd.read_csv(path_data+file_train_y)
with open(file_grid_params, 'r') as f:
    params = json.load(f)

# training
print('training...')
clf_lr = LogisticRegression(**params)
clf = OneVsRestClassifier(clf_lr)
clf.fit(X, y)
joblib.dump(clf, file_lr_ovr_model)