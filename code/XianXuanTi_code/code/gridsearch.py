import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
import json
from config import *
from functions import *
import warnings
warnings.filterwarnings('ignore')

'''
1. data reading
2. grid searching
3. persistence
'''

# data reading
print('data reading...')
X = np.load(path_data+file_train_X)
y = pd.read_csv(path_data+file_train_y)
ss = ShuffleSplit(n_splits=10)
gen_cv = ss.split(X, y)
cv_0 = gen_cv.__next__()[1]
sub_X = X[cv_0]
sub_y = y.loc[cv_0, :]

# grid searching
print('grid searching...')
param_grid = {'estimator__C': [0.01, 0.1, 1, 10], \
              'estimator__penalty': ['l1', 'l2']}
clf_lr = LogisticRegression()
grid = GridSearchCV(OneVsRestClassifier(clf_lr), param_grid=param_grid, scoring=make_scorer(f1_avg_scorer), verbose=2).fit(sub_X, sub_y)

# persistence
print('persistence...')
best_params = {}
for k,v in grid.best_params_.items():
    best_params[k.strip('estimator__')] = v
with open(file_grid_params, 'w') as f:
    json.dump(best_params, f)