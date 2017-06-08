import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import r2_score
from load_data import X_train, y_train, X_test, submission

xgb_params = {
    'n_trees': 500, 
    'eta': 0.1,
    'max_depth': 8,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'seed': 37
}

def xgb_r2_score(preds, dmatrix):
    labels = dmatrix.get_label()
    return 'R2', r2_score(labels, preds)

print('Train cross validation on original training set')
dtrain = xgb.DMatrix(X_train, y_train)

num_round = 2000
xgb.cv(
    xgb_params,
    dtrain,
    num_round,
    nfold = 5,
    metrics = {'rmse'},
    seed = 0,
    feval = xgb_r2_score,
    maximize = True,
    verbose_eval = 10,
    early_stopping_rounds = 10
)

clf = xgb.train(
    xgb_params,
    dtrain,
    60
)

dtest = xgb.DMatrix(X_test)
preds = clf.predict(dtest)

submission['y'] = preds
submission.to_csv('submission_xgb.csv', index = False)
