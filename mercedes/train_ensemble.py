import sys

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from load_data import X_train, y_train, X_test, submission

# Initial idea of the ensembling is borrowed/stolen from https://www.kaggle.com/eikedehling/stack-of-svm-elasticnet-xgboost-rf-0-55

class AddColumns(BaseEstimator, TransformerMixin):
    def __init__(self, transform_ = None):
        self.transform_ = transform_

    def fit(self, X, y = None):
        self.transform_.fit(X, y)
        return self

    def transform(self, X, y = None):
        xform_data = self.transform_.transform(X, y)
        return np.append(X, xform_data, axis = 1)

class LogExpPipeline(Pipeline):
    def fit(self, X, y):
        super(LogExpPipeline, self).fit(X, np.log1p(y))

    def predict(self, X):
        return np.expm1(super(LogExpPipeline, self).predict(X))

# SVM regression
svm_params = {
    'kernel' : 'rbf',
    'C' : 1.0,
    'epsilon' : 0.05
}
svm_pipe = LogExpPipeline(_name_estimators([
    RobustScaler(),
    PCA(),
    SVR(**svm_params)
]))
results = cross_val_score(svm_pipe, X_train, y_train, cv = 10, scoring = 'r2', n_jobs = -1)
print('SVM cross validation score: {}+/-{}'.format(results.mean(), results.std()))

# ElasticNet regression
en_params = {
    'alpha' : 0.001,
    'l1_ratio' : 0.1
}
en_pipe = LogExpPipeline(_name_estimators([
    RobustScaler(),
    PCA(n_components = 125),
    ElasticNet(**en_params)
]))
results = cross_val_score(en_pipe, X_train, y_train, cv = 10, scoring = 'r2', n_jobs = -1)
print('ElasticNet cross validation score: {}+/-{}'.format(results.mean(), results.std()))

# Lasso
la_params = {
    'alpha' : 0.001,
    'max_iter' : 5000
}
la_pipe = LogExpPipeline(_name_estimators([
    RobustScaler(),
    PCA(),
    Lasso(**la_params)
]))
results = cross_val_score(la_pipe, X_train, y_train, cv = 10, scoring = 'r2', n_jobs = -1)
print('Lasso cross validation score: {}+/-{}'.format(results.mean(), results.std()))

# Random Forest
rf_params = {
    'n_estimators' : 250,
    'n_jobs' : 1,
    'min_samples_split' : 25,
    'min_samples_leaf' : 25,
    'max_depth' : 3
}
rf_pipe = RandomForestRegressor(**rf_params)
results = cross_val_score(rf_pipe, X_train, y_train, cv = 10, scoring = 'r2', n_jobs = -1)
print('Random forest cross validation score: {}+/-{}'.format(results.mean(), results.std()))

# Xgboost regression
xgb_params = {
    'n_estimators': 70,
    'learning_rate': 0.1,
    'max_depth': 8,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'nthread': 1,
    'silent': 1,
    'seed': 37
}
n_components = 12
xgb_pipe = LogExpPipeline(_name_estimators([
    AddColumns(transform_ = PCA(n_components = n_components)),
    AddColumns(transform_ = FastICA(n_components = n_components, max_iter = 500)),
    xgb.sklearn.XGBRegressor(**xgb_params)
]))
results = cross_val_score(xgb_pipe, X_train, y_train, cv = 10, scoring = 'r2', n_jobs = -1)
print('XGBoost regressor cross validation score: {}+/-{}'.format(results.mean(), results.std()))

# Ensemble classifiers

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(
            n_splits = self.n_splits,
            shuffle = True,
            random_state = 0
        ).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]

                print('Model {} fold {} score {}'.format(i, j, r2_score(y_holdout, y_pred)))

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_test[:, i] = S_test_i.mean(axis = 1)

        results = cross_val_score(self.stacker, S_train, y, cv = 10, scoring = 'r2')
        print('Stacker cross validation score: {}+/-{}'.format(results.mean(), results.std()))

        self.stacker.fit(S_train, y)
        result = self.stacker.predict(S_test)[:]

        return result

stacker = Ensemble(
    n_splits = 10,
    stacker = ElasticNet(l1_ratio = 0.1, alpha = 1.4),
    base_models = (svm_pipe, en_pipe, rf_pipe, la_pipe, xgb_pipe)
)
y_test = stacker.fit_predict(X_train, y_train, X_test)

submission['y'] = y_test
submission.to_csv('submission_ensemble.csv', index = False)
