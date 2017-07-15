import numpy as np
import pandas as pd

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators

from load_data import X_train, y_train, X_test, submission

class LogExpPipeline(Pipeline):
    def fit(self, X, y):
        super(LogExpPipeline, self).fit(X, np.log1p(y))

    def predict(self, X):
        return np.expm1(super(LogExpPipeline, self).predict(X))

# SGD Regression
sgdr_params = {
    'alpha' : 0.001,
    'l1_ratio' : 0.1
}
sgdr_pipe = LogExpPipeline(_name_estimators([
    SGDRegressor(**sgdr_params)
]))
results = cross_val_score(sgdr_pipe, X_train, y_train, cv = 10, scoring = 'r2', n_jobs = -1)
print('SGD Regression cross validation score: {}+/-{}'.format(results.mean(), results.std()))
