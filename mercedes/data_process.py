import numpy as np

from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.base import BaseEstimator, TransformerMixin

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
