#!/usr/bin/env python

# Mostly copied from https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb

import os
import re
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

data_dir = '/media/noud/data/noud/toxic/'

df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
df_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

print(df_train.head())

print(df_train['comment_text'][0])
train_length = df_train.comment_text.str.len()

print(train_length.mean(), train_length.std(), train_length.max())

#train_length.hist()
#plt.show()

# Create a 'none' label, which contains the data of comments that have no label
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df_train['none'] = 1 - df_train[classes].max(axis = 1)

print(df_train.describe())

# Replace empty comments with 'unknown'
df_train['comment_text'].fillna('unknown', inplace = True)
df_test['comment_text'].fillna('unknown', inplace = True)

# Build the model
tfidf_vec = TfidfVectorizer()
train_term_doc = tfidf_vec.fit_transform(df_train['comment_text'])
test_term_doc = tfidf_vec.transform(df_test['comment_text'])

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from scipy import sparse

class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C = 1.0, dual = False, n_jobs = 1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        y = y.values
        x, y = check_X_y(x, y, accept_sparse = True)

        def pr(s, y_i, y):
            p = x[y == y_i].sum(0)
            return (p+1) / ((y == y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(
            C = self.C,
            dual = self.dual,
            n_jobs = self.n_jobs
        ).fit(x_nb, y)

        return self

test_preds = np.zeros((len(df_test), len(classes)))

bag_of_classifiers = dict()
for i, c in enumerate(classes):
    print('Fitting {}'.format(c))
    model = NbSvmClassifier(C = 4, dual = True, n_jobs = -1)
    model.fit(train_term_doc, df_train[c])
    bag_of_classifiers[c] = model
    test_preds[:,i] = model.predict_proba(test_term_doc)[:,1]

df_new_submission = pd.concat([
    pd.DataFrame({'id' : df_submission['id']}),
    pd.DataFrame(test_preds, columns = classes)
], axis = 1)
df_new_submission.to_csv('submission-nb-svm.csv', index = False)
