import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

from load_data import X_train, y_train, X_test, submission
from data_process import LogExpPipeline, _name_estimators

y_train_log = np.log1p(y_train)

regressor = BaggingRegressor

def find_semi_optimal_params_bagging(X, y):
    print('Search for (semi-)best parameters for Bagging Regressor using cross validation.')

    # Default Bagging Regressor parameters
    params = {
        'base_estimator' : None,
        'n_estimators' : 10,
        'max_samples' : 1.0,
        'max_features' : 1.0,
        'bootstrap' : True,
        'bootstrap_features' : False,
        'oob_score' : False,
        'warm_start' : False,
        'n_jobs' : 1,
        'random_state' : 0,
        'verbose' : 0
    }
    pipe = regressor(**params)

    print('Find (semi-)optimal n_estimators, max_samples, max_features, bootstrap, and bootstrap_features.')
    search = {
        'n_estimators' : [5, 10, 50, 100],
        'max_samples' : [1.0, 0.9, 0.8],
        'max_features' : [1.0, 0.9, 0.8],
        'bootstrap' : [True, False],
        'bootstrap_features' : [True, False],
    }
    gs_clf = GridSearchCV(pipe, search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_n_estimators = gs_clf.best_estimator_.n_estimators
    best_max_samples = gs_clf.best_estimator_.max_samples
    best_max_features = gs_clf.best_estimator_.max_features
    best_bootstrap = gs_clf.best_estimator_.bootstrap
    best_bootstrap_features = gs_clf.best_estimator_.bootstrap_features

    print('Best score: {} (n_estimators = {}, max_samples = {}, max_features = {}, bootstrap = {}, boostrap_features = {})'.format(
        gs_clf.best_score_, best_n_estimators, best_max_samples, best_max_features, best_bootstrap, best_bootstrap_features
    ))

    for key in search.keys():
        params[key] = eval('best_{}'.format(key))

    pipe = regressor(**params)

    print('Test some random seeds')
    search = {
        'random_state' : range(100)
    }
    gs_clf = GridSearchCV(pipe, search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_random_state = gs_clf.best_estimator_.random_state
    print('Best score: {} (random_state = {})'.format(
        gs_clf.best_score_, best_random_state
    ))

    params['random_state'] = best_random_state

    return params

params = find_semi_optimal_params_bagging(X_train, y_train_log)
print('(semi-)Best parameters for Bagging Regressor:')
print(params)
pipe = regressor(**params)

pipe.fit(X_train, y_train_log)
preds = np.expm1(pipe.predict(X_test))

submission['y'] = preds
submission.to_csv('submission_bagging.csv', index = False)
