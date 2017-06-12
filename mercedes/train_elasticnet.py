import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV

from load_data import X_train, y_train, X_test, submission
from data_process import LogExpPipeline, _name_estimators

y_train_log = np.log1p(y_train)

def find_semi_optimal_params_elasticnet(X, y):
    print('Search for (semi-)best parameters for ElasticNet using cross validation.')

    # Default ElasticNet parameters
    en_params = {
        'alpha' : 1.0,
        'l1_ratio' : 0.5,
        'fit_intercept' : True,
        'normalize' : False,
        'precompute' : False,
        'max_iter' : 1000,
        'tol' : 0.0001,
        'warm_start' : False,
        'positive' : False,
        'selection' : 'cyclic',
        'random_state' : 0
    }
    en_pipe = ElasticNet(**en_params)

    print('Find (semi-)optimal alpha and max_iter.')
    en_search = {
        'alpha' : [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
        'max_iter' : [10, 50, 100, 500, 1000, 1500, 2000]
    }
    gs_clf = GridSearchCV(en_pipe, en_search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_alpha = gs_clf.best_estimator_.alpha
    best_max_iter = gs_clf.best_estimator_.max_iter
    print('Best score: {} (alpha = {}, max_iter = {})'.format(
        gs_clf.best_score_, best_alpha, best_max_iter
    ))

    en_params['alpha'] = best_alpha
    en_params['max_iter'] = best_max_iter
    en_pipe = ElasticNet(**en_params)

    print('Find (semi-)optimal l1_ratio.')
    en_search = {
        'l1_ratio' : np.arange(0, 1.1, 0.1)
    }
    gs_clf = GridSearchCV(en_pipe, en_search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_l1_ratio = gs_clf.best_estimator_.l1_ratio
    print('Best score: {} (l1_ratio = {})'.format(
        gs_clf.best_score_, best_l1_ratio
    ))

    en_params['l1_ratio'] = best_l1_ratio
    en_pipe = ElasticNet(**en_params)

    print('Find (semi-)optimal fit_intercept, normalize, and warm_start.')
    en_search = {
        'fit_intercept' : [True, False],
        'normalize' : [True, False],
        'warm_start' : [True, False],
    }
    gs_clf = GridSearchCV(en_pipe, en_search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_fit_intercept = gs_clf.best_estimator_.fit_intercept
    best_normalize = gs_clf.best_estimator_.normalize
    best_warm_start = gs_clf.best_estimator_.warm_start
    print('Best score: {} (fit_intercept = {}, normalize = {}, warm_start = {})'.format(
        gs_clf.best_score_, best_fit_intercept, best_normalize, best_warm_start
    ))

    en_params['fit_intercept'] = best_fit_intercept
    en_params['normalize'] = best_normalize
    en_params['warm_start'] = best_warm_start
    en_pipe = ElasticNet(**en_params)


    print('Find (semi-)optimal tolerance (tol), and selection.')
    en_search = {
        'tol' : [0.1, 0.01, 0.001, 0.0001, 0.00001],
        'selection' : ['cyclic', 'random']
    }
    gs_clf = GridSearchCV(en_pipe, en_search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_tol = gs_clf.best_estimator_.tol
    best_selection = gs_clf.best_estimator_.selection
    print('Best score: {} (tol = {}, selection = {})'.format(
        gs_clf.best_score_, best_tol, best_selection
    ))

    en_params['tol'] = best_tol
    en_params['selection'] = best_selection

    en_pipe = ElasticNet(**en_params)

    print('Test some random seeds')
    en_search = {
        'random_state' : range(100)
    }
    gs_clf = GridSearchCV(en_pipe, en_search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_random_state = gs_clf.best_estimator_.random_state
    print('Best score: {} (random_state = {})'.format(
        gs_clf.best_score_, best_random_state
    ))

    en_params['random_state'] = best_random_state

    return en_params

en_params = find_semi_optimal_params_elasticnet(X_train, y_train_log)
print('(semi-)Best parameters for ElasticNet:')
print(en_params)
en_pipe = ElasticNet(**en_params)

en_pipe.fit(X_train, y_train_log)
preds = np.expm1(en_pipe.predict(X_test))

submission['y'] = preds
submission.to_csv('submission_elasticnet.csv', index = False)
