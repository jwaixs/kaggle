import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV

from load_data import X_train, y_train, X_test, submission
from data_process import LogExpPipeline, _name_estimators

y_train_log = np.log1p(y_train)

def find_semi_optimal_params_lasso(X, y):
    print('Search for (semi-)best parameters for Lasso using cross validation.')

    # Default Lasso parameters
    la_params = {
        'alpha' : 1.0,
        'fit_intercept' : True,
        'normalize' : False,
        'copy_X' : True,
        'precompute' : False,
        'max_iter' : 1000,
        'tol' : 0.0001,
        'warm_start' : False,
        'positive' : False,
        'selection' : 'cyclic',
        'random_state' : 0

    }
    la_pipe = Lasso(**la_params)

    print('Find (semi-)optimal alpha and max_iter.')
    la_search = {
        'alpha' : [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
        'max_iter' : [10, 50, 100, 500, 1000, 1500, 2000]
    }
    gs_clf = GridSearchCV(la_pipe, la_search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_alpha = gs_clf.best_estimator_.alpha
    best_max_iter = gs_clf.best_estimator_.max_iter
    print('Best score: {} (alpha = {}, max_iter = {})'.format(
        gs_clf.best_score_, best_alpha, best_max_iter
    ))

    best_alpha = 0.001
    best_max_iter = 500

    la_params['alpha'] = best_alpha
    la_params['max_iter'] = best_max_iter
    la_pipe = Lasso(**la_params)


    print('Find (semi-)optimal fit_intercept, normalize, and warm_start.')
    la_search = {
        'fit_intercept' : [True, False],
        'normalize' : [True, False],
        'warm_start' : [True, False],
    }
    gs_clf = GridSearchCV(la_pipe, la_search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_fit_intercept = gs_clf.best_estimator_.fit_intercept
    best_normalize = gs_clf.best_estimator_.normalize
    best_warm_start = gs_clf.best_estimator_.warm_start
    print('Best score: {} (fit_intercept = {}, normalize = {}, warm_start = {})'.format(
        gs_clf.best_score_, best_fit_intercept, best_normalize, best_warm_start
    ))

    best_fit_intercept = True
    best_normalize = False
    best_warm_start = True

    la_params['fit_intercept'] = best_fit_intercept
    la_params['normalize'] = best_normalize
    la_params['warm_start'] = best_warm_start
    la_pipe = Lasso(**la_params)


    print('Find (semi-)optimal tolerance (tol), and selection.')
    la_search = {
        'tol' : [0.1, 0.01, 0.001, 0.0001, 0.00001],
        'selection' : ['cyclic', 'random']
    }
    gs_clf = GridSearchCV(la_pipe, la_search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_tol = gs_clf.best_estimator_.tol
    best_selection = gs_clf.best_estimator_.selection
    print('Best score: {} (tol = {}, selection = {})'.format(
        gs_clf.best_score_, best_tol, best_selection
    ))

    la_params['tol'] = best_tol
    la_params['selection'] = best_selection

    la_pipe = Lasso(**la_params)

    print('Test some random seeds')
    la_search = {
        'random_state' : range(1000)
    }
    gs_clf = GridSearchCV(la_pipe, la_search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_random_state = gs_clf.best_estimator_.random_state
    print('Best score: {} (random_state = {})'.format(
        gs_clf.best_score_, best_random_state
    ))

    la_params['random_state'] = best_random_state

    return la_params

la_params = find_semi_optimal_params_lasso(X_train, y_train_log)
#la_params = {'warm_start': True, 'selection': 'random', 'max_iter': 500, 'precompute': False, 'copy_X': True, 'alpha': 0.001, 'normalize': False, 'fit_intercept': True, 'positive': False, 'random_state': 102, 'tol': 0.1}
print('(semi-)Best parameters for Lasso:')
print(la_params)
la_pipe = Lasso(**la_params)

la_pipe.fit(X_train, y_train_log)
preds = np.expm1(la_pipe.predict(X_test))

submission['y'] = preds
submission.to_csv('submission_lasso.csv', index = False)
