import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV

from load_data import X_train, y_train, X_test, submission
from data_process import LogExpPipeline, _name_estimators

y_train_log = np.log1p(y_train)

def find_semi_optimal_params_ridge(X, y):
    print('Search for (semi-)best parameters for Ridge using cross validation.')

    # Default Ridge parameters
    ri_params = {
        'alpha' : 1.0,
        'fit_intercept' : True,
        'max_iter' : 1000,
        'normalize' : False,
        'solver' : 'auto',
        'tol' : 0.0001,
        'random_state' : 0
    }
    ri_pipe = Ridge(**ri_params)

    print('Find (semi-)optimal alpha, solver, and max_iter.')
    ri_search = {
        'alpha' : [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
        'max_iter' : [10, 50, 100, 500, 1000, 1500, 2000],
        'solver' : ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
    }
    gs_clf = GridSearchCV(ri_pipe, ri_search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_alpha = gs_clf.best_estimator_.alpha
    best_max_iter = gs_clf.best_estimator_.max_iter
    best_solver = gs_clf.best_estimator_.solver
    print('Best score: {} (alpha = {}, max_iter = {}, solver = {})'.format(
        gs_clf.best_score_, best_alpha, best_max_iter, best_solver
    ))

    ri_params['alpha'] = best_alpha
    ri_params['max_iter'] = best_max_iter
    ri_params['solver'] = best_solver
    ri_pipe = Ridge(**ri_params)

    print('Find (semi-)optimal fit_intercept, and normalize.')
    ri_search = {
        'fit_intercept' : [True, False],
        'normalize' : [True, False],
    }
    gs_clf = GridSearchCV(ri_pipe, ri_search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_fit_intercept = gs_clf.best_estimator_.fit_intercept
    best_normalize = gs_clf.best_estimator_.normalize
    print('Best score: {} (fit_intercept = {}, normalize = {})'.format(
        gs_clf.best_score_, best_fit_intercept, best_normalize
    ))

    ri_params['fit_intercept'] = best_fit_intercept
    ri_params['normalize'] = best_normalize
    ri_pipe = Ridge(**ri_params)


    print('Find (semi-)optimal tolerance (tol).')
    ri_search = {
        'tol' : [0.1, 0.01, 0.001, 0.0001, 0.00001],
    }
    gs_clf = GridSearchCV(ri_pipe, ri_search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_tol = gs_clf.best_estimator_.tol
    print('Best score: {} (tol = {})'.format(
        gs_clf.best_score_, best_tol
    ))

    ri_params['tol'] = best_tol

    ri_pipe = Ridge(**ri_params)

    print('Test some random seeds')
    ri_search = {
        'random_state' : range(100)
    }
    gs_clf = GridSearchCV(ri_pipe, ri_search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_random_state = gs_clf.best_estimator_.random_state
    print('Best score: {} (random_state = {})'.format(
        gs_clf.best_score_, best_random_state
    ))

    ri_params['random_state'] = best_random_state

    return ri_params

ri_params = find_semi_optimal_params_ridge(X_train, y_train_log)
print('(semi-)Best parameters for Ridge:')
print(ri_params)
ri_pipe = Ridge(**ri_params)

ri_pipe.fit(X_train, y_train_log)
preds = np.expm1(ri_pipe.predict(X_test))

submission['y'] = preds
submission.to_csv('submission_ridge.csv', index = False)
