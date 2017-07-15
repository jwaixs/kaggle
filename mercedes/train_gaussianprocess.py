import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

from load_data import X_train, y_train, X_test, submission
from data_process import LogExpPipeline, _name_estimators

y_train_log = np.log1p(y_train)

regressor = GaussianProcessRegressor

def find_semi_optimal_params_gaussian_process(X, y):
    print('Search for (semi-)best parameters for Gaussian Process Regressor using cross validation.')

    # Default Bagging Regressor parameters
    params = {
        'kernel' : None,
        'alpha' : 1e-10,
        'optimizer' : 'fmin_l_bfgs_b',
        'n_restarts_optimizer': 0,
        'normalize_y' : False,
        'copy_X_train' : True,
        'random_state' : 0
    }

    pipe = regressor(**params)

    print('Find (semi-)optimal alpha, n_restarts_optimizer, and normalize_y.')
    search = {
        'alpha' : [1e-10, 1e-11],
        'n_restarts_optimizer' : [0],
        'normalize_y' : [True],
    }
    gs_clf = GridSearchCV(pipe, search, cv = 4, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_alpha = gs_clf.best_estimator_.alpha
    best_n_restarts_optimizer = gs_clf.best_estimator_.n_restarts_optimizer
    best_normalize_y = gs_clf.best_estimator_.normalize_y

    print('Best score: {} (alpha = {}, n_restarts_optimizer = {}, normalize_y = {})'.format(
        gs_clf.best_score_, best_alpha, best_n_restarts_optimizer, best_normalize_y
    ))

    for key in search.keys():
        params[key] = eval('best_{}'.format(key))

    pipe = regressor(**params)

    return params

    print('Test some random seeds')
    search = {
        'random_state' : range(10)
    }
    gs_clf = GridSearchCV(pipe, search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_random_state = gs_clf.best_estimator_.random_state
    print('Best score: {} (random_state = {})'.format(
        gs_clf.best_score_, best_random_state
    ))

    params['random_state'] = best_random_state

    return params

from sklearn.decomposition import PCA
pca = PCA(n_components = 8)
X_train_pca = pca.fit_transform(X_train)

params = find_semi_optimal_params_gaussian_process(X_train_pca, y_train_log)
print('(semi-)Best parameters for Gaussian Process Regressor:')
print(params)
#pipe = regressor(**params)
#
#pipe.fit(X_train, y_train_log)
#preds = np.expm1(pipe.predict(X_test))
#
#submission['y'] = preds
#submission.to_csv('submission_gaussian_process.csv', index = False)
