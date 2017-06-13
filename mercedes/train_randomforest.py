import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

from load_data import X_train, y_train, X_test, submission
from data_process import LogExpPipeline, _name_estimators

y_train_log = np.log1p(y_train)

def find_semi_optimal_params_ridge(X, y):
    print('Search for (semi-)best parameters for RandomForestRegressor using cross validation.')

    # Default RandomForestRegressor parameters
    rf_params = {
        'n_estimators' : 10,
        'criterion' : 'mse',
        'max_features' : 'auto',
        'max_depth' : None,
        'min_samples_split' : 2,
        'min_samples_leaf' : 1,
        'min_weight_fraction_leaf' : 0,
        'max_leaf_nodes' : None,
        'min_impurity_split' : 0.0000001,
        'bootstrap' : True,
        'oob_score' : False,
        'n_jobs' : 1,
        'random_state' : 0,
        'verbose' : 0,
        'warm_start' : 0
    }
    rf_pipe = RandomForestRegressor(**rf_params)

    print('Find (semi-)optimal n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf.')
    rf_search = {
        'n_estimators' : [10, 50, 100, 500],
        'max_features' : ['auto', 'sqrt', 'log2'],
        'max_depth' : [None, 3, 5, 7],
        'min_samples_split' : [0.5, 2, 3, 4],
        'min_samples_leaf' : [0.5, 2, 3, 4]
    }
    gs_clf = GridSearchCV(rf_pipe, rf_search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_n_estimators = gs_clf.best_estimator_.n_estimators
    best_max_features = gs_clf.best_estimator_.max_features
    best_max_depth = gs_clf.best_estimator_.max_depth
    best_min_samples_split = gs_clf.best_estimator_.min_samples_split
    best_min_samples_leaf = gs_clf.best_estimator_.min_samples_leaf

    print('Best score: {} (n_estimators = {}, max_features = {}, max_depth = {}, min_samples_split = {}, min_samples_leaf = {})'.format(
        gs_clf.best_score_, best_n_estimators, best_max_features, best_max_depth, best_min_samples_split, best_min_samples_leaf
    ))

    rf_params['n_estimators'] = best_n_estimators
    rf_params['max_features'] = best_max_features
    rf_params['min_samples_split'] = best_min_samples_split
    rf_params['min_samples_leaf'] = best_min_samples_leaf

    rf_pipe = RandomForestRegressor(**rf_params)

    print('Test some random seeds')
    rf_search = {
        'random_state' : range(100)
    }
    gs_clf = GridSearchCV(rf_pipe, rf_search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_random_state = gs_clf.best_estimator_.random_state
    print('Best score: {} (random_state = {})'.format(
        gs_clf.best_score_, best_random_state
    ))

    rf_params['random_state'] = best_random_state

    return rf_params

rf_params = find_semi_optimal_params_ridge(X_train, y_train_log)
print('(semi-)Best parameters for RandomForestRegressor:')
print(rf_params)
rf_pipe = RandomForestRegressor(**rf_params)

rf_pipe.fit(X_train, y_train_log)
preds = np.expm1(rf_pipe.predict(X_test))

submission['y'] = preds
submission.to_csv('submission_randomforest.csv', index = False)
