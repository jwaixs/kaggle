import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

from load_data import X_train, y_train, X_test, submission
from data_process import LogExpPipeline, _name_estimators

y_train_log = np.log1p(y_train)

regressor = GradientBoostingRegressor

def find_semi_optimal_params_gradient_boosting(X, y):
    print('Search for (semi-)best parameters for Gradient Boosting Regressor using cross validation.')

    # Default Bagging Regressor parameters
    params = {
        'loss' : 'ls',
        'learning_rate' : 0.1,
        'n_estimators' : 100,
        'subsample' : 1.0,
        'criterion' : 'friedman_mse',
        'min_samples_split' : 2,
        'min_samples_leaf' : 1,
        'min_weight_fraction_leaf' : 0.0,
        'max_depth' : 3,
        'min_impurity_split' : 1e-07,
        'init' : None,
        'random_state' : 0,
        'max_features' : None,
        'alpha' : 0.9,
        'verbose' : 0,
        'max_leaf_nodes' : None,
        'warm_start' : False,
        'presort' : 'auto'
    }
    pipe = regressor(**params)

    print('Find (semi-)optimal n_estimators, max_samples, max_features, bootstrap, and bootstrap_features.')
    search = {
        'learning_rate' : [1, 0.1, 0.01],
        'n_estimators' : [10, 100, 1000],
        'loss' : ['ls', 'lad', 'huber', 'quantile']
    }
    gs_clf = GridSearchCV(pipe, search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_learning_rate = gs_clf.best_estimator_.learning_rate
    best_n_estimators = gs_clf.best_estimator_.n_estimators
    best_loss = gs_clf.best_estimator_.loss

    print('Best score: {} (learning_rate = {}, n_estimators = {}, loss = {})'.format(
        gs_clf.best_score_, best_learning_rate, best_n_estimators, best_loss
    ))

    for key in search.keys():
        params[key] = eval('best_{}'.format(key))

    pipe = regressor(**params)

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

params = find_semi_optimal_params_gradient_boosting(X_train, y_train_log)
print('(semi-)Best parameters for Gradient Boosting Regressor:')
print(params)
pipe = regressor(**params)

pipe.fit(X_train, y_train_log)
preds = np.expm1(pipe.predict(X_test))

submission['y'] = preds
submission.to_csv('submission_gbr.csv', index = False)
