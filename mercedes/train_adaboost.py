import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

from load_data import X_train, y_train, X_test, submission
from data_process import LogExpPipeline, _name_estimators

y_train_log = np.log1p(y_train)

regressor = AdaBoostRegressor

def find_semi_optimal_params_adaboost(X, y):
    print('Search for (semi-)best parameters for AdaBoost Regressor using cross validation.')

    # Default AdaBoost parameters
    params = {
        'base_estimator' : None,
        'n_estimators' : 50,
        'learning_rate' : 1,
        'loss' : 'linear',
        'random_state' : 0
    }
    pipe = regressor(**params)

    print('Find (semi-)optimal n_estimators, learning rate, and loss.')
    search = {
        'n_estimators' : [10, 50, 100, 200, 500],
        'learning_rate' : [1, 0.1, 0.05, 0.01, 0.005],
        'loss' : ['linear', 'square', 'exponential']
    }
    gs_clf = GridSearchCV(pipe, search, cv = 10, scoring = 'r2', n_jobs = -1)
    gs_clf.fit(X, y)

    best_n_estimators = gs_clf.best_estimator_.n_estimators
    best_learning_rate = gs_clf.best_estimator_.learning_rate
    best_loss = gs_clf.best_estimator_.loss

    print('Best score: {} (n_estimators = {}, learning_rate = {}, loss = {})'.format(
        gs_clf.best_score_, best_n_estimators, best_learning_rate, best_loss
    ))

    params['n_estimators'] = best_n_estimators
    params['learning_rate'] = best_learning_rate
    params['loss'] = best_loss

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

params = find_semi_optimal_params_adaboost(X_train, y_train_log)
print('(semi-)Best parameters for AdaBoost Regressor:')
print(params)
pipe = regressor(**params)

pipe.fit(X_train, y_train_log)
preds = np.expm1(pipe.predict(X_test))

submission['y'] = preds
submission.to_csv('submission_adaboost.csv', index = False)
