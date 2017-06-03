import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from load_data import X_train, y_train, X_test, submission

lasso = Lasso(random_state = 0, max_iter = 5000)
alpha_space = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06]

n_folds = 5

for alpha in alpha_space:
    lasso.alpha = alpha
    cur_scores = cross_val_score(lasso, X_train, y_train, cv = n_folds, n_jobs = 4)
    print('{} {} {}'.format(alpha, np.mean(cur_scores), np.std(cur_scores)))

#lasso.alpha = 0.03
#lasso.fit(X_train, y_train)
#
#prediction = lasso.predict(X_test)
#
#submission['y'] = prediction
#submission.to_csv('./submission_lasso.csv', index = False)
