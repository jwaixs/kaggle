from sklearn.model_selection import KFold

class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)

def get_oof(clf, x, y, n_splits, random_state):
    kf = KFold(n_splits, shuffle = True, random_state = random_state)
    oof_train = np.zeros((len(x), ))

    for train_index, test_index in kf.split(x):
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]

        clf.fit(x_train, y_train)

        oof_train[test_index] = clf.predict(x_test)

    return oof_train.reshape(-1, 1)

import numpy as np
from sklearn import svm, datasets
from sklearn.multiclass import OneVsRestClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = OneVsRestClassifier(svm.SVC(kernel = 'linear', probability = True, random_state = 0))
for true, pred in zip(y, get_oof(clf, X, y, 10, 0)):
    print true, pred

