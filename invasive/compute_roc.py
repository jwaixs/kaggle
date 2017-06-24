import sys

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

f0 = pd.read_csv(sys.argv[1])
f1 = pd.read_csv(sys.argv[2])
f2 = pd.read_csv(sys.argv[3])
f3 = pd.read_csv(sys.argv[4])
train_results = pd.concat([f0, f1, f2, f3], axis = 0)
train_results = train_results.fillna(1.0)

labels = pd.read_csv('./train_labels.csv')

y_true = list()
y_score = list()
for i, r in labels.iterrows():
    tr = train_results[train_results['name'] == r.values[0]]
    y_true.append(r.values[1])
    y_score.append(tr.values[0][1])

fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)
print('Area under the curve: {}'.format(auc))
