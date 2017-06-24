import os
import shutil
import random

import pandas as pd

random.seed(37)

number_of_folds = 4

labels = pd.read_csv('./train_labels.csv')
size = labels.shape[0]

size_per_fold = size / number_of_folds

labeled_images = list()
for i, r in labels.iterrows():
    labeled_images.append(('{}.jpg'.format(r.values[0]), r.values[1]))
random.shuffle(labeled_images)

def kfold(full_list, n_folds):
    size_per_fold = len(full_list) / n_folds
    if size_per_fold < 0:
        raise 'Fold size is not positive'
    for i in range(n_folds - 1):
        yield full_list[i*size_per_fold:(i+1)*size_per_fold]
    yield full_list[(n_folds - 1)*size_per_fold:]

folds = list()
for fold in kfold(labeled_images, number_of_folds):
    folds.append(fold)

if not os.path.isdir('./dltrain'):
    os.makedirs('./dltrain')

for i in range(number_of_folds):
    fold_dir = '/data/kaggle/invasive/dltrain/fold_{}'.format(i)
    if not os.path.isdir(fold_dir):
        os.makedirs(fold_dir)
        os.makedirs('{}/train/0'.format(fold_dir))
        os.makedirs('{}/train/1'.format(fold_dir))
        os.makedirs('{}/valid/0'.format(fold_dir))
        os.makedirs('{}/valid/1'.format(fold_dir))
        os.makedirs('{}/test/test/'.format(fold_dir))
    for j in range(number_of_folds):
        output_dir = '{}/train'.format(fold_dir)
        if i == j:
            output_dir = '{}/valid'.format(fold_dir)
        for fname, label in folds[j]:
            print '{}/{}/{}'.format(output_dir, label, fname)
            os.symlink(
                '/data/kaggle/invasive/train/{}'.format(fname),
                '{}/{}/{}'.format(output_dir, label, fname)
            )
    for testimage in os.listdir('/data/kaggle/invasive/test/'):
        print testimage
        os.symlink(
            '/data/kaggle/invasive/test/{}'.format(testimage),
            '{}/test/test/{}'.format(fold_dir, testimage)
        )
