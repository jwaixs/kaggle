import numpy as np

from load_data import train_df, test_df

print('Size of training set: {} rows and {} columns'.format(*train_df.shape))
print('Size of test set: {} rows and {} columns'.format(*test_df.shape))

print('Head of training set:')
print(train_df.head())

y_train = train_df['y'].values
print('Distribution in training of target value')
print('Min: {}, max: {}, mean: {}, std: {}'.format(
    min(y_train),
    max(y_train),
    y_train.mean(),
    y_train.std()
))

print('Study the features.')
feature_names = [c for c in train_df.columns if 'X' in c]
print('Number of features: {}'.format(len(feature_names)))
print('Feature types:')
print(train_df[feature_names].dtypes.value_counts())
categorical_features = list()
for fname in feature_names:
    typ = train_df[fname].dtype
    feature = train_df[fname].values
    if typ == np.int64:
        print('Feature: {} (int64), min: {}, max: {}, mean: {}, std: {}'.format(
            fname,
            min(feature),
            max(feature),
            feature.mean(),
            feature.std()
        ))
    else:
        print('Feature: {} (categorical), num: {}'.format(
            fname,
            len(np.unique(feature))
        ))
        categorical_features.append(fname)
