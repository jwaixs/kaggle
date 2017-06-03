import pandas as pd
import numpy as np

train_df = pd.read_csv('/home/noud/data/kaggle/mercedes/train.csv')
test_df = pd.read_csv('/home/noud/data/kaggle/mercedes/test.csv')
submission = pd.read_csv('/home/noud/data/kaggle/mercedes/sample_submission.csv')

X_train = train_df.drop(['y'], axis = 1)
y_train = train_df['y']
X_test = test_df

# Drop constant features
constant_features = [
    'X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347'
]
X_train = X_train.drop(constant_features, axis = 1)
X_test = X_test.drop(constant_features, axis = 1)

# Convert categorical features to integers
categorical_features = [
    'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'
]
for cfname in categorical_features:
    unique_features = np.unique(np.append(train_df[cfname].values, test_df[cfname].values))
    cf_conversion = dict(zip(unique_features, range(len(unique_features))))
    X_train[cfname] = X_train[cfname].apply(lambda x : cf_conversion[x])
    X_test[cfname] = X_test[cfname].apply(lambda x : cf_conversion[x])
