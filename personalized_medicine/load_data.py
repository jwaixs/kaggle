import pandas as pd
import os

data_dir = '/data/kaggle/personalized_medicine/'

test_text_file = os.path.join(data_dir, 'test_text')
test_variants_file = os.path.join(data_dir, 'test_variants')
training_text_file = os.path.join(data_dir, 'training_text')
training_variants_file = os.path.join(data_dir, 'training_variants')

train_variants_df = pd.read_csv(training_variants_file)
train_text_df = pd.read_csv(
    training_text_file,
    sep = '\|\|',
    engine = 'python',
    header = None,
    skiprows = 1,
    names = ['ID', 'Text']
)
test_variants_df = pd.read_csv(test_variants_file)
test_text_df = pd.read_csv(
    test_text_file,
    sep = '\|\|',
    engine = 'python',
    header = None,
    skiprows = 1,
    names = ['ID', 'Text']
)
