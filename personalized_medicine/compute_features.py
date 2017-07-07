import os

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from load_data import train_text_df, test_text_df

def compute_tfidf_vec(return_test = False):
    tfidf = TfidfVectorizer(
        min_df = 5,
        max_features = 16000,
        strip_accents = 'unicode',
        lowercase = True,
        analyzer = 'word',
        token_pattern = r'\w+',
        ngram_range = (1, 3),
        use_idf = True,
        smooth_idf = True,
        sublinear_tf = True,
        stop_words = 'english'
    )
    tfidf.fit(train_text_df['Text'])

    train_tfidf_vec = tfidf.transform(train_text_df['Text'])

    if return_test:
        test_tfidf_vec = tfidf.transform(test_text_df['Text'])
        return train_tfidf_vec, test_tfidf_vec
    else:
        return train_tfidf_vec

def save_tfidf_vec():
    train_text_tfidf_vec, test_tfidf_vec = compute_tfidf_vec(return_test = True)
    np.save('train_tfidf_vec', train_tfidf_vec)
    np.save('test_tfidf_vec', test_tfidf_vec)
    return train_tfidf_vec, test_tfidf_vec

def load_tfidf_vec():
    if not all(map(os.path.isfile, 
            ['train_tfidf_vec.npy', 'test_tfidf_vec.npy'])):
        return save_tfidf_vec()
    return np.load('train_tfidf_vec.npy'), np.load('test_tfidf_vec.npy')
