import pickle

from gensim.models.word2vec import Word2Vec
from create_sentences import get_sentences

sentences = get_sentences()
model = Word2Vec(sentences, workers = 5, min_count = 2)
model.save('word2vec.model')
