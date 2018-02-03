#!/usr/bin/env python

import gensim

import numpy as np

from train import tokenizer

class DocToVec(object):
    def __init__(self, model_path):
        self.model = gensim.models.Doc2Vec.load(model_path)

    def __call__(self, inp):
        return [self.model[w] for s in tokenizer(inp) for w in s]

    def toNumpy(self, inp):
        return np.expand_dims(np.array(self(inp)), 0)
