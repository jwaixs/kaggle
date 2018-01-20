#!/usr/bin/env python

import gensim

document = [
    gensim.models.doc2vec.LabeledSentence(
        'Dit is de eerste zin'.split(), tags = ['SENT_0']
    ),
    gensim.models.doc2vec.LabeledSentence(
        'Dit is de tweede zin'.split(), tags = ['SENT_1']
    )
]

doc2vec_model = gensim.models.Doc2Vec(
    alpha = .025, min_alpha = .025, min_count = 1
)
doc2vec_model.build_vocab(document)
doc2vec_model.train(document, total_examples = len(document), epochs = 10)

doc2vec_model.save('my-first-model.doc2vec')
doc2vec_model = gensim.models.Doc2Vec.load('my-first-model.doc2vec')
