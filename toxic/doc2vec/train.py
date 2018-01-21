#!/usr/bin/env python

import argparse
import gensim
import re

import pandas as pd

from tqdm import tqdm

def tokenizer(raw_doc):
    '''Tokenize a raw document string to list of list of words.
    
    Example:
        > s = "Nonsense?  kiss off, geek. what I said is true.  I'll have your account terminated."
        > tokenizer(s)
        [['nonsense'],
         ['kiss', 'off', 'geek'],
         ['what', 'i', 'said', 'is', 'true'],
         ['ill', 'have', 'your', 'account', 'terminated']]
        
    '''
    if len(raw_doc) > 10000:
        sentences = re.findall(r'(?ms)\s*(.*?(?:\.|\?|!))', raw_doc[:10000] + '.')
    else:
        sentences = re.findall(r'(?ms)\s*(.*?(?:\.|\?|!))', raw_doc + '.')
    sentences = map(lambda s : s.split(), sentences)
    remove_non_alpha = re.compile('[^a-zA-Z0-9]')
    for i, s in enumerate(sentences):
        sentences[i] = map(lambda w : remove_non_alpha.sub('', w), s)
        sentences[i] = map(lambda w : w.lower(), sentences[i])
    sentences = filter(lambda s : len(s[0]) != 0, sentences)
    return sentences

class LabeledLineSentence(object):
    def __init__(self, data_frames, label):
        self.data_frames = data_frames
        self.label = label

    def __len__(self):
        lengths = map(lambda df : df.shape[0], self.data_frames)
        return sum(lengths)

    def __iter__(self):
        i = 0
        for df in self.data_frames:
            for _, row in tqdm(df.iterrows()):
                sentences = tokenizer(str(row[self.label]))
                for sentence in sentences:
                    if len(sentence) <= 1:
                        continue

                    labeled_sentence = gensim.models.doc2vec.LabeledSentence(
                        sentence, tags = ['SENT_{}'.format(i)]
                    )
                    yield labeled_sentence
                    i += 1

def main(args):
    data_frames = map(lambda csv : pd.read_csv(csv), args.df)
    lls = LabeledLineSentence(data_frames, args.label)

    print('Initialize doc2vec model.')
    doc2vec_model = gensim.models.Doc2Vec(
        alpha = .025, min_alpha = .025, min_count = 1
    )

    print('Build vocabular.')
    doc2vec_model.build_vocab(lls)

    print('Start training doc2vec model.')
    doc2vec_model.train(
        lls, total_examples = len(lls), epochs = args.num_epochs
    )

    doc2vec_model.save(args.output)
    print('Done! Saving doc2vec model to {}.'.format(args.output))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Train a doc2vec classifier'
    )
    parser.add_argument(
        '--df', type = str, nargs = '+',
        help = 'List of pandas data frames'
    )
    parser.add_argument('--num-epochs', type = int, default = 10)
    parser.add_argument(
        '--label', type = str, help = 'Label to extract from data frames'
    )
    parser.add_argument(
        '--output', type = str
    )

    args = parser.parse_args()
    main(args)
