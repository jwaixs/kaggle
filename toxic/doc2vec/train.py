#!/usr/bin/env python

import argparse
import gensim
import re

import pandas as pd

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
    sentences = re.findall(r'(?ms)\s*(.*?(?:\.|\?|!))', raw_doc)
    sentences = map(lambda s : s.split(), sentences)
    remove_non_alpha = re.compile('[^a-zA-Z0-9]')
    for i, s in enumerate(sentences):
        sentences[i] = map(lambda w : remove_non_alpha.sub('', w), s)
        sentences[i] = map(lambda w : w.lower(), sentences[i])
    return sentences

class LabeledLineSentence(object):
    def __init__(self, data_frames, label):
        self.data_frames = data_frames
        self.label = label

    def __iter__(self):
        i = 0
        for df in self.data_frames:
            for _, row in df.iterrows():
                sentences = tokenizer(row[self.label])
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
    for i in lls:
        print(i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Train a doc2vec classifier'
    )
    parser.add_argument(
        '--df', metavar = '...', type = str, nargs = '+',
        help = 'List of pandas data frames'
    )
    parser.add_argument(
        '--label', type = str, help = 'Label to extract from data frames'
    )
    parser.add_argument(
        '--output', type = str
    )

    args = parser.parse_args()
    main(args)
