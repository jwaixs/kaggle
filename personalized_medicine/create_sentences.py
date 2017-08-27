import nltk

from tqdm import tqdm

from load_data import train_text_df

def text2sentences(s):
    tokens = nltk.word_tokenize(s.decode('utf-8')) 
    tokens = [w.lower() for w in tokens]
    sentence_index = [i for i, x in enumerate(tokens) if x == '.']

    if len(sentence_index) == 0:
        return list()

    sentences = list()
    sentences.append(tokens[:sentence_index[0]])
    for i in range(len(sentence_index) - 1):
        b1 = sentence_index[i]
        b2 = sentence_index[i+1]
        sentence = tokens[b1+1:b2]
        sentences.append(list(sentence))
    return sentences

def generate_lines(sentences):
    for sentence in sentences:
        yield sentence

def get_sentences():
    sentences = list()
    for i in tqdm(train_text_df.index):
        s = train_text_df.ix[i]['Text']
        s_sen = text2sentences(s)
        sentences.extend(s_sen)
    return sentences

def get_report(i):
    s = train_text_df.ix[i]['Text']
    s_sen = text2sentences(s)
    return s_sen
