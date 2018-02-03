#!/usr/bin/env python

import gensim
import argparse
import re
import copy

import numpy as np
import pandas as pd

from tqdm import tqdm
from model import CNNText

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

class DocToVec(object):
    def __init__(self, model_path):
        self.model = gensim.models.Doc2Vec.load(model_path)

    def __call__(self, inp):
        ret = list()
        for sentence in tokenizer(inp):
            for word in sentence:
                try:
                    ret.append(self.model[word])
                except:
                    continue
        return ret

    def toNumpy(self, inp):
        return np.expand_dims(np.array(self(inp)), 0)

class CommentData(object):
    def __init__(self, csv_path, doc2vec_path):
        self.doc2vec = DocToVec(doc2vec_path)
        self.df = pd.read_csv(csv_path)

    def get_label(self, row):
        if row['toxic'] == 1:
            return 1
        elif row['severe_toxic'] == 1:
            return 2
        elif row['obscene'] == 1:
            return 3
        elif row['threat'] == 1:
            return 4
        elif row['insult'] == 1:
            return 5
        elif row['identity_hate'] == 1:
            return 6
        else:
            return 0

    def get_data(self, row):
        text = row['comment_text']
        return self.doc2vec.toNumpy(text)

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        for i, r in self.df.iterrows():
            yield self.get_label(r), self.get_data(r)
            if i == 1000:
                break

train_database = CommentData(
    csv_path = '/media/noud/data/noud/toxic/train.csv',
    doc2vec_path = '../doc2vec/toxic.doc2vec',
)

from torch.autograd import Variable
import time
import torch
from torch import nn

def train_model(model, criterion, optimizer, scheduler, data_loader,
		use_gpu = True, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        class_correct = list(0. for i in range(7))
        class_total = list(0. for i in range(7))

        # Each epoch has a training and validation phase
	scheduler.step()
	model.train(True)  # Set model to training mode

	running_loss = 0.0
	running_corrects = 0

	# Iterate over data.
	for label, data in tqdm(data_loader):
	    # wrap them in Variable
            if len(data[0]) == 0:
                data = np.ones((1, 1, 100)).astype(float)
            inputs = torch.from_numpy(np.array([data]))
            labels = torch.from_numpy(np.array([label]))
	    if use_gpu:
		inputs = Variable(inputs.cuda()).float()
		labels = Variable(labels.cuda())
	    else:
		inputs, labels = Variable(inputs).float(), Variable(labels)

	    # zero the parameter gradients
	    optimizer.zero_grad()

	    # forward
	    outputs = model(inputs)
	    _, preds = torch.max(outputs.data, 1)
	    loss = criterion(outputs, labels)

            label1 = np.array([1]) if label == 1 else np.array([0])
            label1 = Variable(torch.from_numpy(label1).cuda())
            out1 = model.class1()
            loss1 = criterion(out1, label1)

	    loss.backward()
	    optimizer.step()

	    # statistics
	    running_loss += loss.data[0] * inputs.size(0)
	    running_corrects += torch.sum(preds == labels.data)
            c = (preds == labels.data).squeeze()
            class_correct[label] += c[0]
            class_total[label] += 1

	epoch_loss = running_loss / len(data_loader)
	epoch_acc = float(running_corrects) / len(data_loader)

	print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

	# deep copy the model
	best_acc = epoch_acc
	best_model_wts = copy.deepcopy(model.state_dict())

        for i in range(7):
            print('Class: {}, Total: {}, Accuracy: {}'.format(
                i, class_total[i], class_correct[i] / (class_total[i] + 1.0)
            ))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model = CNNText().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                         train_database, use_gpu = True, num_epochs = 15)

