import numpy as np
import sys

import torch
from torch.autograd import Variable
from torch import optim, nn

import gensim

from create_sentences import get_report

def to_torch_var(var, **args):
    if torch.cuda.is_available():
        return torch.autograd.Variable(var.cuda(), **args)
    else:
        return torch.autograd.Variable(var, **args)

class LSTMNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)
        self.softmax = nn.Softmax()

    def forward(self, x):
        batch_size = x.size()[1]
        h0 = to_torch_var(torch.zeros([1, batch_size, self.hidden_dim]), requires_grad=False)
        c0 = to_torch_var(torch.zeros([1, batch_size, self.hidden_dim]), requires_grad=False)
        fx, _ = self.lstm.forward(x, (h0, c0))
        fx = self.linear.forward(fx[-1])
        fx = self.softmax(fx)
        return fx

word2vec_model = gensim.models.Word2Vec.load('./train.word2vec')


def get_report_vec(i):
    sentence = [w for s in get_report(0) for w in s]
    w2v_sentence = list()
    for w in sentence:
        if w in word2vec_model:
            w2vw = word2vec_model[w]
            w2v_sentence.append(w2vw)
    w2v_sentence = np.array(w2v_sentence)
    shape = w2v_sentence.shape
    w2v_sentence.resize((shape[0], 1, shape[1]))

    return w2v_sentence, i%7

rnn = LSTMNet(100, 1024, 7).cuda()
loss = torch.nn.CrossEntropyLoss(size_average = True)
optimizer = optim.SGD(rnn.parameters(), lr = 0.01, momentum = 0.9)

for i in range(250):
    mod = 2
    xvec, yvec = get_report_vec(i%mod)
    x = to_torch_var(torch.from_numpy(xvec))
    y = to_torch_var(torch.from_numpy(np.array([yvec])))
    optimizer.zero_grad()
    fx = rnn.forward(x)
    output = loss.forward(fx, y)
    output.backward()
    optimizer.step()
    print(i%mod, fx, output.data[0])
