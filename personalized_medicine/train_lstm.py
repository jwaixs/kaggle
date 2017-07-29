import numpy as np
import sys

import torch
from torch.autograd import Variable
from torch import optim, nn

import gensim

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

sentence = [u'previous',
             u'work',
             u'has',
             u'shown',
             u'that',
             u'cdk10',
             u'silencing',
             u'increases',
             u'ets2',
             u'(',
             u'v-ets',
             u'erythroblastosis',
             u'virus',
             u'e26',
             u'oncogene',
             u'homolog',
             u'2',
             u')',
             u'-driven',
             u'activation',
             u'of',
             u'the',
             u'mapk',
             u'pathway',
             u',',
             u'which',
             u'confers',
             u'tamoxifen',
             u'resistance',
             u'to',
             u'breast',
             u'cancer',
             u'cells']

w2v_sentence = np.array(map(lambda e : word2vec_model[e], sentence))
shape = w2v_sentence.shape
w2v_sentence.resize((shape[0], 1, shape[1]))

rnn = LSTMNet(100, 128, 7).cuda()
loss = torch.nn.CrossEntropyLoss(size_average = True)
optimizer = optim.SGD(rnn.parameters(), lr = 0.1, momentum = 0.9)

for i in range(10):
    x = to_torch_var(torch.from_numpy(w2v_sentence))
    y = to_torch_var(torch.from_numpy(np.array([1])))
    optimizer.zero_grad()
    fx = rnn.forward(x)
    output = loss.forward(fx, y)
    output.backward()
    optimizer.step()
    print(i, fx, output.data[0])
