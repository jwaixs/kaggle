import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils

from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from load_data import X_train, y_train, X_test

torch.manual_seed(0)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train.values, y_train.values, test_size = 0.33, random_state = 42
)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_valid_tensor = torch.FloatTensor(X_valid)
y_valid_tensor = torch.FloatTensor(y_valid)

train = data_utils.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = data_utils.DataLoader(train, batch_size = 50, shuffle = True)
valid = data_utils.TensorDataset(X_valid_tensor, y_valid_tensor)
valid_loader = data_utils.DataLoader(valid, batch_size = 50, shuffle = True)

class NNet(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(NNet, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = NNet(n_features = X_train.shape[1], n_hidden = 1024, n_output = 1)
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr = 0.1)
loss_func = torch.nn.MSELoss()

for epoch in range(100):
    print('Start epoch: {}'.format(epoch))

    train_loss = list()
    for features, labels in train_loader:
        features = Variable(features)
        labels = Variable(labels)

        predictions = net(features)
        loss = loss_func(predictions, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.data[0])

    valid_loss = list()
    for features, labels in valid_loader:
        features = Variable(features)
        labels = Variable(labels)

        predictions = net(features)
        loss = loss_func(predictions, labels.float())

        valid_loss.append(loss.data[0])

    print('Train loss: {}'.format(sum(train_loss) / len(train_loss)))
    print('Valid loss: {}'.format(sum(valid_loss) / len(valid_loss)))


