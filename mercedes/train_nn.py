import numpy as np
import pandas as pd
import copy
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils

from torch.autograd import Variable
from sklearn.model_selection import KFold
from load_data import X_train, y_train, X_test

torch.manual_seed(0)

use_gpu = torch.cuda.is_available()


class NNet(torch.nn.Module):
    def __init__(self, n_features, n_hidden1, n_hidden2, n_output):
        super(NNet, self).__init__()
        self.hidden1 = torch.nn.Linear(n_features, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x

net = NNet(n_features = X_train.shape[1], n_hidden1 = 32, n_hidden2 = 16, n_output = 1)
if use_gpu:
    net = net.cuda()
print(net)

optimizer = torch.optim.Rprop(net.parameters(), lr = 0.001)
loss_func = torch.nn.MSELoss()

def train_valid_model(model, tloader, vloader, n_epochs = 100):
    best_valid_loss = None
    best_valid_model = None
    best_valid_epoch = None

    for epoch in range(n_epochs):
        train_loss = list()
        for features, labels in tloader:
            if use_gpu:
                features = Variable(features.cuda())
                labels = Variable(labels.cuda()).float()
            else:
                features = Variable(features)
                labels = Variable(labels).float()

            predictions = model(features)
            loss = loss_func(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.data[0])

        valid_loss = list()
        for features, labels in vloader:
            if use_gpu:
                features = Variable(features.cuda())
                labels = Variable(labels.cuda()).float()
            else:
                features = Variable(features)
                labels = Variable(labels).float()

            predictions = model(features)
            loss = loss_func(predictions, labels)

            valid_loss.append(loss.data[0])

        vloss = sum(valid_loss) / len(valid_loss)

        if best_valid_loss == None or vloss < best_valid_loss:
            best_valid_loss = vloss
            best_valid_model = copy.deepcopy(model)
            best_valid_epoch = epoch

    return best_valid_loss, best_valid_epoch, best_valid_model

def train_model(model, tloader, n_epochs = 100):
    for epoch in range(n_epochs):
        train_loss = list()
        for features, labels in tloader:
            if use_gpu:
                features = Variable(features.cuda())
                labels = Variable(labels.cuda()).float()
            else:
                features = Variable(features)
                labels = Variable(labels).float()

            predictions = model(features)
            loss = loss_func(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.data[0])

    return model, train_loss[-1]

def nn_train(X, y):
    X_train_tensor = torch.FloatTensor(X.values)
    y_train_tensor = torch.FloatTensor(y.values)

    train = data_utils.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = data_utils.DataLoader(train, batch_size = 200, shuffle = True)

    model, loss = train_model(net, train_loader, n_epochs = 747)

    return model, loss

def nn_cv(X, y, n_splits = 5):
    valid_loss = list()
    valid_epoch = list()

    kf = KFold(n_splits = n_splits)

    for train, valid in kf.split(X_train):
        _X_train = X.ix[train]
        _y_train = y.ix[train]
        _X_valid = X.ix[valid]
        _y_valid = y.ix[valid]

        _X_train_tensor = torch.FloatTensor(_X_train.values)
        _y_train_tensor = torch.FloatTensor(_y_train.values)
        _X_valid_tensor = torch.FloatTensor(_X_valid.values)
        _y_valid_tensor = torch.FloatTensor(_y_valid.values)

        train = data_utils.TensorDataset(_X_train_tensor, _y_train_tensor)
        train_loader = data_utils.DataLoader(train, batch_size = 50, shuffle = True)
        valid = data_utils.TensorDataset(_X_valid_tensor, _y_valid_tensor)
        valid_loader = data_utils.DataLoader(valid, batch_size = 50, shuffle = True)

        loss, epoch, model = train_valid_model(net, train_loader, valid_loader)
        valid_loss.append(loss)
        valid_epoch.append(epoch)

        print('Finished fold, loss: {}'.format(loss)) 

    return valid_loss, valid_epoch

vloss, vepochs = nn_cv(X_train, y_train)
print(np.mean(vloss), np.std(vloss), np.mean(vepochs), np.std(vepochs))

def predict(model, X):
    X_test_tensor = torch.FloatTensor(X.values)
    y_test_tensor = torch.zeros(len(X.values))
    test = data_utils.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = data_utils.DataLoader(test, batch_size = 200, shuffle = False)

    all_predictions = list()
    for features, _ in test_loader:
        if use_gpu:
            features = Variable(features.cuda())
        else:
            features = Variable(features)

        predictions = model(features)

        for i in range(len(predictions)):
            all_predictions.append(predictions.data[i][0])

    return all_predictions

#model, loss = nn_train(X_train, y_train)
