from  __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import os
import pandas as pd
import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

np.random.seed(37)
nfold = 10

from load_data import train_image_list, test_image_list
from dltools.dataset.imagelist import ImageList

train_image_list = np.array(train_image_list)
test_image_list = np.array(test_image_list)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=4):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def train_cv_model(model, dst_train_loader, dset_test_loader, num_epochs):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('Use gpu')
    else:
        print('Use cpu')

    since = time.time()

    best_model = model
    best_auc = 0.0
    best_results = list()

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Train phase
        optimizer = exp_lr_scheduler(optimizer_ft, epoch)
        model.train(True)

        running_loss = list()

        for data in dset_train_loader:
            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                    Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss.append(loss.data[0])

        print('Epoch {}, loss: {}'.format(
            epoch, sum(running_loss) / len(running_loss)
        ))

        # Test phase
        model.train(False)

        running_loss = list()

        y_score = list()
        y_true = list()

        for data in dset_test_loader:
            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                    Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            for i in range(len(outputs.data)):
                y_true.append(labels.data[i])
                v = np.exp(outputs.data[i])
                s = v[1] / v.sum()
                if 0 <= s and s <= 1:
                    y_score.append(s)
                else:
                    y_score.append(1.0)

            # statistics
            running_loss.append(loss.data[0])

        epoch_auc = roc_auc_score(y_true, y_score)
        print(y_true, y_score)

        print('Epoch {}, loss: {}, auc: {}'.format(
            epoch, sum(running_loss) / len(running_loss), epoch_auc
        ))

        if epoch_auc > best_auc:
            best_model = copy.deepcopy(model)
            best_auc = epoch_auc
            best_epoch = epoch
            best_results = copy.deepcopy(zip(y_score, y_true))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val auc: {:4f}'.format(best_auc))
    print('Epoch: {}'.format(best_epoch))
    return best_model, best_epoch, best_auc, y_score


kf = KFold(n_splits = nfold, random_state = 37, shuffle = True)
i = 0
for train_index, test_index in kf.split(train_image_list):
    fold_train_list = train_image_list[train_index]
    fold_test_list = train_image_list[test_index]

    dset_train = ImageList(fold_train_list, data_transforms['train'])
    dset_train_loader = torch.utils.data.DataLoader(dset_train, batch_size = 4, shuffle = True, num_workers = 4)
    dset_train_size = len(dset_train)
    dset_train_classes = dset_train.classes

    dset_test = ImageList(fold_test_list, data_transforms['test'])
    dset_test_loader = torch.utils.data.DataLoader(dset_test, batch_size = 4, shuffle = True, num_workers = 4)
    dset_test_size = len(dset_test)

    model_ft = models.resnet18(pretrained = True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    best_model, best_epoch, best_auc, results = train_cv_model(model_ft.cuda(), dset_train_loader, dset_test_loader, 2)
    print('Fold {}, epoch: {}, auc: {}, results: {}'.format(
        i, best_epoch, best_auc, results
    ))
    i += 1
