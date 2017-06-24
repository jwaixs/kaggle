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

np.random.seed(37)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
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

use_gpu = torch.cuda.is_available()
if use_gpu:
    print('Use gpu')
else:
    print('Use cpu')

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()

    best_model = model
    best_auc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            y_score = list()
            y_true = list()

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                for i in range(len(outputs.data)):
                    y_true.append(labels.data[i])
                    v = np.exp(outputs.data[i])
                    s = v[1] / v.sum()
                    if 0 <= s and s <= 1:
                        y_score.append(s)
                    else:
                        y_score.append(1.0)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            epoch_auc = roc_auc_score(y_true, y_score)

            print('{} Loss: {:.4f} Acc: {:.4f} auc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_auc))

            # deep copy the model
            if phase == 'valid' and epoch_auc > best_auc:
                best_epoch = epoch
                best_auc = epoch_auc
                best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val auc: {:4f}'.format(best_auc))
    print('Epoch: {}'.format(best_epoch))
    return best_model, best_auc, best_epoch

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def predict(model, data_set):
    predictions = list()
    for data in data_set:
        inputs, labels = data

        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), \
                Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        for i in range(outputs.size()[0]):
            v = np.exp(outputs[i].data)
            score = v[1] / v.sum()
            predictions.append(score)

    filenames = data_set.dataset.imgs
    results = list()
    for fname, score in zip(filenames, predictions):
        image_id = int(fname[0].split('/')[-1].split('.')[0])
        results.append((image_id, score))
    results.sort(key = lambda x : x[0])
    name = list()
    invasive = list()
    for n, s in results:
        name.append(n)
        invasive.append(round(s, 4))

    return name, invasive

# ResNet 18, 34, 50, 101, 152
def general_vgg_model(model_name, n_classifiers = 2, pretrained = True, train_all_params = True):
    model_ft = getattr(models, model_name)(pretrained = pretrained)
    if not train_all_params:
        for param in model_ft.parameters():
            param.requires_grad = False
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    return model_ft

def resnet18(n_classifiers = 2, pretrained = True, train_all_params = True):
    return general_vgg_model('resnet18', n_classifiers, pretrained, train_all_params)

def resnet34(n_classifiers = 2, pretrained = True, train_all_params = True):
    return general_vgg_model('resnet34', n_classifiers, pretrained, train_all_params)

def resnet50(n_classifiers = 2, pretrained = True, train_all_params = True):
    return general_vgg_model('resnet50', n_classifiers, pretrained, train_all_params)

def resnet101(n_classifiers = 2, pretrained = True, train_all_params = True):
    return general_vgg_model('resnet101', n_classifiers, pretrained, train_all_params)

def resnet152(n_classifiers = 2, pretrained = True, train_all_params = True):
    return general_vgg_model('resnet152', n_classifiers, pretrained, train_all_params)

# VGGs 11, 13, 16, 19
def general_vgg_model(model_name, n_classifiers = 2, pretrained = True, train_all_params = True):
    model_ft = getattr(models, model_name)(pretrained = pretrained)
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace = True),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace = True),
        nn.Linear(4096, n_classifiers)
    )
    return model_ft

def vgg11(n_classifiers = 2, pretrained = True, train_all_params = True):
    return general_vgg_model('vgg11', n_classifiers, pretrained, train_all_params)

def vgg13():
    return general_vgg_model('vgg13', n_classifiers, pretrained, train_all_params)

def vgg16():
    return general_vgg_model('vgg16', n_classifiers, pretrained, train_all_params)

def vgg19():
    return general_vgg_model('vgg19', n_classifiers, pretrained, train_all_params)

# DenseNets 121, 161, 169, 201
def general_densenet_model(model_name, n_classifiers = 2, pretrained = True, train_all_params = True):
    model_ft = getattr(models, model_name)(pretrained = pretrained)
    if not train_all_params:
        for param in model_ft.parameters():
            param.requires_grad = False
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, n_classifiers)
    return model_ft

def densenet121(n_classifiers = 2, pretrained = True, train_all_params = True):
    return general_densenet_model('densenet121', n_classifiers, pretrained, train_all_params)

def densenet161(n_classifiers = 2, pretrained = True, train_all_params = True):
    return general_densenet_model('densenet161', n_classifiers, pretrained, train_all_params)

def densenet169(n_classifiers = 2, pretrained = True, train_all_params = True):
    return general_densenet_model('densenet121', n_classifiers, pretrained, train_all_params)

def densenet201(n_classifiers = 2, pretrained = True, train_all_params = True):
    return general_densenet_model('densenet121', n_classifiers, pretrained, train_all_params)

number_of_epochs = 25
date = datetime.datetime.now().strftime('%Y%m%d')

for model_func, model_name in [
            (vgg11, 'vgg11'), (vgg13, 'vgg13'), (vgg16, 'vgg16'), (vgg19, 'vgg19'),
            (densenet121, 'densenet121'), (densenet161, 'densenet161'), (densenet169, 'densenet169'), (densenet201, 'densenet201'),
            (resnet18, 'resnet18'), (resnet34, 'resnet34'), (resnet50, 'resnet50'), (resnet101, 'resnet101'), (resnet152, 'resnet152')
        ]:
    model_name += '_pretrained_best_auc'
    print('Training model {}'.format(model_name))
    for fold in range(4):
	data_dir = 'dltrain/fold_{}'.format(fold)
	dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
		 for x in ['train', 'valid', 'test']}
	dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=4,
						       shuffle = True if x == 'train' else False, num_workers=4)
			for x in ['train', 'valid', 'test']}
	dset_sizes = {x: len(dsets[x]) for x in ['train', 'valid', 'test']}
	dset_classes = dsets['train'].classes

	model_ft = model_func(pretrained = True)
	if use_gpu:
	    model_ft = model_ft.cuda()

	criterion = nn.CrossEntropyLoss()
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)
	best_model, best_acc, best_epoch = train_model(
	    model_ft,
	    criterion,
	    optimizer_ft,
	    exp_lr_scheduler,
	    num_epochs = number_of_epochs
	)

	torch.save(best_model, './models/{}_{}.torchmodel'.format(date, model_name))

	name, invasive = predict(best_model, dset_loaders['valid'])
	submission = pd.DataFrame()
	submission['name'] = name
	submission['invasive'] = invasive
	submission.to_csv('./submissions/{}_submission_valid_{}_fold_{}.csv'.format(date, model_name, fold), index = False)

	name, invasive = predict(best_model, dset_loaders['test'])
	submission = pd.DataFrame()
	submission['name'] = name
	submission['invasive'] = invasive
	submission.to_csv('./submissions/{}_submission_test_{}_fold_{}.csv'.format(date, model_name, fold), index = False)

        del model_ft
        del best_model
