import torch
import os

import numpy as np
import pandas as pd

from torchvision import transforms

np.random.seed(37)

data_folder = '/data/kaggle/invasive/'
train_image_folder = os.path.join(data_folder, 'train')
test_image_folder = os.path.join(data_folder, 'test')
train_labels_file = os.path.join(data_folder, 'train_labels.csv')

train_labels = pd.read_csv(train_labels_file)
train_image_list = list()
for i, r in train_labels.iterrows():
    train_image_list.append(
        (os.path.join(train_image_folder, '{}.jpg'.format(r.name+1)),
         'invasive' if r.invasive == 1 else 'non-invasive')
    )
test_image_list = list()
for test_file in os.listdir(test_image_folder):
    test_image_list.append(
        (os.path.join(test_image_folder, test_file), 'unknown')
    )

# Data augmentation and normalization for training
# Just normalization for validation
#data_transforms = {
#    'train': transforms.Compose([
#        transforms.RandomSizedCrop(224),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#    ]),
#    'test': transforms.Compose([
#        transforms.Scale(256),
#        transforms.CenterCrop(224),
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#    ]),
#}
#
#dset = ImageList(image_list, data_transforms['train'])
#dset_loader = torch.utils.data.DataLoader(dset, batch_size = 4, shuffle = True)
#dset_size = len(dset)
#dset_classes = dset.classes
#
#print(dset_size)
#print(dset_classes)
#
#for i, c in dset_loader:
#    print i, c
