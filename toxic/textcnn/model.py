#!/usr/bin/env python

from torch import nn

class CNNText(nn.Module):
    def __init__(self):
        super(CNNText, self).__init__()

        self.features = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = (1, 2), stride = (1, 2)),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace = True),
            nn.Linear(32, 7)
        )

        self.class1ft = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace = True),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        self.feat_out = self.features(x)
        out = self.classifier(self.feat_out.view(self.feat_out.size(0), -1))
        return out

    def class1(self):
        return self.class1ft(self.feat_out.view(self.feat_out.size(0), -1))

