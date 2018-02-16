'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar, load_best, get_data, train, test, sparsify, count_params
from torch.autograd import Variable

import sgd as bnopt

image_dim = 28 # size of input image - use this to calculate memory footprint in compute_penalties

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2   = nn.BatchNorm2d(16)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

train_loader, test_loader = get_data()



'''
Equation (2) on page 6
'''
def compute_penalties(model):
    penalties = []

    # only considering conv layers with batchnorm
    layers = filter(lambda l : isinstance(l, nn.Conv2d), ls)

    # zip xs ([tail xs]) - need to know kernel size of follow-up layer
    for i in range(len(layers))
        l    = layers[i]
        tail = layers[i+1:]

        i_w, i_h = image_dim, image_dim
        k_w, k_h = l1.kernel_size, l1.kernel_size
        c_prev   = l1.in_channels
        c_next   = l1.out_channels

        ista     = (1 / i_w * i_h) * (k_w * k_h * c_prev + reduce(lambda l: image_size = ((image_size - k_w + 2*l1.padding / l1.stride) + 1 ) l.kernel_size * l.kernel_size * l.in_channels + (), tail))


# assume training always done on GPU - so we don't check for CPU conversions here
def train_models(model_name, model_weights, num_epochs):

    best_acc = 0.
    learning_rate = 0.1

    for epoch in range(1,num_epochs):
        optimizer    = optim.SGD(model_weights.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        bn_optimizer = bnopt.BatchNormSGD([model_weights.bn2.weight], lr=learning_rate, ista=[1], momentum=0.9)

        train(model_weights, epoch, optimizer, bn_optimizer, train_loader)
        best_acc = test(model_name, model_weights, test_loader, best_acc)

    return best_acc


model = LeNet()

layers = list(model.children())


# zip xs (tail xs) - need to know kernel size of follow-up layer
for i in range(len(layers)):
    l1 = layers[i]
    l2 = layers[(i+1):]
    print(l1, l2)
    print("\n")


# count batch norm layers
# filter out batch norm layers
# create list of tuples saying whether conv has follow up batch norm


#train_models(model_name="LeNet",model_weights=model,num_epochs=2)
