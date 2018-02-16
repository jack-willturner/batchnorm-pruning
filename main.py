''' https://arxiv.org/pdf/1802.00124v1.pdf '''
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
def compute_penalties(model, image_dim=28, rho=0.000001):
    penalties = []

    # only considering conv layers with batchnorm
    layers = list(filter(lambda l : isinstance(l, nn.Conv2d), list(model.children())))

    # zip xs ([tail xs]) - need to know kernel size of follow-up layer
    for i in range(len(layers)):
        l    = layers[i]
        tail = layers[i+1:]

        i_w, i_h = image_dim, image_dim
        k_w, k_h = l.kernel_size[0], l.kernel_size[1]
        c_prev   = l.in_channels
        c_next   = l.out_channels

        follow_up_cost = 0.
        for follow_up_conv in tail:
            image_dim       = ((image_dim - k_w + 2*l.padding[0]) / l.stride[0]) + 1
            follow_up_cost += follow_up_conv.kernel_size[0] * follow_up_conv.kernel_size[1] * follow_up_conv.in_channels + image_dim**2

        ista     = rho * ((1 / i_w * i_h) * (k_w * k_h * c_prev + follow_up_cost))

        penalties.append(ista)

    return penalties



def scale_down_gammas(alpha, layers):
    for layer in layers:
        layer.weight = alpha * layer.weight


def train_models(model_name, model_weights, num_epochs):

    best_acc = 0.
    learning_rate = 0.1

    ista_penalties = compute_penalties(model_weights)
    print(ista_penalties)

    optimizer    = optim.SGD(model_weights.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    bn_optimizer = bnopt.BatchNormSGD([model_weights.bn2.weight], lr=learning_rate, ista=ista_penalties, momentum=0.9)

    for epoch in range(1,num_epochs):

        train(model_weights, epoch, optimizer, bn_optimizer, train_loader)
        best_acc = test(model_name, model_weights, test_loader, best_acc)

    return best_acc


model = LeNet()
train_models(model_name="LeNet",model_weights=model,num_epochs=2)
