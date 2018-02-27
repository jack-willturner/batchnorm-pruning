''' https://arxiv.org/pdf/1802.00124v1.pdf '''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter, FileWriter

import os
import argparse

from utils import *
from torch.autograd import Variable

import sgd as bnopt

from models import *

from models.layers import bn

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

def scale_gammas(alpha, model, scale_down=True):
    # get pairs of consecutive layers
    layers = list(model.children())

    alpha_ = 1 / alpha

    if not scale_down:
        # after training we want to scale back up so need to invert alpha
        alpha_  = alpha
        alpha   = 1 / alpha

    for l1, l2 in zip(layers,layers[1:]):
        if(isinstance(l1, bn.BatchNorm2dEx) and isinstance(l2, nn.Conv2d)):
            l1.weight.data = l1.weight.data * alpha
            l2.weight.data = l2.weight.data * alpha_

    return model


def switch_to_follow(model):
    first = True # want to skip the first bn layer - only do follow up layers
    for layer in list(model.children()):
        if isinstance(layer, bn.BatchNorm2dEx):
            if not first:
                layer.follow = True
            first = False

def train_model(model_name, model_weights, ista_penalties, num_epochs):

    best_acc = 0.
    learning_rate = 0.1

    # should weight decay be zero?
    optimizer    = optim.SGD(filter(lambda l : not isinstance(l, bn.BatchNorm2dEx), list(model.parameters())), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    bn_optimizer = bnopt.BatchNormSGD([l.weight for l in list(model_weights.children()) if isinstance(l, bn.BatchNorm2dEx)], lr=learning_rate, ista=ista_penalties, momentum=0.9)

    for epoch in range(1,num_epochs):
        train(model_weights, epoch, writer, "train", optimizer, bn_optimizer, train_loader)
        best_acc = test(model_name, model_weights, epoch, writer, "train", test_loader, best_acc)

    return best_acc


if __name__=='__main__':
    train_loader, test_loader = get_data()

    writer = SummaryWriter()


    # get the model
    model = ResNet18()
    model_name = "ResNet-18"

    # fixed hyperparams for now - need to add parsing support
    alpha = 1.
    rho   = 0.001

    # step one: compute ista penalties
    ista_penalties = compute_penalties(model, rho)

    # step two: gamma rescaling trick
    scale_gammas(alpha, model=model, scale_down=True)

    count_sparse_bn(model, writer, 0)

    # step three: end-to-end-training
    train_model(model_name=model_name, model_weights=model, ista_penalties=ista_penalties, num_epochs=2)

    # step four: remove constant channels by switching bn to "follow" mode
    switch_to_follow(model)

    # step five: gamma rescaling trick
    scale_gammas(alpha, model=model, scale_down=False)

    # step six: finetune
    num_retraining_epochs=1
    best_acc = 0.
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    for epoch in range(1, num_retraining_epochs):
        train(model, epoch, writer,"finetune", optimizer, bn_optimizer=None, trainloader=train_loader, finetune=True)
        best_acc = test(model_name, model, epoch, writer,"finetune", test_loader, best_acc)
        count_sparse_bn(model, writer, epoch)



    ##### Remove all unnecessary channels
    model_name = "ResNet18Compressed"

    # zero out any channels that have a 0 batchnorm weight
    print("Compressing model...")
    sparsify_on_bn(model)

    new_model = compress_convs(model)

    # step six: finetune
    num_retraining_epochs=30
    best_acc = 0.
    new_optimizer = optim.SGD(new_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    for epoch in range(1, num_retraining_epochs):
        train(new_model, epoch, writer, "compress_finetune",  new_optimizer, bn_optimizer=None, trainloader=train_loader, finetune=True)
        best_acc = test(model_name, new_model, epoch, writer, "compress_finetune", test_loader, best_acc)

    writer.close()
