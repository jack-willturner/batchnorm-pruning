'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
from __future__ import print_function

import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init

from collections import namedtuple
from models.layers.bn import BatchNorm2dEx

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

criterion = nn.CrossEntropyLoss()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#####################
## data preprocessing
#####################

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

#####################
## data augmentation
#####################

class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:,y0:y0+self.h,x0:x0+self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}

    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)

class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x

    def options(self, x_shape):
        return {'choice': [True, False]}

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:,y0:y0+self.h,x0:x0+self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}


class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k,v) in choices.items()}
            data = f(data, **args)
        return data, labels

    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k:np.random.choice(v, size=N) for (k,v) in options.items()})


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def get_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    return trainloader, testloader

def save_state(model_name, model_weights, acc):
    print('==> Saving model ...')
    state = {
            'acc': acc,
            'state_dict': model_weights.state_dict(),
            }
    for key in list(state['state_dict'].keys()):
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)

    torch.save(state, 'saved_models/ckpt'+model_name+'.t7')

def load_best(model_name, model_wts):
    filename   = 'saved_models/ckpt' + model_name + '.t7'
    checkpoint = torch.load(filename)

    best_acc = checkpoint['acc']
    print("Loading checkpoint with best_acc: ", best_acc)

    state_dict = checkpoint['state_dict']
    model_wts.load_state_dict(state_dict)

    return model_name, model_wts, best_acc

# Training
def train(model, epoch, writer, plot_name,  optimizer, bn_optimizer, trainloader, finetune=False):
    #model_name, model = model[0], model[1]
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    
    print('\nEpoch: %d' % epoch)

    model.train()

    train_loss = 0
    correct    = 0
    total      = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        if not finetune:
            bn_optimizer.zero_grad()

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        if not finetune:
            bn_optimizer.step()

        train_loss  += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total       += targets.size(0)
        correct     += predicted.eq(targets.data).cpu().sum()

        acc = 100.*correct/total

        writer.add_scalar((plot_name + ": Train/Loss"), loss, epoch)
        writer.add_scalar((plot_name + ": Train/Top1"), acc,  epoch)


        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(model_name, model, epoch, writer,plot_name, testloader, best_acc):
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total

    writer.add_scalar((plot_name + ": Val/Loss"), loss, epoch)
    writer.add_scalar((plot_name + ": Val/Top1"), acc,  epoch)

    if acc > best_acc:
        print('Saving..')
        save_state(model_name, model, acc)
        best_acc = acc
    print(best_acc)

    return best_acc

# deep compression

def count_params(model):
    total = 0
    for param in model.parameters():
        flat = param.view(param.size(0), -1)
        flat = flat.data.cpu().numpy()
        total = total + np.count_nonzero(flat)
        print(total)
    print("====================")
    return total

def compute_dims(model):
    image_dims = []

    input_width  = 40.
    input_height = 40.

    ls = expand_model(model, []) # this seems like the most reasonable way to iterate

    num_input_channels = 3 # keep track of the number of channels so that if we see a decrease, we know we have hit a shortcut and can ignore it

    for l1 in ls:
        if isinstance(l1, nn.Conv2d):
            if l1.in_channels >= num_input_channels:
                k_w, k_h = l1.kernel_size[0], l1.kernel_size[1]
                padding_h, padding_w  = l1.padding[0], l1.padding[1]
                stride = l1.stride[0]

                input_height = ((input_height + 2 * padding_h - l1.dilation[0] * (k_h - 1) - 1) / stride) + 1
                input_width  = ((input_width  + 2 * padding_w - l1.dilation[1] * (k_w - 1) - 1) / stride) + 1
                assert(input_height == input_width)

                input_height = int(input_height)
                input_width  = int(input_width)

                num_input_channels = l1.out_channels
                image_dims.append(input_height)
            else:
                image_dims.append(input_height)

        elif isinstance(l1, nn.MaxPool2d):
            k_w, k_h = l1.kernel_size, l1.kernel_size
            padding_w, padding_h  = l1.padding, l1.padding
            stride = l1.stride

            input_height = ((input_height + 2 * padding_h - l1.dilation * (k_h - 1) - 1) / stride) + 1
            input_width  = ((input_width  + 2 * padding_w - l1.dilation * (k_w - 1) - 1) / stride) + 1
            assert(input_height == input_width)
            image_dims.append(int(input_height))

    return image_dims

def count_sparse_bn(model, writer, epoch):
    total = 0.

    input_width  = 28.
    input_height = 28.

    ls = expand_model(model, []) # this seems like the most reasonable way to iterate

    for l1, l2 in zip(ls, ls[1:]):
        if isinstance(l1, nn.Conv2d) and isinstance(l2, BatchNorm2dEx):
            num_nonzero = np.count_nonzero(l2.weight.data.cpu().numpy())

            writer.add_scalar(str(l1), num_nonzero, epoch)
            k_w, k_h = l1.kernel_size[0], l1.kernel_size[1]
            padding_w, padding_h  = l1.padding[0], l1.padding[1]
            stride = l1.stride[0]

        mac_ops_per_kernel = (input_width + padding_w) * (input_height + padding_h) * k_w * k_h

        input_height = (input_height - k_h + (2 * padding_h) / stride) + 1
        input_width  = (input_width  - k_w + (2 * padding_w) / stride) + 1


        mac_ops = mac_ops_per_kernel * num_nonzero
        total  += mac_ops


    writer.add_scalar("MAC ops", total, epoch)
    return total

def print_layer_ista_pair(model, istas):
    print("\n\n\n======PENALTY LAYER PAIRS======\n")
    bn_layers = [l for l in expand_model(model, []) if isinstance(l, BatchNorm2dEx)]
    for layer, penalty in zip(bn_layers, istas):
        print(layer, "\t\t:\t\t", penalty)
    print("\n\n\n")

def print_sparse_bn(model):
    nonzeros = []

    for layer in expand_model(model, []):
        if isinstance(layer, BatchNorm2dEx):
            num_nonzero = np.count_nonzero(layer.weight.cpu().data.numpy())
            nonzeros.append(num_nonzero)
            print(layer,"\t\t:\t\t",  num_nonzero)
    return nonzeros

def get_sparse_bn(layer):
    num_nonzero = np.count_nonzero(layer.weight.cpu().data.numpy())
    return num_nonzero

import numpy as np

def calculate_threshold(weights, ratio):
    return np.percentile(np.array(torch.abs(weights).cpu().numpy()), ratio)


def sparsify(model, sparsity_level=50.):
    for name, param in model.named_parameters():
        if 'weight' in name:
            threshold = calculate_threshold(param.data, sparsity_level)
            mask      = torch.gt(torch.abs(param.data), threshold).float()

            param.data = param.data * mask
    return model


def sparsify_on_bn(model):
    '''
    Here we zero out whole planes where their batchnorm weight is 0
    1. Consider lists in pairs
    2. If conv followed by batchnorm - get nonzeros from batchnorm
    3. Zero out whole conv filters
    '''

    for l1, l2 in zip(expand_model(model, []), expand_model(model, [])[1:]):
        if isinstance(l1, nn.Conv2d) and isinstance(l2, BatchNorm2dEx):
            zeros = argwhere_nonzero(l2.weight, batchnorm=True)
            for z in zeros:
                l1.weight.data[z] = 0.

def count_zeros(layer):
    weights = layer.weight.cpu().data.numpy()
    return len(np.where(weights==0)[0])


def argwhere_nonzero(layer, batchnorm=False):
    indices=[]
    # for batchnorms we want to do the opposite
    if batchnorm:
        for idx,w in enumerate(layer):
            if torch.sum(torch.abs(w)).data.cpu().numpy() == 0.:
                indices.append(idx)
    else:
        for idx,w in enumerate(layer):
            if torch.sum(torch.abs(w)).data.cpu().numpy() != 0.:
                indices.append(idx)

    return indices

def prune_conv(indices, layer, follow=False):
    # follow tells us whether we need to prune input channels or output channels
    a,b,c,d = layer.weight.data.cpu().numpy().shape

    if not follow:
        # prune output channels
        layer.weight.data = torch.from_numpy(layer.weight.data.cpu().numpy()[indices])
        if layer.bias is not None:
            layer.bias.data   = torch.from_numpy(layer.bias.data.cpu().numpy()[indices])
    else:
        # prune input channels - so don't touch biases because we're not changing the number of neurons/nodes/output channels
        layer.weight.data = torch.from_numpy(layer.weight.data.cpu().numpy()[:,indices])

def prune_fc(indices, channel_size, layer, follow_conv=True):
    a,b = layer.weight.data.cpu().numpy().shape
    if follow_conv:
        # if we are following a conv layer we need to expand each index by the size of the plane
        indices = [item for sublist in list((map(lambda i : np.arange((i * channel_size), (i*channel_size+channel_size)), indices))) for item in sublist]

    layer.weight.data = torch.from_numpy(layer.weight.data.cpu().numpy()[:,indices])

def prune_bn(indices, layer):
    layer.weight.data = torch.from_numpy(layer.weight.data.cpu().numpy()[indices])
    layer.bias.data   = torch.from_numpy(layer.bias.data.cpu().numpy()[indices])

    layer.running_mean = torch.from_numpy(layer.running_mean.cpu().numpy()[indices])
    layer.running_var  = torch.from_numpy(layer.running_var.cpu().numpy()[indices])

def compress_convs(model, compressed):

    ls = expand_model(model, [])

    channels = []
    nonzeros = []
    skip_connection = []

    for l1, l2 in zip(ls, ls[1:]):
        if isinstance(l1, nn.Conv2d):

            nonzeros = argwhere_nonzero(l1.weight)
            nonzeros_altered = True

            channels.append(len(nonzeros))
            channel_size = l1.kernel_size[0] * l1.kernel_size[1]
            prune_conv(nonzeros, l1)

            if isinstance(l2, nn.Conv2d):
                prune_conv(nonzeros, l2, follow=True)
            elif isinstance(l2, nn.Linear):
                prune_fc(nonzeros, channel_size, l2, follow_conv=True)
            elif isinstance(l2, nn.Sequential):
                # save for skip connection
                skip_connection = nonzeros

        elif isinstance(l1, nn.BatchNorm2d):
            # no need to append to channels since we will already have done it
            # i.e. num of channels in bn is same as num of channels in last conv layer

            assert nonzeros_altered, "batch norm layer appeared before a convolutional layer"

            l1_channels = l1.num_features

            prune_bn(nonzeros, l1)

            if isinstance(l2, nn.Conv2d):
                if (l2.in_channels < l1_channels) and (len(skip_connection) > 0): # if this is a skip connection:
                    prune_conv(skip_connection, l2, follow=True)
                elif l1_channels == l2.in_channels:
                    prune_conv(nonzeros, l2, follow=True)
            elif isinstance(l2, nn.Linear):
                prune_fc(nonzeros, channel_size, l2, follow_conv=True) # TODO fix this please

    print("remaining channels: ", channels)

    new_model = compressed(channels)

    #for layer in model.children():
    #    print(layer)


    #print("\n\n\n======================\n\n\n")

    #for layer in new_model.children():
    #    print(layer)

    #print("\n\n\n=====================\n\n\n")

    for original, compressed in zip(expand_model(model, []), expand_model(new_model, [])):
        print("original: ", original)
        print("compressed: ", compressed)
        print("===\n\n\n\n")
        if not isinstance(original, nn.Sequential) and not isinstance(original, nn.MaxPool2d):
            compressed.weight.data = original.weight.data
            if original.bias is not None:
                compressed.bias.data   = original.bias.data

    return new_model

def expand_model(model, layers=[]):
    for layer in model.children():
         if len(list(layer.children())) > 0:
             expand_model(layer, layers)
         else:
             layers.append(layer)
    return layers
