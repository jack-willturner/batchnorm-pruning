'''
Spatial batch normalization layer with ISTA regularisation

Thanks to: https://twitter.com/jeremyphoward/status/938882675175186432?lang=en-gb
'''

import torch
import torch.nn as nn

'''
Batchnorm as a module - autograd takes care of gamma/beta
Unsure how to add ISTA penalty to gamma update
'''
class BatchNorm2d(nn.Module):
    def __init__(self, size, ista, follow=False, stride=2):
        super().__init__()
        self.beta  = nn.Parameter(torch.zeros(size, 1, 1))
        self.gamma = nn.Parameter(torch.ones(size, 1, 1))
        self.follow = follow # used to decide on avg update strategy

    def forward(self, x):
        x_norm = x.transpose(0,1).contiguous().view(x.size(1), -1)

        if self.training:
            self.means = x_norm.mean(1)[:,None,None]
            self.stds  = x_norm.std (1)[:,None,None]

        x = x - self.means
        x = x / self.stds

        if follow:
            # update means differently
            self.means - <mask of 1s where gamma = 0> * relu(self.beta).t * sum_reduce(W)

        return self.gamma * x + self.beta
