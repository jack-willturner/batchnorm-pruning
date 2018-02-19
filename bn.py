'''
Spatial batch normalization layer with ISTA regularisation
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm2dEx(nn.BatchNorm2d):
    def __init__(self, size, follow=False):
        super().__init__(size)
        self.alpha = None
        self.follow = follow # used to decide on avg update strategy

    def forward(self, input, W):
        out =  F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)

        if self.follow:
            activations = F.relu(self.bias).view(1,-1) * torch.sum(W)
            mean = self.running_mean - (torch.eq(self.weight, 0).float().mul(activations)).data
            mean = mean.view(-1)
            self.running_mean = mean
        return out
