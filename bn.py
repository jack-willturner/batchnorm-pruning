'''
Spatial batch normalization layer with ISTA regularisation

Thanks to: https://twitter.com/jeremyphoward/status/938882675175186432?lang=en-gb
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Batchnorm as a module - autograd takes care of gamma/beta
'''
class BatchNormLayer(nn.Module):
    def __init__(self, size, alpha=None, follow=False):
        super().__init__()
        self.beta  = nn.Parameter(torch.zeros(size, 1, 1))
        self.gamma = nn.Parameter(torch.ones (size, 1, 1))

        self.alpha = None
        if alpha:
            self.alpha = nn.Parameter(torch.ones (size, 1, 1) * alpha)
            self.alpha.requires_grad = False

        self.follow = follow # used to decide on avg update strategy

    # W = weights from prior convolution
    def forward(self, x, W):
        #x_norm = x.transpose(0,1).contiguous().view(x.size(1), -1)
        x_norm = x
        if self.training:
            self.means = x_norm.mean(1)[:,None,None]
            self.stds  = x_norm.std (1)[:,None,None]

        x = x - self.means
        x = x / self.stds

        #if self.follow:
            # unsure about this.
            # page 5: equation at bottom
            #self.means - torch.eq(self.gamma, 0) * F.relu(self.beta).transpose(0,1) * torch.sum(W)

        return self.gamma * x + self.beta

    def reduce_gammas(self):
        if self.alpha:
            print("scaling gammas")
            self.gamma * self.alpha


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
