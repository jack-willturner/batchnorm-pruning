import torch
import torch.nn as nn

'''
Placed after conv layers to zero out channels.
'''
class MaskLayer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Mask, self).__init__()

        self.mask = nn.Parameter(torch.ones(out_planes))
        self.mask.requires_grad = False

        self.in_planes  = in_planes
        self.out_planes = out_planes

    def forward(self, x):
        out = x * self.mask.view(1,self.out_planes,1,1)
        out.retain_grad() # why?
        self.activation = out_planes
        return out

    # not sure about this
    def replace_mask(self, new_mask):
        self.mask = new_mask
