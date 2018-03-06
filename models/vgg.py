'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

import torch.nn.functional as F

from .layers import bn

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=3)
        self.bn1   = bn.BatchNorm2dEx(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=3)
        self.bn2   = bn.BatchNorm2dEx(64)
        self.pool1 = nn.MaxPool2d(2, stride=2) # TODO check this

        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=2)
        self.bn3   = bn.BatchNorm2dEx(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=2)
        self.bn4   = bn.BatchNorm2dEx(128)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5   = bn.BatchNorm2dEx(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn6   = bn.BatchNorm2dEx(256)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7   = bn.BatchNorm2dEx(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn8   = bn.BatchNorm2dEx(512)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.conv9  = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn9    = bn.BatchNorm2dEx(512)
        self.conv10 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn10   = bn.BatchNorm2dEx(512)
        self.pool5  = nn.MaxPool2d(2, stride=2)

        self.fc1    = nn.Linear(512, 1028)
        self.fc2    = nn.Linear(1028, 10)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x), self.conv1.weight))
        out = F.relu(self.bn2(self.conv2(out), self.conv2.weight))
        out = self.pool1(out)

        out = F.relu(self.bn3(self.conv3(out), self.conv3.weight))
        out = F.relu(self.bn4(self.conv4(out), self.conv4.weight))
        out = self.pool2(out)

        out = F.relu(self.bn5(self.conv5(out), self.conv5.weight))
        out = F.relu(self.bn6(self.conv6(out), self.conv6.weight))
        out = self.pool3(out)

        out = F.relu(self.bn7(self.conv7(out), self.conv7.weight))
        out = F.relu(self.bn8(self.conv8(out), self.conv8.weight))
        out = self.pool4(out)

        out = F.relu(self.bn9(self.conv9(out), self.conv9.weight))
        out = F.relu(self.bn10(self.conv10(out), self.conv10.weight))
        out = self.pool5(out)

        #  reshape
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class VGGCompressed(nn.Module):
    def __init__(self, channels):
        super(VGGCompressed, self).__init__()
        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, padding=3)
        self.bn1   = nn.BatchNorm2d(channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=3)
        self.bn2   = nn.BatchNorm2d(channels[1])
        self.pool1 = nn.MaxPool2d(2, stride=2) # TODO check this

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=1, padding=2)
        self.bn3   = nn.BatchNorm2d(channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=1, padding=2)
        self.bn4   = nn.BatchNorm2d(channels[3])
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv5 = nn.Conv2d(channels[3], channels[4], 3, stride=1, padding=1)
        self.bn5   = nn.BatchNorm2d(channels[4])
        self.conv6 = nn.Conv2d(channels[4], channels[5], 3, stride=1, padding=1)
        self.bn6   = nn.BatchNorm2d(channels[5])
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.conv7 = nn.Conv2d(channels[5], channels[6], 3, stride=1, padding=1)
        self.bn7   = nn.BatchNorm2d(channels[6])
        self.conv8 = nn.Conv2d(channels[6], channels[7], 3, stride=1, padding=1)
        self.bn8   = nn.BatchNorm2d(channels[7])
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.conv9  = nn.Conv2d(channels[7], channels[8], 3, stride=1, padding=1)
        self.bn9    = nn.BatchNorm2d(channels[8])
        self.conv10 = nn.Conv2d(channels[8], channels[9], 3, stride=1, padding=1)
        self.bn10   = nn.BatchNorm2d(channels[9])
        self.pool5  = nn.MaxPool2d(2, stride=2)

        self.fc1    = nn.Linear(channels[9] * 3 * 3, 512)
        self.fc2    = nn.Linear(512, 10)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool1(out)

        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.pool2(out)

        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        out = self.pool3(out)

        out = F.relu(self.bn7(self.conv7(out)))
        out = F.relu(self.bn8(self.conv8(out)))
        out = self.pool4(out)

        out = F.relu(self.bn9(self.conv9(out)))
        out = F.relu(self.bn10(self.conv10(out)))
        out = self.pool5(out)

        #  reshape
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
