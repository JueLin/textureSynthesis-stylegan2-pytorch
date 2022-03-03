from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models

import utils


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        # for x in range(4):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(4, 9):
        #     self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(9, 16):
        #     self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(16, 23):
        #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        self.slice1.add_module("3", nn.ReLU(inplace=False))
        for x in range(4, 8):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        self.slice2.add_module("8", nn.ReLU(inplace=False))
        for x in range(9, 15):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        self.slice3.add_module("15", nn.ReLU(inplace=False))
        for x in range(16, 22):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        self.slice4.add_module("22", nn.ReLU(inplace=False))
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # Assume X in range [0, 1]
        X = torch.clamp(X, min=0, max=1)
        X = utils.normalize_batch_vgg(X)
        h = self.slice1(X)
        h_conv1_2 = h
        h = self.slice2(h)
        h_conv2_2 = h
        h = self.slice3(h)
        h_conv3_3 = h
        h = self.slice4(h)
        h_conv4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3'])
        out = vgg_outputs(h_conv1_2, h_conv2_2, h_conv3_3, h_conv4_3)
        return out