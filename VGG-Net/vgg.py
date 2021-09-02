'''
    This is an implementation of VGGNET as described in their paper
    at arXiv:[insert citation here].
    This is not meant to represent orginal work, but rather I am
    implementing this for my own educational purposes. For any
    practical use, use an professionally made implementation
    included in the torchvision libraries or the like
'''
##These are my personal machine learning import statements, they cover most of what I regularly need for CV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#for larger jupyter plots
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 200 # 200 e.g. is really fine, but slower
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets
#we can use this to set our device. If we have a GPU available we should probably use it.
#currently the code is not written to make use of mulitple GPUs and will default to the
#first available GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#these set cuda backend parameters so that we should get more reproducable results.
#some cuda functions will prefer non-deterministic backends in some cases, eg. conv2d,
#which we would like to fix to slower deterministic backends for sharing reproducable
#experiments


def conv_3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels,3,
                    stride = 1, padding = 1, bias = False)

#want to define layers based off of their number of convolutional layers
class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(Layer,self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(conv_3x3(in_channels, out_channels))
            else:
                layers.append(conv_3x3(out_channels, out_channels))
        self._main_ = nn.Sequential(*layers)
    def forward(self,x):
        return self._main_(x)
'''
    Class to implement VGG-16 and 19 as described in their paper
    at arXiv:[cite]
    the layers of a VGG are given by convolutional layers with 3x3
    filters and stride of 1. all downsampling is performed by
    maxpool layers. No layers have batch normalization
    all inputs are assumed to be of 3 x 224 x 224
'''
class VGG(nn.Module):
    def __init__(self,type = '16',num_classes = 100,in_channels = 1):
        super(VGG,self).__init__()
        '''
            define layers for VGG 16 and VGG 19
        '''
        if type == '16':
            self.layer1 = Layer(in_channels,64,2)
            self.layer2 = Layer(64,128,2)
            self.layer3 = Layer(128,256,3)
            self.layer4 = Layer(256,512,3)
            self.layer5 = Layer(512,512,3)
        elif type == '19':
            self.layer1 = Layer(in_channels,64,2)
            self.layer2 = Layer(64,128,2)
            self.layer3 = Layer(128,256,4)
            self.layer4 = Layer(256,512,4)
            self.layer5 = Layer(512,512,4)
        # there are also 4 max pool layers
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096,num_classes)

        self._main_ = nn.Sequential(
                    self.layer1, nn.MaxPool2d(2),
                    self.layer2, nn.MaxPool2d(2),
                    self.layer3, nn.MaxPool2d(2),
                    self.layer4, nn.MaxPool2d(2),
                    self.layer5, nn.MaxPool2d(2),
                    self.fc1, self.fc1, self.fc2
                    )

    def forward(self,x):
        '''
            run a forward pass through the network
        '''
        
        return self._main_(x)
