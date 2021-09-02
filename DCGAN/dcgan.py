##These are my personal machine learning import statements, they cover most of what I regularly need for CV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

'''
    Because Unet consists entirely of skip connections, we cannot package it
    conveniently using pytorch to my knowledge. Either way, this is an
    implementation of the version of U-net from the original paper
'''
'''
    DCGAN network implemented as described in the original work by
    Radford et al.
'''
class Discriminator(nn.Module):

    def __init__(self, featmap_dim=64, n_channel=1):
        super(Discriminator, self).__init__()
        self.featmap_dim = featmap_dim

        '''
            need to convolve and downsample from 32x32 to 1x1, I do this in
            4 convolutions. 3 padded convolutions reduce the image resolution
            in half. The final 4x4 convolution has no padding and downsamples
            the 4x4 input to a 1x1 output
        '''

        self.main = nn.Sequential(
                    #first pass, in dim 1 x 32 x 32
                    nn.Conv2d(1,featmap_dim,4,2,1,bias = False),
                    nn.LeakyReLU(0.2,inplace = True),
                    #second pass, ft.dim*2 x 16 x 16
                    nn.Conv2d(featmap_dim,featmap_dim*2,4,2,1,bias = False),
                    nn.BatchNorm2d(featmap_dim*2),
                    nn.LeakyReLU(0.2,inplace = True),
                    #third pass, ft.dim*4 x 8 x 8
                    nn.Conv2d(featmap_dim*2,featmap_dim*4,4,2,1,bias = False),
                    nn.BatchNorm2d(featmap_dim*4),
                    nn.LeakyReLU(0.2,inplace = True),
                    #third pass, ft.dim*4 x 8 x 8
                    nn.Conv2d(featmap_dim*4,featmap_dim*8,4,2,1,bias = False),
                    nn.BatchNorm2d(featmap_dim*8),
                    nn.LeakyReLU(0.2,inplace = True),
                    #fourth pass, ft.dim*2 x 4 x 4

                    nn.Conv2d(featmap_dim*8,1,4,1,0,bias = False),
                    nn.Sigmoid()
                    )

    def forward(self, x):
        """
        Strided convulation layers,
        Batch Normalization after convulation but not at input layer,
        LeakyReLU activation function with slope 0.2.
        """
        return self.main(x)


class Generator(nn.Module):
    def __init__(self, featmap_dim=64, n_channel=1, noise_dim=100):
        super(Generator, self).__init__()
        self.featmap_dim = featmap_dim

        '''
            The Generator needs to take in an input latent code and
            upconvolve it into a 32x32 output image. This is achieved
            with upconvolutions, batchnorms, and ReLU activations
        '''
        ##first pass, project the input
        #self.project = nn.Sequential(nn.Linear(noise_dim, featmap_dim*4*4), nn.ReLU(True))
        #then perform convolutions
        #self.fc = nn.Sequential(nn.Linear(noise_dim, 512*2*2, bias = False), nn.ReLU(True))
        self.main = nn.Sequential(
                    ##input tensor is 512 x 4 x 4
                    nn.ConvTranspose2d(noise_dim,featmap_dim*8,4,1,0,bias = False),
                    nn.BatchNorm2d(featmap_dim*8),
                    nn.ReLU(True),
                    ##second pass, state size 256 x 8 x 8
                    nn.ConvTranspose2d(featmap_dim*8,featmap_dim*4,4,2,1,bias = False),
                    nn.BatchNorm2d(featmap_dim*4),
                    nn.ReLU(True),
                    ##third pass, state size 128 x 16 x 16
                    nn.ConvTranspose2d(featmap_dim*4,featmap_dim*2,4,2,1,bias = False),
                    nn.BatchNorm2d(featmap_dim*2),
                    nn.ReLU(True),
                    ##fourth pass, state size 1 x 32 x 32
                    nn.ConvTranspose2d(featmap_dim*2,featmap_dim,4,2,1,bias = False),
                    nn.BatchNorm2d(featmap_dim),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(featmap_dim,1,4,2,1,bias = False),
                    nn.Tanh()
                    )
    def forward(self, x):
        """
        Project noise to featureMap * width * height,
        Batch Normalization after convulation but not at output layer,
        ReLU activation function.
        """
        #x = self.fc(x)
        #y = torch.reshape(x, (x.shape[0], x.shape[1],1,1))
        return self.main(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0,1.0)
