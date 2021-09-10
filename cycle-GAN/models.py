
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
from collections import *



'''
    need to create a 70x70 patch gan network
'''
class conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, has_BN = True,
                leaky = False, has_dropout = False,
                mode = 'downsample'):
        super(conv_layer, self).__init__()
        relu_leak = .2
        dropout_rate = .5
        '''
            all convolutions are stride 2 with kernel of size 4x4 and padding
        '''
        stride = 2
        filter = 4
        padding = 1
        out_padding = 0
        layers = []
        if mode == 'upsample':
            layers.append(nn.ConvTranspose2d(in_channels,out_channels,filter, stride = stride,
                                            padding = padding, output_padding = out_padding,
                                            bias = False))
        elif mode == 'downsample':
            layers.append(nn.Conv2d(in_channels,out_channels,filter,stride = stride,
                                            padding = padding,
                                            bias = False))
        else:
            print('Invalid convolution layer mode: upsample/downsample')
            return None

        if has_BN:
            layers.append(nn.InstanceNorm2d(out_channels))

        if has_dropout:
            layers.append(nn.Dropout2d(p = dropout_rate, inplace = True))
        if leaky:
            layers.append(nn.LeakyReLU(relu_leak, True))
        else:
            layers.append(nn.ReLU(True))
        self._main_ = nn.Sequential(*layers)
    def forward(self, x):
        return self._main_(x)

class Patch_discriminator(nn.Module):
    def __init__(self,im_channels, im_size):
        super(Patch_discriminator, self).__init__()
        self._patch_size = 70 #fixed patch extraction size
        self.full_im_size = im_size
        self._patches = self._get_patchs()
        layers = []
        features = 64
        for i in range(4):
            if i == 0:
                layers.append(conv_layer(im_channels, features, has_BN = False, leaky = True, has_dropout = False, mode = 'downsample'))
            if i == 3:
                layers.append(conv_layer(features, features*2, has_BN = False, leaky = True, has_dropout = False, mode = 'downsample'))
            else:
                layers.append(conv_layer(features, features*2, has_BN = True, leaky = True, has_dropout = False, mode = 'downsample'))
                features *= 2

        self._main_ = nn.Sequential(*layers)
        self._final_conv = nn.Conv2d(1024,1,2,1, padding  =0, bias= False)

    def forward(self,im):
        out = 0

        for a,b,c,d in self._patches:
            out += self._final_conv(self._main_(im[:,:,a:b,c:d])).view(-1)
        out /= len(self._patches)
        return out

    def _get_patchs(self):
        spacing = np.ceil(self.full_im_size/self._patch_size) #returns approximate spacing
        bounds = []
        a = 0
        b = self._patch_size
        while b < self.full_im_size:
            bounds.append([a,b])
            a += self._patch_size
            b += self._patch_size
            if b >= self.full_im_size:
                diff = self.full_im_size - b
                b += diff
                a += diff
                bounds.append([a,b])
                break
        patches = []
        for u in bounds:
            for v in bounds:
                patches.append([u[0],u[1], v[0],v[1]])
        return patches

'''
    generator in the style of cycle GAN
'''
class residual_block(nn.Module):
    def __init__(self, channels = 128):
        super(residual_block, self).__init__()
        self._main_ = nn.Sequential(
                                    nn.Conv2d(128,128, 3, stride = 1, padding = 1, bias = False,padding_mode = 'reflect'),
                                    nn.InstanceNorm2d(128),
                                    nn.ReLU(True),
                                    nn.Conv2d(128,128, 3, stride = 1, padding = 1, bias = False,padding_mode = 'reflect'),
                                    nn.InstanceNorm2d(128)
                                    )
    def forward(self,x):
        y = x
        x = self._main_(x)
        return x+y
class cycleGAN_Generator(nn.Module):
    def __init__(self,in_channels = 3,num_blocks = 6):
        super(cycleGAN_Generator,self).__init__()
        self.conv1 = nn.Sequential(
                                    nn.Conv2d(3,32,7,stride = 1, padding = 3, bias= False,padding_mode = 'zeros'),
                                    nn.InstanceNorm2d(32),
                                    nn.ReLU(True)
                                    )
        self.conv2 = nn.Sequential(
                                    nn.Conv2d(32,64,3,stride = 2, padding = 1, bias= False,padding_mode = 'zeros'),
                                    nn.InstanceNorm2d(64),
                                    nn.ReLU(True)
                                    )
        self.conv3 = nn.Sequential(
                                    nn.Conv2d(64,128,3,stride = 2, padding = 1, bias= False,padding_mode = 'zeros'),
                                    nn.InstanceNorm2d(128),
                                    nn.ReLU(True)
                                    )
        self.upconv1 = nn.Sequential(
                                    nn.ConvTranspose2d(128,64,3, stride = 2, padding = 1,output_padding=1,  bias = False,padding_mode = 'zeros'),
                                    nn.InstanceNorm2d(64),
                                    nn.ReLU(True),
                                    nn.Conv2d(64,64,3, stride = 1, padding = 1,  bias = False,padding_mode = 'zeros'),
                                    nn.InstanceNorm2d(64),
                                    nn.ReLU(True)
                                    )
        self.upconv2 = nn.Sequential(
                                    nn.ConvTranspose2d(64,32,3, stride = 2, padding = 1,output_padding=1,  bias = False,padding_mode = 'zeros'),
                                    nn.InstanceNorm2d(32),
                                    nn.ReLU(True),
                                    nn.Conv2d(32,32,3, stride = 1, padding = 1,  bias = False,padding_mode = 'zeros'),
                                    nn.InstanceNorm2d(32),
                                    nn.ReLU(True)
                                    )
        self.upconv3 = nn.Sequential(
                                    nn.Conv2d(32,3,7,stride = 1, padding = 3, bias= False,padding_mode = 'zeros')
                                    )
        layers = []
        for i in range(num_blocks):
            layers.append(residual_block(channels =128))
        self.residual = nn.Sequential(*layers)
        self._main_ = nn.Sequential(
                                    self.conv1,
                                    self.conv2,
                                    self.conv3,
                                    self.residual,
                                    self.upconv1,
                                    self.upconv2,
                                    self.upconv3
                                    )

    def forward(self,x):
        return torch.tanh(self._main_(x))

LossOutput = namedtuple(
    "LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3


class LossNetwork(nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork,
              self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
                                             '3': "relu1_2",
                                             '8': "relu2_2",
                                             '15': "relu3_3",
                                             '22': "relu4_3"
                                             }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
                #print(name, self.layer_name_mapping[name], output.keys())
        return LossOutput(**output)
