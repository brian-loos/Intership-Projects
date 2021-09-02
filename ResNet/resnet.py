'''
    This is script containing the classes for defining Resnet18 and ResNet34
    These are created based on the torchvision implementation. I rewrote them
    myself for educational purposes and to make clearer to myself. For anyone
    wanting to implement ResNet, I would suggest using the version included in
    torchvision libraries, not this.
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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

def conv_3x3(in_channels,out_channels):
    if in_channels == out_channels:#normal convolution
        return nn.Conv2d(in_channels, out_channels,3, stride = 1,
                            padding = 1, bias = False)
    else: #downsampling convolution
        return nn.Conv2d(in_channels, out_channels, 3, stride =2,
                            padding = 1, bias = False)

'''
    this class creates a residual block given the number of in channels
    and the number of out channels. Given this it infers what type of
    block it is and how to set up the skip connection
    Only inputs to this are number of input channels and number of output
    channels
'''
class ResBlock(nn.Module):
    def __init__(self,in_channels, out_channels):
        '''
            We need to define the residual blocks for resnet
            For basic ResNet(18/34), these are simple blocks
            consisting of 2 convolutions and have BN and
            ReLu inbetween two convolutional layers with a
            skip connection around the entire block
        '''
        '''
            There are two kinds of basic blocks:
                (1) in/out channel has varying sizes: first
                    layer performs a downsampling with a
                    stride 2 convolution
                (2) in/out channels match: there is no downsampling
                    and the skip connection is the
                    identity mapping
        '''

        super(ResBlock,self).__init__()
        self._rescale_block =False
        if in_channels != out_channels:
            '''
                set flag and create downsampling block if this is a
                block which performs downsampling
            '''
            self._rescale_block = True
            self.downsample = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,
                                            stride= 2, bias = False),
                                            nn.BatchNorm2d(out_channels)
                                            )
        self.conv_1 = conv_3x3(in_channels, out_channels)
        self.conv_2 = conv_3x3(out_channels, out_channels)
        self._main_ = nn.Sequential(self.conv_1,
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(True),
                        self.conv_2,
                        nn.BatchNorm2d(out_channels))

    def forward(self,x):
        y = x #copy identity
        if self._rescale_block: #need to map the identity by a downsampling operation
            y = self.downsample(y)

        x = self._main_(x)
        x = x + y
        x = F.relu(x) #final activation
        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes = 1000,in_channels = 1):
        super(ResNet18,self).__init__()
        #we define our CNN layers
        '''
            This network takes in 224x224 crops from 256x256 RGB images
            Expect input image tensors of size 3 x 224 x 224
        '''
        '''
            The first layer of ResNet18, two stride 2 operations produce
            4x downsampling on the input
            Takes in tensor of 3 x 224 x 224
            Output tensor ---> 64 x 56 x 56
        '''

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_channels,64,7,stride = 2,padding = 1,bias = False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(True),
                        nn.MaxPool2d(3, stride = 2, padding =1)
                        )

        self.conv2_x = self._make_layer(64,64,num_blocks = 2)
        self.conv3_x = self._make_layer(64,128,num_blocks = 2)
        self.conv4_x = self._make_layer(128,256,num_blocks = 2)
        self.conv5_x = self._make_layer(256,512,num_blocks = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        '''
            initialization taken from torchvision documentation
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        '''
            connected all the resnet layers
        '''
        self._main_ = nn.Sequential(self.conv_1, self.conv2_x,
                                    self.conv3_x, self.conv4_x,
                                    self.conv5_x, self.avgpool
                                    )
    '''
        to make the layers, we have to know how many blocks there are
        what the input size to the layer is and what the output size
        of the layer is
        Given an particular input number of channels and output number
        of channels, the block should downsample and output a tensor
        with half the width and height but the desired number of output
        channels
        Each layers also needs to know how many blocks it should consist
        of. For ResNet18, this number is 2 blocks per layer

    '''
    def _make_layer(self, in_channels, out_channels, num_blocks =2):
        blocks = []
        for i in range(num_blocks):
            if i == 0:
                blocks.append(ResBlock(in_channels, out_channels))
            else:
                blocks.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*blocks)


    def forward(self,x):
        '''
            Here we define our forward pass
        '''
        x = self._main_(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc(x)
        return x

class ResNet34(nn.Module):
    def __init__(self, num_classes = 1000):
        super(ResNet34,self).__init__()
        #we define our CNN layers
        '''
            This network takes in 224x224 crops from 256x256 RGB images
            Expect input image tensors of size 3 x 224 x 224
        '''
        '''
            The first layer of ResNet18, two stride 2 operations produce
            4x downsampling on the input
            Takes in tensor of 3 x 224 x 224
            Output tensor ---> 64 x 56 x 56
        '''

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(3,64,7,stride = 2,padding = 1,bias = False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(True),
                        nn.MaxPool2d(3, stride = 2, padding =1)
                        )

        self.conv2_x = self._make_layer(64,64,num_blocks = 3)
        self.conv3_x = self._make_layer(64,128,num_blocks = 4)
        self.conv4_x = self._make_layer(128,256,num_blocks = 6)
        self.conv5_x = self._make_layer(256,512,num_blocks = 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 7*7, num_classes)

        '''
            initialization taken from torchvision documentation
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        '''
            connected all the resnet layers
        '''
        self._main_ = nn.Sequential(self.conv_1, self.conv2_x,
                                    self.conv3_x, self.conv4_x,
                                    self.conv5_x, self.avgpool
                                    )
    '''
        to make the layers, we have to know how many blocks there are
        what the input size to the layer is and what the output size
        of the layer is
        Given an particular input number of channels and output number
        of channels, the block should downsample and output a tensor
        with half the width and height but the desired number of output
        channels
        Each layers also needs to know how many blocks it should consist
        of. For ResNet18, this number is 2 blocks per layer

    '''
    def _make_layer(self, in_channels, out_channels, num_blocks =2):
        blocks = []
        for i in range(num_blocks):
            if i == 0:
                blocks.append(ResBlock(in_channels, out_channels))
            else:
                blocks.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*blocks)


    def forward(self,x):
        '''
            Here we define our forward pass
        '''
        x = self._main_(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc(x)
        return x

class simple_ResNet(nn.Module):
    def __init__(self, num_classes = 1000,in_channels = 1, num_blocks = 3):
        super(simple_ResNet,self).__init__()
        #we define our CNN layers
        '''
            This network takes in 32x32 crops from padded 32x32 images
            Expect input image tensors of size image_channels x 32 x 32
        '''
        '''
            The first layer of ResNet18, two stride 2 operations produce
            4x downsampling on the input
            Takes in tensor of 3 x 224 x 224
            Output tensor ---> 64 x 56 x 56
        '''
        '''
            first block is the same in every network,
            3x3 stride 1 input convolution
        '''
        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_channels,16,3,stride = 1,padding = 1,bias = False),
                        nn.BatchNorm2d(16),
                        nn.ReLU(True),
                        )
        '''
            input is a 16 x 32 x 32 tensor, with no downsampling
        '''
        self.conv2_x = self._make_layer(16,16,num_blocks = num_blocks)
        '''
            input is a 16 x  32 x 32 tensor
        '''
        self.conv3_x = self._make_layer(16,32,num_blocks = num_blocks)
        '''
            input is a 32 x 16 x 16 tensor
        '''
        self.conv4_x = self._make_layer(32,64,num_blocks = num_blocks)
        '''
            input is a 64 x 8 x 8 tensor
        '''
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

        '''
            initialization taken from torchvision documentation
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        '''
            connected all the resnet layers
        '''
        self._main_ = nn.Sequential(self.conv_1, self.conv2_x,
                                    self.conv3_x, self.conv4_x
                                    )
    '''
        to make the layers, we have to know how many blocks there are
        what the input size to the layer is and what the output size
        of the layer is
        Given an particular input number of channels and output number
        of channels, the block should downsample and output a tensor
        with half the width and height but the desired number of output
        channels
        Each layers also needs to know how many blocks it should consist
        of. For ResNet18, this number is 2 blocks per layer

    '''
    def _make_layer(self, in_channels, out_channels, num_blocks =2):
        blocks = []
        for i in range(num_blocks):
            if i == 0:
                blocks.append(ResBlock(in_channels, out_channels))
            else:
                blocks.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*blocks)


    def forward(self,x):
        '''
            Here we define our forward pass
        '''
        x = self._main_(x)
        x = self.avgpool(x)

        x = torch.flatten(x, start_dim = 1)
        x = self.fc(x)
        return x
