import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.models as visionmodels
import torchvision.transforms as transforms
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets
from collections import *
import os
from torchvision.io import read_image
from progress.bar import Bar

'''
    custom files
'''
import models, util

generator = models.styleGAN_generator(target_im_size = 512, im_channels = 3)
discriminator = models.styleGAN_discriminator(target_im_size = 512,
                                                in_channels = 3)
print(generator)
print(discriminator)
model_parameters = filter(lambda p: p.requires_grad, generator.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('number of generator parameters: ', params)
model_parameters = filter(lambda p: p.requires_grad, discriminator.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('number of discriminator parameters:', params)
latent = torch.rand(16, 512)
out = generator(latent)
print('generatro out shape ',out.shape)
scores = discriminator(out)
print('discriminator shape out', scores.shape)
