from util import *
from models_v2 import *

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
batch_size = 16
latent_size = 512
input_channels = 13
output_channels = 7
resolution = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
x = (torch.ones(batch_size,input_channels,resolution,resolution)*torch.rand(input_channels).view(1,-1,1,1)).cuda()
style = torch.randn(batch_size,latent_size).cuda()
skip = torch.randn(batch_size,3,4,4).cuda()

lin = ModulatedConv2d(input_channels,output_channels, stride = 1,padding= 1, upsample = False, fused = True , device = device, dtype = torch.float32).to(device)
#lin = ModulatedConv2d(input_channels,output_channels,  stride = 2, padding = 1,
#                    upsample = True, fused = True).cuda()
#lin = ToRGB(input_channels,upsample = True).cuda()

#noise = torch.randn(batch_size, 1,resolution*2,resolution*2).cuda()

#generator = styleGenerator(target_im_size = 256,latent_dim = latent_size).cuda()
#discriminator = styleDiscriminator(target_im_size = 256 ,latent_size = latent_size).cuda()
test_out, out2 = lin(x,style)
print(test_out.shape,out2.shape)
print(test_out==out2)
#print(test_out)
#print(discriminator(test_out))
