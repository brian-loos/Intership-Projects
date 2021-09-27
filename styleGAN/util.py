'''
    helper functions for the styleGAN models implementation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
'''
    exponential moving average between two tensors
    given (a, b, t) compute

    a = a + (b-a)*t
'''
def EMA(a,b,t):
    return a + (b-a)*t

'''
    pixel normalization
'''
class PixelNorm(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


'''
    helper class that adds bias on the channels of x
'''
class Bias(nn.Module):
    def __init__(self, channels = 512,device = torch.device("cpu")):
        super(Bias, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(channels), requires_grad = True).to(device)
        self.weight.data.uniform_(-.001, .001)
    def forward(self,x):
        return x + self.weight.unsqueeze(-1).unsqueeze(-1)

'''
    class which adds noise at the feature map level of
    an input tensor, x, and noise signal, nz, with
    per channel weighting, b,
'''
class AddNoise(nn.Module):
    def __init__(self,channels = 512,device = torch.device("cpu"), dtype = torch.float32):
        super(AddNoise, self).__init__()
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.zeros(channels,device = self.device, dtype = self.dtype), requires_grad = True)
    def forward(self, x, nz):
        return x + nz*(self.weight.view(-1,1,1))
'''
    class that applies the style mod
'''
class StyleMod(nn.Module):
    def __init__(self,
                latent_dim = 512,
                channels = 512,
                device = torch.device("cpu")
                ):
        super(StyleMod, self).__init__()
        self._embed_style = nn.Linear(latent_dim, 2*channels, bias = True)

    def forward(self,x ,w):
        w = self._embed_style(w).reshape(2,x.shape[0],x.shape[1],1,1)
        return x*w[0] + w[1]





'''
    Blurring layer
    use fractional stride for blured downsample
    use stride > 2 for upsampling
'''
@torch.jit.script
def blur(x, f, stride:int, groups:int, padding:int, upsample: bool):
    if upsample:
        return (float(stride)**2)*F.conv_transpose2d(x, f, stride = stride,
                      groups = groups)
    else:
        if padding > 0:
            pad = (1,1,1,1)
            x = F.pad(x,pad)

        return F.conv2d(x,f, stride = stride, padding = 0, groups = groups)

class Blur2d(nn.Module):
    def __init__(self, filter = 0, channels = 512, stride = 1, padding = 0,
                    device = torch.device('cpu'), dtype = torch.float32):
        super(Blur2d, self).__init__()
        self.device = device
        self.dtype  = dtype
        assert stride > 0
        if stride < 1:
            self.stride = int(1//stride)
            self.upsample = True
        else:
            self.stride = stride
            self.upsample = False
        self.padding = padding
        f = torch.Tensor(filter).to(self.dtype).to(device)
        assert len(f.shape) == 1
        f = f[None, :]*f[:,None]
        f /= f.sum()
        self.f = f.tile(channels, 1, 1,1)

    def forward(self,x):
        '''
            use channels as groups to get correct depthwise convolution behaviour
        '''
        return blur(x,self.f,self.stride, x.shape[1], self.padding, upsample = self.upsample)

'''
    equalized linear layer
'''
class EqualizedLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias_init= 0.0,  lr_mult = 1.0 ,bias = True,
                device = torch.device('cpu'), dtype = torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.randn(output_dim,input_dim,device = self.device, dtype = self.dtype)*lr_mult)
        fan_in = input_dim
        self.scale = lr_mult/np.sqrt(fan_in)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim,device = self.device, dtype = self.dtype).fill_(bias_init))
        else:
            self.bias = None
        self.lr = lr_mult
    def forward(self,input):
        if self.bias is not None:
            return F.linear(input,self.weight*self.scale, bias = self.bias*self.lr)
        else:
            return F.linear(input,self.weight*self.scale)
'''
    equalized convolution
'''
class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel,
                    kernel = 3, stride = 1,
                    padding = 1, bias = True,
                    bias_init = 0.,
                    activation = True,
                    device = torch.device('cpu'),
                    dtype = torch.float32):
        super(EqualConv2d,self).__init__()
        self.device = device
        self.dtype = dtype
        self.stride = stride
        self.padding = 0
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel, kernel,device = self.device, dtype = self.dtype))
        fan_in = (in_channel*kernel*kernel + 1e-8)
        self.scale = 1/np.sqrt(fan_in)
        if padding == 1:
            self.pad = (1,1,1,1)
        else:
            self.pad = None
        if activation:
            self.act = nn.LeakyReLU(.2, inplace = False)
        else:
            self.act = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(1,out_channel, 1, 1,device = self.device, dtype = self.dtype).fill_(bias_init))
        else:
            self.bias = None

    def forward(self,input):
        if self.pad is not None:
            input = F.pad(input, self.pad)
        out = F.conv2d(input, self.weight*self.scale, stride = self.stride, padding = self.padding)
        if self.act is not None:
            out = self.act(out)
        if self.bias is not None:
            out += self.bias
        return out

class EqualConvTranspose2d(nn.Module):
    def __init__(self, in_channel,
                    out_channel,
                    kernel, stride = 1,
                    padding = 1,
                    bias = True,
                    bias_init = 0.0,
                    activation = True,
                    device = torch.device('cpu'),
                    dtype = torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.stride = 1
        self.padding = 2

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel, kernel,device = self.device, dtype = self.dtype))
        fan_in = (in_channel*kernel*kernel + 1e-8)
        self.scale = 1/np.sqrt(fan_in)

        if activation:
            self.act = nn.LeakyReLU(.2, inplace = False)
        else:
            self.act = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(1,out_channel, 1, 1,device = self.device, dtype = self.dtype).fill_(bias_init))
        else:
            self.bias = None

    def forward(self,input):
        out = F.conv2d(input, self.weight*self.scale, stride = self.stride, padding = self.padding)
        if self.act is not None:
            out = self.act(out)
        if self.bias is not None:
            out += self.bias
        return out

class ToRGB(nn.Module):
    def __init__(self,in_channel,im_channels = 3,
                    kernel = 1,
                    bias = False,
                    style_dim = 512,
                    init_bias = 0.0,
                    upsample  = True,
                    blur_fitler = [1,2,1],
                    device = torch.device('cpu'),
                    dtype = torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        if upsample:
            self.upsample = Blur2d(filter = [.5,.5], channels = im_channels, stride = .5, padding = 1,device = self.device, dtype = self.dtype)

        self.conv = EqualConv2d(in_channel,im_channels, kernel = 1, stride = 1, padding = 0,bias = False,activation = False,device = self.device, dtype = self.dtype)

        if bias:
            self.bias = nn.Parameter(torch.zeros(1,im_channels,1,1,device = self.device, dtype = self.dtype).fill_(bias_init))
        else:
            self.bias = None
    def forward(self,x, skip = None):

        x = self.conv(x)
        if self.bias is not None:
            x += self.bias
        if skip is not None:
            skip = self.upsample(skip)
            return x + skip
        return x

class FromRGB(nn.Module):
    def __init__(self, in_channel = 3, out_channel = 32,
                kernel = 1,
                bias = False,
                style_dim = 512,
                device = torch.device('cpu'),
                dtype = torch.float32
                ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.conv = EqualConv2d(in_channel, out_channel, kernel = kernel, stride = 1,padding = 0,bias = False, activation = False,device = self.device, dtype = self.dtype )
        if bias:
            self.bias = nn.Parameter(torch.zeros(1,out_channel,1,1,device = self.device, dtype = self.dtype).fill_(0.0))
        else:
            self.bias = None

    def forward(self,x):
        x = self.conv(x)
        if self.bias is not None:
            x += self.bias
        return x



class ModulatedConv2d(nn.Module):
    def __init__(self,in_channel,out_channel,
                kernel = 3,
                stride = 1,
                padding = 1,
                style_dim = 512,
                upsample = False,
                blur_filter = [1,2,1],
                fused = True,
                demod = True,
                bias_init = 1.0,
                device = torch.device('cpu'),
                dtype = torch.float32
                ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        fan_in = (in_channel*kernel*kernel + 1e-8)
        self.scale = 1/np.sqrt(fan_in)
        self.stride = stride
        self.padding = padding
        if padding > 0:
            self.pad = (1,1,1,1)
        else:
            self.pad = None
        self.demodulate = demod
        self.weight = nn.Parameter(torch.randn(out_channel,in_channel, kernel, kernel,device = self.device, dtype = self.dtype))
        self.mod_style = EqualizedLinear(style_dim,in_channel, bias_init = bias_init, lr_mult = 1.0, bias = True,device = self.device, dtype = self.dtype)



        if upsample:
            self.blur = Blur2d(filter = blur_filter, channels = out_channel, stride = 1, padding = 1,device = self.device, dtype = self.dtype)
        else:
            self.blur = None


        self.fused = fused

    def forward(self,input, style):
        s = input.shape #batchsize x inchannels x h x w
        ws = self.weight.shape
        if not self.fused:
            '''
                modulate weights
            '''
            mod_weight = self.scale*self.weight
            style = self.mod_style(style)
            input = input*style.view((s[0], s[1], 1,1))
            '''
                demod weights
            '''
            if self.demodulate:
                demod_weight = mod_weight.view(1, -1, s[1], 3,3)*style.view(s[0],1,s[1], 1,1 )
                demod_weight = (demod_weight.square().sum((2,3,4)) + 1e-8).rsqrt()


            if self.blur is not None:
                input = self.blur(input)
                out = F.conv_transpose2d(input, mod_weight.transpose(0,1),
                                        stride = self.stride, padding = self.padding, output_padding = self.padding)
            else:
                if self.pad is not None:
                    input = F.pad(input,self.pad) #adds reflection padding
                    self.padding = 0
                out = F.conv2d(input, mod_weight,
                                stride = self.stride, padding = self.padding)

            if self.demodulate:
                out = out*(demod_weight.view(s[0], out.shape[1],1,1))
            return out
        else:#fuse the style into the weights
            '''
                modulate weights
            '''
            style = self.mod_style(style)

            if self.blur is not None:
                res = s[2]*2
            else:
                res = s[3]


            mod_weight = (self.scale*self.weight*style.view(s[0],-1,ws[1],1,1))
            if self.demodulate:
                mod_weight = mod_weight*((mod_weight.pow(2).sum((2,3,4),keepdims = True) + 1e-8).rsqrt())
            if self.blur is not None:
                out = F.conv_transpose2d(input.view(1,s[0]*s[1],s[2],s[3]),mod_weight.transpose(1,2).reshape(s[0]*ws[1], ws[0],ws[2], ws[3]),stride = self.stride, padding = self.padding,output_padding = self.padding, groups = s[0]).view(s[0],-1, res,res)
                out = self.blur(out)
            else:
                if self.pad is not None:
                    input = F.pad(input,self.pad)
                    self.padding = 0
                out  = F.conv2d(input.view(1,s[0]*s[1],s[2]+2,s[3]+2),mod_weight.view(s[0]*ws[0], ws[1],ws[2], ws[3]),stride = self.stride, padding = self.padding, groups = s[0]).view(s[0], -1, s[2],s[3])
            return out

        return

'''
    moduated conv --> activation --> bias
'''
class StyleConvLayer(nn.Module):
    def __init__(self,
                in_channel,out_channel,
                style_dim = 512,
                upsample = False,
                noise = False,
                device = torch.device('cpu'),
                dtype = torch.float32
                ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        if upsample:
            self.conv = ModulatedConv2d(in_channel, out_channel, stride = 2, padding = 1,
                                        style_dim = style_dim,
                                        upsample = True,device = self.device, dtype = self.dtype)
        else:
            self.conv = ModulatedConv2d(in_channel, out_channel, stride = 1, padding = 1,
                                        style_dim = style_dim,
                                        upsample = False,device = self.device, dtype = self.dtype)
        if noise:
            self.add_noise = AddNoise(out_channel,device = self.device, dtype = self.dtype)
        else:
            self.add_noise = None
        self.activation = nn.LeakyReLU(.2,inplace = False)


    def forward(self,input,style,noise = None):

        out = self.conv(input,style)
        if self.add_noise is not None:
            out = self.add_noise(out,noise)
        out = self.activation(out)
        return out




'''
    Residual blocks for discriminator
    residual connection from layer in ---> out channel
'''
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel,
                bias = True,
                bias_init = 0.0,
                activation = True,
                blur_filter = [1,2,1],
                device = torch.device('cpu'),
                dtype = torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.conv1 = nn.Sequential(EqualConv2d(in_channel, in_channel, stride = 1,
                                bias = bias, bias_init = bias_init, activation = activation,device = self.device, dtype = self.dtype)
                                )
        self.skip_conv = nn.Sequential(
                                Blur2d(filter = blur_filter, channels = in_channel, stride = 1, padding = 1,device = self.device, dtype = self.dtype),
                                EqualConv2d(in_channel, out_channel, stride = 2, bias = False, bias_init = bias_init, activation = False,device = self.device, dtype = self.dtype )
                                        )
        self.conv2 = nn.Sequential(
                                Blur2d(filter = blur_filter, channels = in_channel, stride =1 ,  padding = 1,device = self.device, dtype = self.dtype),
                                EqualConv2d(in_channel, out_channel, stride = 2,bias_init = bias_init,
                                bias = bias, activation = activation,device = self.device, dtype = self.dtype)
                                )

    def forward(self,input):
        skip = self.skip_conv(input)
        out = self.conv2(self.conv1(input))
        return out + skip
