import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import util

class styleGenerator(nn.Module):
    def __init__(self,
                target_im_size = 512,           #target image size
                im_channels = 3,
                latent_dim = 512,               #latent dimension
                label_dim = 0,               #dimension of labels
                feat_maps = 8192,               #used for computing number of feature maps
                max_features = 512,            #minimum number of feature maps
                track_W = True,                 #track center of mass of latent space
                dlatent_avg_beta = .995,         #EMA weight for tracking center of mass
                use_style_mixing = True,        #use style mix regularization
                prob_style_mix = .5,
                use_truncation = True,          #use truncation trick
                truncation_cutoff_layer = 4,    #truncation cutoff
                truncation_weight = .7,        #truncation weighting,
                device = torch.device('cpu'),
                dtype = torch.float32
                ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self._resolution = int(np.log2(target_im_size))
        assert target_im_size == 2**self._resolution and (target_im_size > 4)
        self._num_layers = self._resolution*2 -3
        self._track_W = track_W
        self._dlatent_avg_beta = dlatent_avg_beta
        if track_W: self._dlatent_avg = torch.zeros(latent_dim,dtype = self.dtype, device = self.device)
        self._use_style_mixing = use_style_mixing
        if use_style_mixing:
            self._prob_style_mix = prob_style_mix
            self._layer_index = torch.arange(0,self._num_layers,dtype = self.dtype, device = self.device)
        if use_truncation:
            self._use_truncation = use_truncation
            self._truncation_cutoff_layer = truncation_cutoff_layer
            self._truncation_weight = truncation_weight

        self.mapping_net = MappingNetwork(latent_dim = latent_dim, dtype = self.dtype,device = self.device)

        self.synthesis_net = SynthesisNetwork(target_im_size = target_im_size, latent_dim = latent_dim,dtype = self.dtype, device = self.device)

    def forward(self,latent, label = None):
        s = latent.shape
        dlatent = self.mapping_net(latent,label)
        '''
            compute EMA center of mass of latent distribution W
        '''
        if self._track_W:#use exponential moving average to update latent
            batch_avg = dlatent.detach().mean(dim = 0, keepdim = True)
            self._dlatent_avg = util.EMA(self._dlatent_avg, batch_avg, self._dlatent_avg_beta)
        '''
        implement truncation trick
        '''
        #compute truncated value
        if self._use_truncation:
            dlatent_truncated = util.EMA(dlatent, self._dlatent_avg, self._truncation_weight)
        else:
            dlatent_truncated = None

        '''
            compute style mixing
        '''
        '''
            first need to broadcast dlatent correctly
        '''
        if self._use_style_mixing and (np.random.rand() <= self._prob_style_mix):
            latent_2 = torch.randn(s[0], s[1], dtype = self.dtype,device = self.device)
            dlatent2 = self.mapping_net(latent_2, label)
            while True:
                mix_cutoff = np.random.randint(0,s[1])#how many channels to mix
                if mix_cutoff > self._truncation_cutoff_layer:
                    break
        else:
            dlatent2 = None
            mix_cutoff = None
        images = self.synthesis_net(dlatent,latent_truncated = dlatent_truncated,truncation_cutoff = self._truncation_cutoff_layer, mix_latent = dlatent2,mix_cutoff =  mix_cutoff)
        return images

'''
    8 layer equalized MLP with learning rate mult of .01
'''

class MappingNetwork(nn.Module):
    def __init__(self,
                target_im_size = 512,           #target image size
                latent_dim = 512,               #latent dimension
                label_dim = 0,                  #dimension of labels
                num_layers = 8,                 #number of layer
                num_weights = 512,               #number of weights per layer
                normalize_latent = True,        #whether or not to normalize the latent z
                lr_mult = .01,
                bias_init = 0.,
                device = torch.device('cpu'),
                dtype = torch.float32
                ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        if normalize_latent:
            self.norm = util.PixelNorm(device = self.device,dtype = self.dtype)
        else:
            self.norm = None
        if label_dim:
            self._embed_label = util.EqualizedLinear(label_dim, latent_dim, lr_mult = lr_mult,  bias = False,device = self.device,dtype = self.dtype)
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers += [util.EqualizedLinear(latent_dim + label_dim,latent_dim, lr_mult = lr_mult,bias_init = bias_init,bias = True,device = self.device,dtype = self.dtype)]
                layers += [nn.LeakyReLU(.2, inplace = False)]
            else:
                layers += [util.EqualizedLinear(latent_dim, latent_dim, lr_mult = lr_mult,bias_init = bias_init,bias = True,device = self.device,dtype = self.dtype)]
                layers += [nn.LeakyReLU(.2, inplace = False)]
        self.main = nn.Sequential(*layers)

    def forward(self,latent,label = None):
        if label:
            label = self._embed_label(label)
            latent = torch.cat((latent,label),dim=1)
        if self.norm is not None:
            latent = self.norm(latent)
        return self.main(latent)

class SynthesisNetwork(nn.Module):
    def __init__(self,
                target_im_size = 512,           #target image size
                im_channels = 3,
                latent_dim = 512,               #latent dimension
                feat_maps = 8192,               #used for computing number of feature maps
                max_features = 512,            #minimum number of feature maps
                feat_map_decay = 1.0,           #rate of decay for feature map sizes, leave at 1
                num_layers = 8,
                device = torch.device('cpu'),
                dtype = torch.float32
                ):
        super().__init__()
        self._resolution = int(np.log2(target_im_size))
        self.num_layers = num_layers
        self._feat_maps = feat_maps
        self._max_features = max_features
        self._feat_map_decay = feat_map_decay
        self.device = device
        self.dtype = dtype
        self.convs = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        self.convs += [util.StyleConvLayer(self.num_features(2), self.num_features(2),upsample = False,noise = False, style_dim = latent_dim,device = self.device, dtype = self.dtype)]
        for i in range(2,self._resolution):
            self.to_rgb += [util.ToRGB(self.num_features(i),3,upsample = True, style_dim = latent_dim,device = self.device, dtype = self.dtype)]
            self.convs += [util.StyleConvLayer(self.num_features(i),self.num_features(i+1), upsample = True,noise = True, style_dim = latent_dim,device = self.device, dtype = self.dtype)]
            self.convs += [util.StyleConvLayer(self.num_features(i+1), self.num_features(i+1), upsample = False,noise = True, style_dim = latent_dim,device = self.device, dtype = self.dtype)]
        self.to_rgb += [util.ToRGB(self.num_features(self._resolution),3,upsample = True, style_dim = latent_dim,device = self.device, dtype = self.dtype)]

    def forward(self, latent, latent_truncated=None, truncation_cutoff=None, mix_latent=None, mix_cutoff=None):
        s = latent.shape
        self.c = torch.ones(s[0],512,4,4,device = self.device,dtype = self.dtype)
        if (truncation_cutoff is not None) and (mix_cutoff is not None):
            assert truncation_cutoff < mix_cutoff

        if truncation_cutoff is not None:
            input = self.convs[0](self.c,latent_truncated)
        elif mix_cutoff is not None:
            if mix_cutoff == 0:
                input = self.convs[0](self.c,mix_latent)
            else:
                input = self.convs[0](self.c,latent)
        else:
            input = self.convs[0](self.c,latent)
        if mix_cutoff is None:
            mix_cutoff = 1000
        for i in range(0,self._resolution-2):#resolution starts at 8x8 after this point
            skip = self.to_rgb[i](input) if i == 0 else self.to_rgb[i](input,skip)
            if (2*i + 1 < truncation_cutoff) and (2*i+ 1 < mix_cutoff):
                input = self.convs[(2*i)+1](input, latent_truncated, self.create_noise(batch_size = s[0],resolution = i+3,device =self.device,dtype = self.dtype))
            elif  (2*i+ 1 < mix_cutoff):
                input = self.convs[(2*i)+1](input, latent,  self.create_noise(batch_size = s[0],resolution = i+3,device =self.device,dtype = self.dtype))
            else:
                input = self.convs[(2*i)+1](input, mix_latent,  self.create_noise(batch_size = s[0],resolution = i+3,device =self.device,dtype = self.dtype))

            if (2*i + 2 < truncation_cutoff) and (2*i+ 2 < mix_cutoff):
                input = self.convs[(2*i)+2](input,latent_truncated, self.create_noise(batch_size = s[0],resolution = i+3,device =self.device,dtype = self.dtype))
            elif (2*i+ 2 < mix_cutoff):
                input = self.convs[(2*i)+2](input,latent, self.create_noise(batch_size = s[0],resolution = i+3,device =self.device,dtype = self.dtype))
            else:
                input = self.convs[(2*i)+2](input,mix_latent, self.create_noise(batch_size = s[0],resolution = i+3,device =self.device,dtype = self.dtype))

        return self.to_rgb[-1](input,skip)

    def create_noise(self, batch_size,resolution,device,dtype):
        return torch.randn(batch_size,1,2**resolution,2**resolution, device = device, dtype = dtype)

    def num_features(self, stage):
        return int(min(self._feat_maps//2**(stage*self._feat_map_decay), self._max_features))

'''
    residual block based discriminator
'''

class styleDiscriminator(nn.Module):
    def __init__(self,
                target_im_size = 512,
                in_channels = 3,
                label_size = 0,
                latent_size = 512,
                feat_maps = 8192,
                feat_map_decay = 1.0,
                max_features= 512,
                minibatch_groups = 4,
                minibatch_channels =1,
                device = torch.device('cpu'),
                dtype = torch.float32
                ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self._feat_maps = feat_maps
        self._feat_map_decay = feat_map_decay
        self._max_features= max_features
        self.minibatch_group_size = minibatch_groups
        self.minibatch_num_channels = minibatch_channels
        self.res = int(np.log2(target_im_size))
        self.from_rgb = util.FromRGB(in_channels, self.num_features(self.res),device = self.device, dtype = self.dtype)

        self.num_layers = self.res - 2
        layers = []
        for i in range(self.res, 2, -1):#downsample to 4x4
            layers += [util.ResidualBlock(self.num_features(i), self.num_features(i-1),bias_init = 0.,device = self.device, dtype = self.dtype)]

        layers
        self.main = nn.Sequential(*layers)

        layers = []
        self.final_conv = util.EqualConv2d(self.num_features(2)+1, self.num_features(2),device = self.device, dtype = self.dtype)
        layers += [util.EqualizedLinear(self.num_features(2)*16, self.num_features(2), bias = True,device = self.device, dtype = self.dtype)]
        layers += [nn.LeakyReLU(.2, inplace = False)]
        layers += [util.EqualizedLinear(self.num_features(2),1, bias = True,device = self.device, dtype = self.dtype)]
        self.output_layer = nn.Sequential(*layers)

    def forward(self,image, label = None):
        x = self.from_rgb(image)
        x = self.main(x)
        if self.minibatch_group_size <= image.shape[0]:
            x = self.minibatch_std_dev(x, group_size = self.minibatch_group_size, new_channels = self.minibatch_num_channels)
        else:
            s = x.shape
            zeros = torch.zeros(s[0], 1, s[2], s[3],dtype = self.dtype, device = self.device)
            x = torch.cat((x,zeros),dim =1 )
        out = self.final_conv(x).flatten(start_dim = 1)
        out = self.output_layer(out)

        if label:
            out = torch.sum(out*label,axis = 1)

        return out

    def num_features(self,stage):
        return int(min(self._feat_maps//2**(stage*self._feat_map_decay),self._max_features))

    '''
        compute minibatch std deviation of a given input
    '''
    def minibatch_std_dev(self,x, group_size = 4, new_channels = 1):
        s = x.shape
        y = torch.reshape(x, (group_size, s[0]//group_size,new_channels, s[1]//new_channels, s[2], s[3] )) # unwrap to G x M x n x c x H x W
        y -= y.mean(axis = 0, keepdim=True)
        y = torch.square(y).mean(axis = 0)
        y = torch.sqrt(y + 1e-8)
        y = y.mean(axis = [2,3,4], keepdim = True)
        y = y.mean(axis = 2)
        y = y.tile(group_size, 1, s[2],s[3]) #creates an N x 1 x H x W tensor
        return torch.cat((x,y), dim = 1) #adds a feature channel with the averages

'''
    R1 regulator from https://github.com/ChristophReich1996/Dirac-GAN/blob/decb8283d919640057c50ff5a1ba01b93ed86332/dirac_gan/loss.py#L292
'''

class R1(nn.Module):
    """
    Implementation of the R1 GAN regularization.
    """

    def __init__(self):
        """
        Constructor method
        """
        # Call super constructor
        super(R1, self).__init__()

    def forward(self, prediction_real: torch.Tensor, real_sample: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the regularization
        :param prediction_real: (torch.Tensor) Prediction of the discriminator for a batch of real images
        :param real_sample: (torch.Tensor) Batch of the corresponding real images
        :return: (torch.Tensor) Loss value
        """
        gamma = .5 # weight
        # Calc gradient
        grad_real = torch.autograd.grad(outputs=prediction_real.sum(), inputs=real_sample, create_graph=True)[0]
        # Calc regularization
        regularization_loss: torch.Tensor =gamma * grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return regularization_loss
