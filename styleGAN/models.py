'''
    want to implement styleGAN as done in the original paper:
    https://arxiv.org/abs/1812.04948
    Their official tensorflow implementation is:
    https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
    I implemented their implementation using pytorch to the best of my abilities

    their original implementation has 4 networks:
        Generator
            mapping network
            synthesis network
        discriminator network

    they also wrote custom kernels for
        blur2d
        upsample
        downsample
        pixelnorm
        instancenorm
        weight_initialization -----> He et al ReLU conv weight initialization from: https://arxiv.org/abs/1502.01852
        dense linear
        conv2d
        conv2d_upsample     ---> upsample with transposed conv
        conv2d_downsample   ---> downsample with stride 2 convolution

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import util

'''
    The generator network implements the mapping network and the
    synthesis networks. The generator implements a few
    regularization tricks as well:
        - Tracking center of mass of latent distribution, W
            - update the moving average with EMA
            dlatent_avg = dlatent_avg + (batch_avg - dlatent_avg)*dlatent_avg_beta
        - style mixing regularization
            -generate 2nd set of latents
            -push through mapping network too
            -use random probability for if to use mixing
            -randomly combine original latents with new latents
            -return
        - truncation trick
            - given # layer cutoff and multiplier < 1
            - compute which layers to effect
            - apply using EMA
            dlatent = dlatent_avg + (dlatent - dlatent_avg)*multiplier
'''
class styleGAN_generator(nn.Module):
    def __init__(self,
                target_im_size = 512,           #target image size
                im_channels = 3,
                latent_dim = 512,               #latent dimension
                label_dim = 0,               #dimension of labels
                feat_maps = 8192,               #used for computing number of feature maps
                max_features = 512,            #minimum number of feature maps
                act_type = 'LeakyReLU',         #which type of activation to use --> 'ReLU' or 'leakyReLU'
                wgain = np.sqrt(2),             #gain to use for weight initialization
                w_scale = True,                 #whether or not to use weight equalization
                track_W = True,                 #track center of mass of latent space
                dlatent_avg_beta = .995,         #EMA weight for tracking center of mass
                use_style_mixing = True,        #use style mix regularization
                prob_style_mix = .5,
                use_truncation = True,          #use truncation trick
                truncation_cutoff_layer = 4,    #truncation cutoff
                truncation_weight = .7,        #truncation weighting,
                device = torch.device("cpu")
                ):
        super(styleGAN_generator,self).__init__()
        self._resolution = int(np.log2(target_im_size))
        assert target_im_size == 2**self._resolution and (target_im_size > 4)
        self._num_layers = self._resolution*2 -2
        self.device = device
        self._track_W = track_W
        self._dlatent_avg_beta = dlatent_avg_beta
        if track_W: self._dlatent_avg = torch.zeros(latent_dim, device = device)
        self._use_style_mixing = use_style_mixing
        if use_style_mixing:
            self._prob_style_mix = prob_style_mix
            self._layer_index = torch.arange(0,self._num_layers, device = device)
        if use_truncation:
            self._use_truncation = use_truncation
            self._truncation_cutoff_layer = truncation_cutoff_layer
            self._truncation_weight = truncation_weight
        #pushes latent code z ---> w
        self.mapping_net = MappingNetwork(target_im_size = target_im_size, latent_dim = latent_dim,
                                    label_dim =label_dim, num_layers = 8, num_weights = 512,
                                    normalize_latent = True, act_type = act_type,device = device)
        self.synthesis_net = SynthesisNetwork(target_im_size = target_im_size,           #target image size
                                            im_channels = im_channels,
                                            latent_dim = latent_dim,               #latent dimension
                                            feat_maps = feat_maps,               #used for computing number of feature maps
                                            max_features = max_features,            #minimum number of feature maps
                                            feat_map_decay = 1.0,           #rate of decay for feature map sizes, leave at 1
                                            num_layers = self._num_layers,
                                            constant_input = True,              #whether or not to use constant input
                                            act_type = act_type,
                                            use_noise = True,
                                            use_bias = True,
                                            use_norm = True,
                                            use_style_mode= True,
                                            device = device)

    def forward(self, latent, label = None):
        s = latent.shape
        dlatent = self.mapping_net(latent,label)
        '''
            compute EMA center of mass of latent distribution W
        '''
        if self._track_W:#use exponential moving average to update latent
            batch_avg = dlatent.detach().mean(dim = 1, keepdim = True)
            self._dlatent_avg = util.EMA(self._dlatent_avg, batch_avg, self._dlatent_avg_beta)

        '''
        implement truncation trick
        '''
        dlatent = dlatent.unsqueeze(1).tile(1,self._num_layers,1)

        if self._use_truncation:
            ones = torch.ones(self._layer_index.shape, device = self.device)
            coefs = torch.where(self._layer_index < self._truncation_cutoff_layer,ones*self._truncation_weight, ones )
            dlatent = dlatent + (self._dlatent_avg.unsqueeze(1).tile(1,self._num_layers,1) - dlatent)*(coefs.unsqueeze(-1).unsqueeze(0))
        '''
            compute style mixing
        '''
        '''
            first need to broadcast dlatent correctly
        '''
        if self._use_style_mixing and (np.random.rand() <= self._prob_style_mix):
            latent_2 = torch.randn(s[0], s[1], device = self.device)
            dlatent2 = self.mapping_net(latent_2, label)
            dlatent2 = dlatent2.unsqueeze(1).tile(1,self._num_layers,1)
            mix_cutoff = np.random.randint(0,s[1])#how many channels to mix
            #modify the latent code
            dlatent = torch.where(self._layer_index.reshape(1,self._num_layers,1) <= mix_cutoff, dlatent, dlatent2)


        images = self.synthesis_net(dlatent)
        return images







'''
    computes mapping from input latent z to style vector w
    uses ReLU or leaky activations,pixelnorm on input latent z
    can add conditioner to this as well by pushing through
    dense encoding network and concatenating to z before pushing
    through dense 8 layer network
'''
class MappingNetwork(nn.Module):
    def __init__(self,
                target_im_size = 512,           #target image size
                latent_dim = 512,               #latent dimension
                label_dim = 0,                  #dimension of labels
                num_layers = 8,                 #number of layer
                num_weights = 512,               #number of weights per layer
                normalize_latent = True,        #whether or not to normalize the latent z
                act_type = 'LeakyReLU',          #type of activations to use
                device = torch.device("cpu")
                ):
        super(MappingNetwork,self).__init__()
        self._normalize = normalize_latent
        self.device = device
        layers = []
        if label_dim:
            self._embed_label = nn.Linear(label_dim, latent_dim, bias = False)
        for i in range(num_layers):
            c_in = 512
            c_out = 512
            if i == 0:
                c_in = latent_dim+label_dim
            layers.append(nn.Linear(c_in,c_out, bias = True))
            if act_type == 'LeakyReLU':
                layers.append(getattr(nn, act_type)(.2, inplace = False))
            elif act_type == 'ReLU':
                layers.append(getattr(nn, act_type)(inplace = True))
        self._main = nn.Sequential(*layers)

    def forward(self, latent, label = None):
        if label:
            label = self._embed_label(label)
            latent = torch.cat((latent,label),dim=1)
        if self._normalize:
            latent = F.normalize(latent, p = 2.0, dim = 1)
        return self._main(latent)
'''
    This contains all the blocks to build the synthesis network
    uses a layer epilogue block to apply the post convolution parts
    Uses no skips unlike styleGAN2
    This network also uses a final 1x1 convolution to map to RGB
    --> noise --> bias --> activation --> norm --> style mod
'''
class LayerEpilogue(nn.Module):
    def __init__(self,
                channels = 512,
                latent_dim = 512,
                use_noise = True,
                use_bias = True,
                act_type = 'LeakyReLU',
                use_instance_norm = True,
                use_style_mod = True,
                device = torch.device("cpu")
                ):
        super(LayerEpilogue,self).__init__()
        self.noise = util.AddNoise(channels = channels, device = device) if use_noise else None
        self.bias = util.Bias(channels = channels, device = device) if use_bias else None
        self.device = device
        if act_type == 'LeakyReLU':
            self.activation = getattr(nn, act_type)(.2, inplace = False)
        elif act_type == 'ReLU':
            self.activation = getattr(nn, act_type)(inplace = True)
        else:
            self.activation = None

        self.norm = nn.InstanceNorm2d(channels, affine = False) if use_instance_norm else None
        self.style_mod  = util.StyleMod(latent_dim = latent_dim, channels = channels ,device = device) if use_style_mod else  None

    def forward(self, x, w, nz):
        print('after conv: ',x.max())
        if self.noise:
            x = self.noise(x,nz)

        print('after noise: ',x.max())
        if self.bias:
            x = self.bias(x)
        print('after bias: ',x.max())
        if self.activation:
            x = self.activation(x)
        print('after act: ',x.max())
        if self.norm:
            x = self.norm(x)
        print('after norm: ',x.max())
        if self.style_mod:
            x = self.style_mod(x,w)
        print('after mod: ',x.max())
        return x

class SynthesisNetwork(nn.Module):
    def __init__(self,
                target_im_size = 512,           #target image size
                im_channels = 3,
                latent_dim = 512,               #latent dimension
                feat_maps = 8192,               #used for computing number of feature maps
                max_features = 512,            #minimum number of feature maps
                feat_map_decay = 1.0,           #rate of decay for feature map sizes, leave at 1
                num_layers = 8,
                constant_input = True,              #whether or not to use constant input
                act_type = 'LeakyReLU',
                use_noise = True,
                use_bias = True,
                use_norm = True,
                use_style_mode= True,
                blur_filter = [1,2,1],
                device = torch.device("cpu")
                ):
        super(SynthesisNetwork,self).__init__()
        self._resolution = int(np.log2(target_im_size))
        self.num_layers = num_layers
        self._feat_maps = feat_maps
        self._max_features = max_features
        self._feat_map_decay = feat_map_decay
        self.constant_input = constant_input
        self.device = device
        if constant_input:
            self.c = torch.ones(1,512,4,4).to(device) #fixed input vector
        else:
            self._dense_input = nn.Linear(latent_dim, self.num_features(0)*16, bias = False)
        if use_noise:
            self._use_noise = use_noise

        '''
            create the layers for the network
            In any case, there are 10 layer epilogues
            the first layer epilogue is unique
        '''
        self.layers = nn.ModuleDict()

        for i in range(2,self._resolution+1):
            if i != 2: #first iter
                #self.layers['blur{}'.format(i)] = util.Blur2d(filter = blur_filter, channels = self.num_features(i-1), device= device)
                self.layers['conv{}_1'.format(i)] = nn.ConvTranspose2d(self.num_features(i-1), self.num_features(i), kernel_size = 3, stride = 2, padding = 1,
                                                output_padding = 1, padding_mode ='zeros', bias = False)
            #add epilogue
            self.layers['layer_epilogue_{}_1'.format(i)]= LayerEpilogue(channels = self.num_features(i), latent_dim = latent_dim, use_noise = use_noise,use_bias = use_bias,
                                            act_type = act_type, use_instance_norm = use_norm, use_style_mod = use_style_mode, device = device)
            #regular convolution and epilogue
            self.layers['conv{}_2'.format(i)] = nn.Conv2d(self.num_features(i),self.num_features(i), kernel_size =3, stride = 1, padding = 1, bias = False)
            self.layers['layer_epilogue_{}_2'.format(i)] = LayerEpilogue(channels = self.num_features(i), latent_dim = latent_dim, use_noise = use_noise , use_bias = use_bias,
                                        act_type = act_type, use_instance_norm = use_norm, use_style_mod = use_style_mode, device = device)
        self.to_rgb = nn.Conv2d(self.num_features(self._resolution),im_channels, kernel_size = 1, stride = 1, bias = True)

    def forward(self, latent):
        s = latent.shape #batch_size x num_layers x latent_dim
        '''
            generate noise inputs
        '''
        if self._use_noise:
            noise = []
            for i in range(self.num_layers):
                res = (i // 2)+ 2
                noise.append(torch.randn(s[0], 1, 2**res, 2**res, device = self.device))

        if not self.constant_input:
            x = self._dense_input(latent[:,0]).reshape(s[0],s[2],4,4) #convert to batchsize x 512 x 4 x 4 tensor
        else:
            x = self.c

        i = 0
        for layer in self.layers:
            if 'conv' in layer: #layer is a convolution
                x = self.layers[layer](x)
            #elif 'blur' in layer:
            #    x = self.layers[layer](x)
            else: #layer is a epilogue layer and requires additional args
                nz = noise.pop(0) #get some noise
                w = latent[:,i]
                x = self.layers[layer](x,w,nz)
                i+=1
                pass

        return self.to_rgb(x)


    def num_features(self, stage):
        return int(min(self._feat_maps//2**(stage*self._feat_map_decay), self._max_features))

'''
    discriminator is similarly simple

    uses label conditioning if labels are presnet from
    https://arxiv.org/pdf/1801.04406.pdf
    This surmounts to applying label conditioning at the output
    of the discriminator
'''
class styleGAN_discriminator(nn.Module):
    def __init__(self,
                target_im_size = 512,
                in_channels = 3,
                label_size = 0,
                feat_maps = 8192,
                feat_map_decay = 1.0,
                max_features= 512,
                blur_filter = [1,2,1],
                minibatch_groups = 4,
                minibatch_channels =1,
                act_type = 'LeakyReLU',
                device = torch.device("cpu")
                ):
        super(styleGAN_discriminator,self).__init__()
        self._feat_maps = feat_maps
        self._feat_map_decay = feat_map_decay
        self._max_features= max_features
        self.minibatch_group_size = minibatch_groups
        self.minibatch_num_channels = minibatch_channels
        self.res = int(np.log2(target_im_size))
        self.device = device
        use_filter = (blur_filter == 0)
        self.from_rgb = nn.Sequential(
                            nn.Conv2d(in_channels, self.num_features(self.res), kernel_size = 1, stride = 1, bias = True),
                            nn.ReLU(.2, inplace = False))

        layers = []
        for i in range(self.res, 2, -1): #2**init_res = target_im_size...... 2**3= 8
            layers.append(nn.Conv2d(self.num_features(i), self.num_features(i), kernel_size = 3, stride =1 , padding = 1, bias = True ))
            if act_type == 'LeakyReLU':
                layers.append(getattr(nn, act_type)(.2, inplace = False))
            elif act_type == 'ReLU':
                layers.append(getattr(nn, act_type)(inplace = True))
            #if not use_filter:
                #layers.append(util.Blur2d(filter = blur_filter, channels = self.num_features(i), device = device))
            layers.append(nn.Conv2d(self.num_features(i), self.num_features(i-1), kernel_size = 3, stride = 2, padding = 1, bias = True))
            if act_type == 'LeakyReLU':
                layers.append(getattr(nn, act_type)(.2, inplace = False))
            elif act_type == 'ReLU':
                layers.append(getattr(nn, act_type)(inplace = True))


        self._main = nn.Sequential(*layers)

        layers= []
        layers.append(nn.Conv2d(self.num_features(2)+1,self.num_features(2), kernel_size = 3, stride = 1, padding =1, bias = True))
        if act_type == 'LeakyReLU':
            layers.append(getattr(nn, act_type)(.2, inplace = False))
        elif act_type == 'ReLU':
            layers.append(getattr(nn, act_type)(inplace = True))
        self._final_conv = nn.Sequential(*layers)


        layers = []
        layers.append(nn.Linear(self.num_features(2)*16, self.num_features(1)*16, bias = True))
        if act_type == 'LeakyReLU':
            layers.append(getattr(nn, act_type)(.2, inplace = False))
        elif act_type == 'ReLU':
            layers.append(getattr(nn, act_type)(inplace = True))
        layers.append(nn.Linear(self.num_features(1)*16, max(label_size,1), bias = True))

        self._final_dense = nn.Sequential(*layers)

    def forward(self,image, label= None):
        x = self.from_rgb(image)
        x = self._main(x)
        print(x.max(),'discriminator out')
        #increases channel count by 1
        if self.minibatch_group_size <= image.shape[0]:
            x = self.minibatch_std_dev(x, group_size = self.minibatch_group_size, new_channels = self.minibatch_num_channels)
        else:
            s = x.shape
            zeros = torch.zeros(s[0], 1, s[2], s[3]).to(self.device)
            x = torch.cat((x,zeros),dim =1 )
        print(x.max(),'discriminator out')
        x = self._final_conv(x)
        print(x.max(),'discriminator out')

        x = self._final_dense(x.flatten(start_dim=1))
        print(x.max(),'discriminator out')
        #apply optional label conditioning
        if label:
            x = torch.sum(x*label, axis = 1)

        return x
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
