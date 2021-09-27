'''
    This folder contains all the helper functions and models using across the style 
    transfer models. So far this module includes models for: 
        Gatys optimization based neural transfer 
        Johson et al Perceptual Neural Transfer Model 
        Ulyanov et al Texture Nets, Improved Texture Nets (both versions)
        Huang and Belongie AdaIN based real time style transfer
'''
import torch 
import numpy as np 
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision
import torchvision.models as m
import torchvision.transforms as t
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
from torch.autograd import Variable
from torchvision.io import read_image 
import pickle5 as pickle
import glob, os
from zipfile import ZipFile
import pytorch_lightning as pl

##_____________________________________________________________________________________________________
##                          UTILITY FUNCTIONS
##_____________________________________________________________________________________________________

#custom dataset, acts on a dataframe of a single column with images
class customDataset(Dataset.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None,index = 2):
        '''
            we just want an unlabelled image dataset
        '''
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.ind =index

    def __len__(self):
        return (len(self.img_labels))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, self.ind])
        image = read_image(img_path)/255
        if self.transform:
            image = self.transform(image)
        return image
#initializes the dataset above into a dataloader
def _initialize_dataset(data_path = None,label_path = None,data_transform = None,index = 2,batch_size = 8):
    dataset = customDataset(label_path, data_path, transform=data_transform,index = index)
    training_set = torch.utils.data.DataLoader(
         dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return training_set,dataset
def plot_image(input_test,test_image,input_val, val_image, style_image, home_dir:str, output_path:str, output_fname:str,save_result: bool = True): 
  fig = plt.figure(figsize = (5,5), dpi = 200) 
  ax = fig.subplots(3,2)
  ax[2,1].imshow(test_image[0].detach().cpu().permute(1,2,0))
  ax[2,1].set_title('Stylized Test Image')
  ax[1,1].imshow(val_image[0].detach().cpu().permute(1,2,0))
  ax[1,1].set_title('Stylized Validation Image')
  ax[0,1].imshow(style_image[0].detach().cpu().permute(1,2,0))
  ax[0,1].set_title('Style Image')
  ax[1,0].imshow(input_val[0].detach().cpu().permute(1,2,0))
  ax[1,0].set_title('Validation Image')
  ax[2,0].imshow(input_test[0].detach().cpu().permute(1,2,0))
  ax[2,0].set_title('Test Image')
  if save_result: 
    os.chdir(home_dir)
    try: 
      os.chdir(output_path) 
    except: 
      for dir in output_path.split('/'):
        try:
          os.chdir(dir)
        except: 
          os.mkdir(dir) 
          os.chdir(dir)
    fname = output_fname 
    num_imgs = len(glob.glob('*.png')) 
    fname += '_'+str(num_imgs)+'.png'
    plt.savefig(fname)
    plt.show(block = False)
    os.chdir('/content/') 
def save_model(model,home_dir, model_path, model_fname): 
  os.chdir(home_dir)
  try: 
    os.chdir(model_path) 
  except: 
    for dir in model_path.split('/'):
      try:
        os.chdir(dir)
      except: 
        os.mkdir(dir) 
        os.chdir(dir)
  fname = model_fname
  num_models = len(glob.glob('*.model')) 
  fname += '_'+str(num_models)+'.model'
  torch.save(model.state_dict(), fname)
  os.chdir('/content/')
#used to initialize weights for different methods
def init_weights_xavier(m):
  if isinstance(m,nn.Conv2d):
    torch.nn.init.xavier_normal_(m.weight)
    if m.bias.data is not None: 
      m.bias.data.fill_(0.)
  elif isinstance(m,nn.ConvTranspose2d): 
    torch.nn.init.xavier_normal_(m.weight)
    if m.bias.data is not None: 
      m.bias.data.fill_(0.)
def init_weights_kaiming(m):
  if isinstance(m,nn.Conv2d):
    torch.nn.init.kaiming_normal_(m.weight)
    if m.bias.data is not None: 
      m.bias.data.fill_(0.)
  elif isinstance(m,nn.ConvTranspose2d): 
    torch.nn.init.kaiming_normal_(m.weight)
    if m.bias.data is not None: 
      m.bias.data.fill_(0.)




#dataset specifically for AdaIN 
class adainDataset(Dataset.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None,index = 2):
        '''
            we just want an unlabelled image dataset
        '''
        file = open(annotations_file, 'rb')
        self.img_labels = pickle.load(file)
        self.style_cols = self.img_labels.columns[1:]
        self.img_dir = img_dir
        self.transform = transform
        self.ind =index
        
    def __len__(self):
        return (len(self.img_labels))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, self.ind])
        image = read_image(img_path)/255
        if self.transform:
            image = self.transform(image)
        target_styles = [self.img_labels[col][idx] for col in self.style_cols]
        return image, target_styles
#initializes the dataset above into a dataloader
def adain_dataloader(data_path = None,label_path = None,data_transform = None,index = 2,batch_size = 8):
    dataset = adainDataset(label_path, data_path, transform=data_transform,index = index)
    loader = torch.utils.data.DataLoader(
         dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return loader,dataset

##_____________________________________________________________________________________________________
##                          LOSS FUNCTIONS
##_____________________________________________________________________________________________________

'''
  Style Losses defined here do not rely on gram matrices but rather losses on statistics
'''
###AdaIN Losses
###computes the AdaIN style Loss
class StyleLoss(nn.Module): 
  def __init__(self): 
    super().__init__() 
    
  def forward(self,input, means, stds): 
    loss = 0
    for s, mean, std in zip(input, means,stds): 
      loss += torch.linalg.norm(self.compute_mean(s)-mean, 2, 1) + torch.linalg.norm(self.compute_var(s,self.compute_mean(s))-std, 2, 1)
    return loss.mean()

  def compute_mean(self, style):
    n,c,h,w = style.shape 
    mu = style.sum(dim = (2,3)) 
    return mu.div(h*w) #---> normalize mean 
  
  def compute_var(self, style, mean, eps = 1e-8): 
    n,c,h,w = style.shape 
    sigma = style.sum(dim =(2,3)).div(h*w) + eps
    return sigma.sqrt() 


'''
  compute normalized MSELoss
'''
###feature channel 2norm ||x - y||_2 <---- may be better to use norm squared here
def normMSELoss(input,target): 
  n, c, h, w = input.shape
  return (1/(n*c*h*w))*torch.linalg.norm(input.view(n,c,-1)-target.view(n,c,-1),2,2).sum()
###computes Normalized Euclidean content loss betweens a set of content maps CONTENTS and TARGET_CONTENTS
def contentLoss(contents, target_content): 
  content_loss = 0 
  for content,target in zip(contents, target_content): 
    content_loss+= normMSELoss(content,target) 
  return content_loss
###PIXELWISELOSS normalized loss
def pixelLoss(gen_image, target_image): 
  assert gen_image.shape == target_image.shape
  n,c,h,w = gen_image.shape
  return (1/(n*c*h*w))*torch.linalg.norm(gen_image.view(n,c,-1)-target_image.view(n,c,-1),2,2).sum()
###computes gram matrix
def gram_matrix(A): 
  N,C, h, w = A.shape
  if N == 1:
    A = A.view(C,-1)
    G = torch.mm(A,A.t())
  else: 
    A = A.view(N,C,-1)
    G = torch.bmm(A,A.transpose(1,2))
  return G.div(C*h*w) #returns CxC normalized gram matrix
#computs gram matrix loss based on euclidean distance
def gramMSELoss(input,target): 
  G = gram_matrix(input)
  return F.mse_loss(G,target) 
#compute Frobenius based loss
def gramFrobLoss(input,target): 
  G = gram_matrix(input) 
  return torch.linalg.norm(G-target,'fro',(1,2)).sum()
###computes styleLosses between set of style representations of the target and source
###can specify the mode between frobenius and 2-norm based
def styleLoss(styles,target_styles,mode :str = 'fro'): 
  style_loss = 0 
  for style,target in zip(styles,target_styles): 
    if mode == 'fro': 
      style_loss += gramFrobLoss(style,target) 
    elif mode == 'mse': 
      style_loss += gramMSELoss(style,target) 
  return style_loss
#TV-loss for Johnson et al's smoothing method
def totalVariationLoss(image): 
  n, c, h, w = image.shape
  z = F.pad(image,(0,1,0,1))
  tv_reg = ((z[:,:,1:,:-1]-z[:,:,:-1,:-1]).pow(2) + (z[:,:,:-1,1:] - z[:,:,:-1,:-1]).pow(2)).sqrt().sum()
  return tv_reg/(n*c*h*w)       

##_____________________________________________________________________________________________________
##                          LIGHTNING DATA MODULES
##_____________________________________________________________________________________________________
'''
    single image class loading modules ---> outputs single images, no pairing
    use for everything except AdaIN module
'''
class LITDataModule(pl.LightningDataModule): 
    def __init__(self,  train_batch_size: int = 8,
                        val_batch_size : int = 2,
                        train_im_size: int = 256, 
                        val_im_size : int = 512, 
                        data_zip : [str] = [],  ####<-------- format these as [traindir, labeldir]
                        data_path :  [str] = [], #### <------------------------| if data_zip is empty, assume these are absolute paths
                        train_anno_path : str = 'data/train.csv',
                        test_anno_path : str = 'data/test.csv',
                        val_anno_path : str = 'data/val.csv', 
                        ): 
        super().__init__() 
        self.batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.im_size = train_im_size
        self.val_im_size = val_im_size
        self.data_dirs = data_path
        self.train_dir = data_path[0]
        self.test_dir = data_path[1]
        self.train_file = train_anno_path
        self.test_file = test_anno_path
        self.val_file = val_anno_path
        self.data_zip = data_zip 
        
    
    def prepare_data(self): 
        #unzip data
        for target_dir, zip_loc in zip(self.data_dirs,self.data_zip): 
            with ZipFile(zip_loc, 'r') as zf: 
                zf.extractall(target_dir)
        
    
    def setup(self, stage: [str] = None): 
        pass #nothing to do here
        
        
    def train_dataloader(self): 
        transform  = t.Compose([ t.Resize(self.im_size), t.CenterCrop((self.im_size,self.im_size))]) 
        loader,dataset = _initialize_dataset(data_path = self.train_dir, label_path = self.train_file,\
                                data_transform = transform, batch_size = self.batch_size, index = 1)
        return loader
        
    # def val_dataloader(self): 
    #     transform  = t.Compose([ t.Resize(self.val_im_size), t.CenterCrop((self.val_im_size,self.val_im_size))]) 
    #     loader,dataset = _initialize_dataset(data_path = self.test_dir, label_path = self.val_file,\
    #                             data_transform = transform, batch_size = self.val_batch_size, index = 1)
    #     return loader
        
    # def test_dataloader(self): 
    #     transform  = t.Compose([ t.Resize(self.val_im_size), t.CenterCrop((self.val_im_size,self.val_im_size))]) 
    #     loader,dataset = _initialize_dataset(data_path = self.test_dir, label_path = self.test_file,\
    #                             data_transform = transform, batch_size = self.val_batch_size, index = 1)
    #     return loader
        
'''
    adain lighting data module, the data for this is stored as a training image and a list of target styles
'''

### THIS MODULE REQUIRES PICKLED DATAFRAMES WHICH CONTAIN THE TARGET STYLE FEATURE MAPS PRECOMPUTED AS TENSORS
class LITAdaINDataModule(pl.LightningDataModule): 
    def __init__(self,  train_batch_size = 8, 
                        val_batch_size = 2, 
                        train_im_size = 256, 
                        val_im_size = 512, 
                        data_zip : [str] = [],  ####<-------- format these as [traindir, labeldir]
                        data_path :  [str] = [], #### <------------------------| if data_zip is empty, assume these are absolute paths
                        train_anno_path : str = 'data/train.pkl',
                        test_anno_path : str = 'data/test.pkl',
                        val_anno_path : str = 'data/val.pkl',
                        ): 
        super().__init__() 
        self.batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.im_size = train_im_size
        self.val_im_size = val_im_size
        self.data_dirs = data_path
        self.train_dir = data_path[0]
        self.test_dir = data_path[1]
        self.train_file = train_anno_path
        self.test_file = test_anno_path
        self.val_file = val_anno_path
        self.data_zip = data_zip
    
    def prepare_data(self): 
        #unzip data
        for target_dir, zip_loc in zip(self.data_dirs,self.data_zip): 
            with ZipFile(zip_loc, 'r') as zf: 
                zf.extractall(target_dir)
        
    def setup(self, stage: [str] = None): 
        pass #nothing to do here
        
        
    def train_dataloader(self): 
        transform  = t.Compose([ t.Resize(self.im_size), t.CenterCrop((self.im_size,self.im_size))]) 
        loader,dataset = adain_dataloader(data_path = self.train_dir, label_path = self.train_file,\
                                data_transform = transform, batch_size = self.batch_size, index =0)
        return loader
        
    # def val_dataloader(self): 
    #     transform  = t.Compose([ t.Resize(self.val_im_size), t.CenterCrop((self.val_im_size,self.val_im_size))]) 
    #     loader,dataset = adain_dataloader(data_path = self.test_dir, label_path = self.val_file,\
    #                             data_transform = transform, batch_size = self.val_batch_size, index = 0)
    #     return loader
        
    # def test_dataloader(self): 
    #     transform  = t.Compose([ t.Resize(self.val_im_size), t.CenterCrop((self.val_im_size,self.val_im_size))]) 
    #     loader,dataset = adain_dataloader(data_path = self.test_dir, label_path = self.test_file,\
    #                             data_transform = transform, batch_size = self.val_batch_size, index = 0)
    #     return loader
    
   
##_____________________________________________________________________________________________________
##                          TEXTURE NET, specify 'BatchNorm2d' or 'InstanceNorm2d'
##_____________________________________________________________________________________________________

'''
    Texture Network Modules
'''
class ConvLayer(nn.Module): 
  def __init__(self,in_channels, out_channels,normalization_method): 
    super().__init__() 
    self.main = nn.Sequential(
                              nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                                        stride = 1,padding =1, padding_mode = 'reflect', bias = True), 
                              getattr(nn,normalization_method)(out_channels, affine = True), 
                              nn.LeakyReLU(.2,inplace= True), 
                              nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                                        stride = 1,padding =1, padding_mode = 'reflect', bias = True), 
                              getattr(nn,normalization_method)(out_channels, affine = True), 
                              nn.LeakyReLU(.2,inplace= True), 
                              nn.Conv2d(out_channels, out_channels, kernel_size = 1, 
                                        stride = 1, bias = True), 
                              getattr(nn,normalization_method)(out_channels, affine = True), 
                              nn.LeakyReLU(.2,inplace= True)     
                              )
  def forward(self,input): 
    return self.main(input) 

'''
    Multiscale Join
'''
class Join(nn.Module): 
  def __init__(self,in_channel_1, in_channel_2,normalization_method):
    super().__init__()
    out_channels = in_channel_1 + in_channel_2
    self.up_branch = nn.Sequential(
                                   nn.UpsamplingNearest2d(scale_factor= 2), 
                                   getattr(nn, normalization_method)(in_channel_1, affine = True)     
                                  )
    self.down_branch = getattr(nn,normalization_method)(in_channel_2, affine = True)
  def forward(self, in1, in2):
    out1 = self.up_branch(in1)
    out2 = self.down_branch(in2)
    return torch.cat((out1,out2),dim=1) 

'''
    Full branch
'''
class ConvJoinBlock(nn.Module): 
  def __init__(self, in_channels_1, out_channels_1, in_channels_2, out_channels_2, normalization_method): 
    super().__init__()
    self.conv_upper = ConvLayer(in_channels_1,out_channels_1,normalization_method)
    self.conv_lower = ConvLayer(in_channels_2,out_channels_2,normalization_method) 
    self.join = Join(out_channels_1, out_channels_2,normalization_method) 

  def forward(self,in1, in2): 
    out1 = self.conv_upper(in1)
    out2 = self.conv_lower(in2)
    out = self.join(out1,out2)
    return out
'''
    Generator
'''
class StyleTransferGenerator(nn.Module): 
  def __init__(self, num_layers = 6, max_im_size = 512,im_channels = 3, feat_maps = 8, normalization_method :str = 'BatchNorm2d'): 
    super().__init__() 
    self.num_layers = num_layers
    self.layers = nn.ModuleList()
    branch_feat_maps = 0
    for i in range(num_layers-1): 
      branch_feat_maps += feat_maps
      if i == 0: 
        self.layers.append(ConvJoinBlock(im_channels,branch_feat_maps,im_channels,feat_maps, normalization_method))
      else: 
        self.layers.append(ConvJoinBlock(branch_feat_maps,branch_feat_maps,im_channels, feat_maps, normalization_method)) 
      
    branch_feat_maps += feat_maps
      
    self.final_block = nn.Sequential(
                                    ConvLayer(branch_feat_maps , branch_feat_maps,normalization_method), 
                                    nn.Conv2d(branch_feat_maps, im_channels,kernel_size = 1, bias = True), 
                                    getattr(nn, normalization_method)(im_channels, affine = True), 
                                    nn.LeakyReLU(.2,inplace = False)
                                    )
  

  def forward(self,input): 
    samples = self.get_downsamples(input)
    out = samples[-1]  
    for i in range(self.num_layers-1,0,-1):
      out = self.layers[self.num_layers - (i+1)](out,samples[i-1] )
    return self.final_block(out)

  def get_downsamples(self,input): 
    downsample = nn.Upsample(scale_factor=.5, mode = 'bilinear') 
    samples = [input]
    for i in range(self.num_layers-1):  
      samples.append(downsample(samples[-1]))
    return samples


'''
    TEXTURE NETWORK 
'''
class TextureNet(nn.Module): 
    def __init__(self,normalization_method: str = 'BatchNorm2d', 
                      num_layers : int = 6, 
                      base_feat_maps: int = 8, 
                      im_channels: int = 3,
                      vgg_net : str = 'vgg19' 
                      ): 
        super().__init__() 
        #create generator network
        self.Transfer_network = StyleTransferGenerator(num_layers = num_layers, im_channels = im_channels, feat_maps = base_feat_maps, normalization_method = normalization_method)
        #create feature extraction network
        assert vgg_net in ['vgg19', 'vgg16', 'vgg19_bn'] 
        if vgg_net == 'vgg19': 
            content_layers = [22]
            style_layers = [3,8,15,18,24]
            vgg_model = m.vgg19(pretrained = True)
        elif vgg_net == 'vgg19_bn': 
            content_layers = [32]
            style_layers = [2,9,26,23,30]
            vgg_model = m.vgg19_bn(pretrained = True) 
        elif vgg_net == 'vgg16':
            content_layers = [8]
            style_layers = [1,6,11,20]
            vgg_model = m.vgg16(pretrained = True)
            
        self.feature_net = FeatureNetwork(list(vgg_model.features), style_layers, content_layers)
        self.Transfer_network.apply(init_weights_xavier)
        
        
    def forward(self,input):#assume noise is fused to input) 
        g_out = self.Transfer_network(input)
        g_out.clip_(0,1) #clip to valid bounds
        #test contents and features
        
        contents,styles = self.feature_net(g_out)
        target_contents, _ = self.feature_net(input) 
        return contents, styles, target_contents
    
#lightning module
class LITTextureNet(pl.LightningModule): 
    def __init__(self,style_image : torch.Tensor,
                      val_image :torch.Tensor,
                      batch_size : int = 8,
                      normalization_method: str = 'BatchNorm2d', 
                      num_layers : int = 6, 
                      base_feat_maps: int = 8, 
                      im_channels: int = 3,
                      vgg_net : str = 'vgg19', 
                      optimizer : str = 'Adam', 
                      learning_rate : float = .1, 
                      betas : tuple = (.5,.99),
                      noise_scale_factor = .01,
                      style_weight = 1e5, 
                      content_weight = 1.
                      ):  
        super().__init__() 
        #initialize model
        self.texture_net = TextureNet(normalization_method = normalization_method, num_layers = num_layers, base_feat_maps = base_feat_maps, im_channels = im_channels, vgg_net = vgg_net)
        self.lr = learning_rate
        self.optimizer = optimizer 
        self.betas = betas
        self.noise_scale_factor = noise_scale_factor
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.save_path = 'Texture_net_model'
        self.start_epoch = 5
        self.val_image = nn.Parameter(val_image)
        '''
            get target styles
        '''
        style_input = Variable(style_image,requires_grad = False)
        _,styles = self.texture_net.feature_net(style_input)
        self.target_styles = nn.ParameterList()
        for i,A in enumerate(styles): 
            self.target_styles.append(nn.Parameter(gram_matrix(A.detach()).tile(batch_size,1,1)))
            
            
        
        
        for p in self.texture_net.feature_net.parameters(): 
            p.requires_grad = False
        self.texture_net.feature_net.eval() #especially necessary if vgg19_bn is selected
        
        
    def forward(self,input): 
        c, s, tc = self.texture_net(input)
        return c, s, tc
    
    def configure_optimizers(self): 
        optimizer = getattr(optim, self.optimizer)(self.texture_net.Transfer_network.parameters(), lr = self.lr, betas = self.betas) 
        return optimizer
    
    def training_step(self, batch, batch_idx): 
        input = batch
        '''
            forward pass
        '''
        input += Variable(torch.rand(input.shape, device = input.device),requires_grad = False)*self.noise_scale_factor
        input.clip_(0,1)
        contents, styles, target_contents = self(input) 
        style_loss = styleLoss(styles, self.target_styles, mode = 'fro')*self.style_weight
        content_loss = contentLoss(contents, target_contents)*self.content_weight
        loss = style_loss + content_loss
        self.log('content_loss', content_loss.detach(), on_step = True, on_epoch = True)
        self.log('style_loss', style_loss.detach(), on_step = True, on_epoch = True) 
        return loss
       
    def training_epoch_end(self, training_step_outputs):
        """ Check if we should save a checkpoint after every train epoch """
        epoch = self.current_epoch
        if epoch >= self.start_epoch and (epoch+1 % 100 == 0):
            ckpt_path = f"{self.save_path}_e{epoch}.ckpt"
            #trainer.save_checkpoint(ckpt_path)
        
        with torch.no_grad(): 
            sample_imgs = self.texture_net.Transfer_network(self.val_image)
            grid = torchvision.utils.make_grid(sample_imgs).unsqueeze(0)
            self.logger.experiment.add_images('generated_images', grid, epoch)
        
    # def validation_step(self, batch,batch_idx): 
    #     input = batch
    #     output = self.texture_net.Transfer_network(input) 
    #     return output
        
##_____________________________________________________________________________________________________
##                          JOHNSON NEURAL STYLE TRANSFER MODEL, specify normalization mode
##_____________________________________________________________________________________________________
'''
  single convolution layers
'''
class JohnsonConvLayer(nn.Module): 
  def __init__(self,in_channels, 
                    out_channels,kernel: 
                    int = 3, stride : int = 1, 
                    padding:int=1, 
                    output_padding:int = 0, 
                    padding_mode:str = 'reflect',
                    bias : bool = True, 
                    upsample: bool = False, 
                    downsample: bool = False,
                    relu_inplace: bool = True,
                    normalization_method : str = 'BatchNorm2d'
                    ): 
    super().__init__() 
    layers = [] 
    if upsample:
      assert stride > 1
      layers += [nn.ConvTranspose2d(in_channels, out_channels,kernel_size = kernel, stride = stride, padding = padding, output_padding= output_padding,bias = bias, padding_mode = 'zeros')]
    if downsample:     
      assert stride > 1
      layers += [nn.Conv2d(in_channels, out_channels, kernel_size = kernel, stride = stride, padding = padding, padding_mode = padding_mode, bias = bias)]
    if (not upsample) and (not downsample): 
      assert stride == 1
      layers += [nn.Conv2d(in_channels, out_channels, kernel_size = kernel, stride = stride, padding = padding, padding_mode = padding_mode, bias = bias)]
    layers += [getattr(nn,normalization_method)(out_channels, affine = True)]
    layers += [nn.LeakyReLU(.2,inplace = relu_inplace)]
    assert len(layers) == 3
    self.conv_layer = nn.Sequential(*layers)  
  
  def forward(self,input): 
    return self.conv_layer(input)

'''
  residual layer
'''
class GrossResidualLayer(nn.Module): 
  def __init__(self,in_channels, 
                    out_channels,
                    kernel: int = 1, 
                    stride : int = 1, 
                    padding:int=1, 
                    padding_mode:str = 'reflect',
                    bias : bool = True,
                    normalization_method:str = 'BatchNorm2d'): 
    super().__init__() 
    self.conv_layer = nn.Sequential(
        nn.Conv2d(in_channels,out_channels, kernel_size = kernel, stride = stride, padding = padding, padding_mode = padding_mode, bias = bias), 
        getattr(nn,normalization_method)(out_channels, affine = True), 
        nn.LeakyReLU(.2,inplace = True), 
        nn.Conv2d(out_channels,out_channels, kernel_size = kernel, stride = stride, padding = padding, padding_mode = padding_mode, bias = bias), 
        getattr(nn,normalization_method)(out_channels, affine = True)
    )

  def forward(self,input): 
    return input + self.conv_layer(input) 

'''
  main generator network
'''
class JohnsonGeneratorNetwork(nn.Module): 
  def __init__(self,normalization_method : str = 'BatchNorm2d'): 
    super().__init__() 
    layers = [] 
    layers += [JohnsonConvLayer(3,32,kernel = 9, stride = 1, padding = 4,normalization_method = normalization_method)] 
    layers += [JohnsonConvLayer(32,64,kernel = 3, stride = 2, downsample = True,normalization_method= normalization_method)]
    layers += [JohnsonConvLayer(64,128,kernel  = 3, stride = 2, downsample = True,normalization_method= normalization_method)]

    for i in range(5): 
      layers += [GrossResidualLayer(128,128, kernel = 3, stride = 1,normalization_method= normalization_method)]
    
    layers += [JohnsonConvLayer(128,64,kernel = 3, stride = 2, padding = 0, output_padding = 1, upsample = True,normalization_method= normalization_method)]
    layers += [JohnsonConvLayer(64,32,kernel = 3, stride = 2, padding = 0, output_padding = 1, upsample = True,normalization_method= normalization_method)]
    layers += [JohnsonConvLayer(32,3,kernel = 9, stride = 1, padding = 1,relu_inplace= False,normalization_method= normalization_method)]

    self.main = nn.Sequential(*layers)

  def forward(self,input): 
    return self.main(input) 
class FastNeuralTransfer(nn.Module): 
    def __init__(self,normalization_method: str = 'BatchNorm2d', 
                      vgg_net : str = 'vgg16' 
                      ): 
        super().__init__() 
        #create generator network
        self.Transfer_network = JohnsonGeneratorNetwork(normalization_method = normalization_method)
        #create feature extraction network
        assert vgg_net in ['vgg19', 'vgg16', 'vgg19_bn'] 
        if vgg_net == 'vgg19': 
            content_layers = [22]
            style_layers = [3,8,15,18,24]
            vgg_model = m.vgg19(pretrained = True)
        elif vgg_net == 'vgg19_bn': 
            content_layers = [32]
            style_layers = [2,9,26,23,30]
            vgg_model = m.vgg19_bn(pretrained = True) 
        elif vgg_net == 'vgg16':
            content_layers = [8]
            style_layers = [1,6,11,20]
            vgg_model = m.vgg16(pretrained = True)
            
        self.feature_net = FeatureNetwork(list(vgg_model.features), style_layers, content_layers)
        for p in self.feature_net.parameters(): 
            p.requires_grad = False
        self.feature_net.eval() #especially necessary if vgg19_bn is selected
        
        self.Transfer_network.apply(init_weights_kaiming)
    
    def forward(self,input): 
        gen_output = self.Transfer_network(input)
        gen_output.clip_(0,1)
        '''
        generate test style and content 
        '''
        contents,styles = self.feature_net(gen_output) 
        '''
        generate actual content
        '''
        target_content, _  = self.feature_net(input)
        return contents, styles, target_content, gen_output
        
class LITFastNST(pl.LightningModule): 
    def __init__(self,style_image : torch.Tensor,
                      val_image :torch.Tensor,
                      batch_size : int = 8,
                      normalization_method: str = 'BatchNorm2d',
                      vgg_net : str = 'vgg19', 
                      optimizer : str = 'Adam', 
                      learning_rate : float = .1, 
                      betas : tuple = (.5,.99),
                      noise_scale_factor = .01,
                      style_weight = 1., 
                      content_weight = 1., 
                      pixel_weight = 10., 
                      tv_weight = 1.
                      ):  
        super().__init__() 
        #initialize model
        self.texture_net = FastNeuralTransfer(normalization_method = normalization_method, vgg_net = vgg_net )
        self.lr = learning_rate
        self.optimizer = optimizer 
        self.betas = betas
        self.noise_scale_factor = noise_scale_factor
        self.style_weight = style_weight 
        self.content_weight = content_weight
        self.pixel_weight = pixel_weight
        self.tv_weight = tv_weight
        self.save_path = 'Texture_net_model'
        self.start_epoch = 5
        self.val_image = nn.Parameter(val_image)
        '''
            get target styles
        '''
        style_input = Variable(style_image,requires_grad = False)
        _,styles = self.texture_net.feature_net(style_input)
        self.target_styles = nn.ParameterList()
        for i,A in enumerate(styles): 
            self.target_styles.append(nn.Parameter(gram_matrix(A.detach()).tile(batch_size,1,1)))
            
          
        
        
    def forward(self,input): 
        c, s, tc,go = self.texture_net(input)
        return c, s, tc,go
    
    def configure_optimizers(self): 
        optimizer = getattr(optim, self.optimizer)(self.texture_net.Transfer_network.parameters(), lr = self.lr, betas = self.betas) 
        return optimizer
    
    def training_step(self, batch, batch_idx): 
        input = batch
        '''
            forward pass
        '''
        input += Variable(torch.rand(input.shape,device = input.device),requires_grad = False)*self.noise_scale_factor
        input.clip_(0,1)
        contents, styles, target_contents,gen_output = self(input) 
        style_loss = styleLoss(styles, self.target_styles, mode = 'fro')*self.style_weight
        content_loss = contentLoss(contents, target_contents)*self.content_weight
        pixel_loss = pixelLoss(gen_output, input) *self.pixel_weight
        tv_loss =  totalVariationLoss(gen_output) *self.tv_weight
        loss = style_loss + content_loss + pixel_loss + tv_loss
        self.log('content loss', content_loss.detach())
        self.log('style loss', style_loss.detach()) 
        self.log('pixel loss', pixel_loss.detach())
        self.log('tv loss', tv_loss.detach())
        return loss
    
    def training_epoch_end(self, training_step_outputs):
        """ Check if we should save a checkpoint after every train epoch """
        epoch = self.current_epoch
        # if epoch >= self.start_epoch and (epoch+1 % 100 == 0):
        #     ckpt_path = f"{self.save_path}_e{epoch}.ckpt"
        #     trainer.save_checkpoint(ckpt_path)
        
        with torch.no_grad(): 
            sample_imgs = self.texture_net.Transfer_network(self.val_image)
            grid = torchvision.utils.make_grid(sample_imgs).unsqueeze(0)
            self.logger.experiment.add_images('generated_images', grid, epoch )    
    # def validation_step(self, batch,batch_idx): 
    #     input = batch
    #     output = self.texture_net.Transfer_network(input) 
    #     return output

##_____________________________________________________________________________________________________
##                          REAL-TIME ARBITRARY STYLE TRANSFER MODEL
##_____________________________________________________________________________________________________

'''
  AdaIN layer
  Takes in a layers and aligns a target images styles to a style images mean and variance
  need to run this for each style
'''
class AdaIN(nn.Module): 
  def __init__(self): 
    super().__init__()  
    
  #given two feature maps --> N x C x K x K 
  #want to compute mean and var
  def forward(self, source_style, target_mean, target_dev): 
    n,c,_,_ = source_style.shape
    mu_x = self.compute_mean(source_style)
    sigma_x = self.compute_var(source_style, mu_x)
    return (target_dev.view(n,c,1,1)*(source_style - mu_x)/sigma_x) + target_mean.view(n,c,1,1) 
  
  #compute the instance mean of a feature tensor
  #returns an N x C x 1 x 1 tensor
  def compute_mean(self, style):
    n,c,h,w = style.shape 
    mu = style.sum(dim = (2,3)) 
    return mu.div(h*w).view(n,c,1,1) #---> normalize mean 
  
  #returns an N x C x 1 x 1 tensor
  def compute_var(self, style, mean, eps = 1e-8): 
    n,c,h,w = style.shape 
    sigma = style.sum(dim = (2,3)).div(h*w) + eps
    return sigma.sqrt().view(n,c,1,1)

'''
  applies adain to each layer to normalize
  this output list now has transformed features and means and stds of origianl output
'''
class AdaModule(nn.Module): 
  def __init__(self): 
    super().__init__() 
    self.adain = AdaIN()
  def forward(self, styles, target_means, target_devs): 
    out = [] 
    for style, target_mean, target_dev in zip(styles, target_means, target_devs): 
      moduled_styles =self.adain(style,target_mean, target_dev)
      out += [moduled_styles]
    return out
'''
  decoder, mimics vgg network layout, inverted, with no normalization
'''
class VggConvBlock(nn.Module): 
  def __init__(self, in_channels, out_channels,num_layers = 4, kernel = 3, stride =1 , padding = 1, padding_mode = 'reflect', bias = True): 
    super().__init__() 
    self.conv_layer = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size= kernel, stride = stride, padding = padding, padding_mode = padding_mode, bias = bias), 
        nn.LeakyReLU(.2, inplace = False) 
    )
    self.final_layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride = stride, padding = padding, padding_mode=padding_mode, bias = bias), 
        nn.LeakyReLU(.2, inplace = False)
    )
    layers = [] 
    for i in range(num_layers-1): 
      layers += [self.conv_layer]
    layers += [self.final_layer]
    self.conv_block = nn.Sequential(*layers)

  def forward(self,input,style = None):
    if style is not None: 
      input = torch.cat((input,style), dim = 1)  
    return self.conv_block(input)
class RealTimeDecoder(nn.Module): 
  def __init__(self): 
    super().__init__() 
    self.module = nn.ModuleDict()
    upsample = nn.Upsample(scale_factor=2, mode = 'nearest')
    self.module['convblock1'] = VggConvBlock(512,256, num_layers = 4)
    self.module['convblock2'] = VggConvBlock(256*2,128, num_layers = 4)
    self.module['convblock3'] = VggConvBlock(128*2,64, num_layers = 2)
    self.module['convblock4'] = VggConvBlock(64*2,3, num_layers = 2)
    self.module['upsample'] = upsample

  '''
    input is a list of styles of lengh 4
  '''
  def forward(self,input): 
    out = self.module['convblock1'](input[0])
    out = self.module['upsample'](out)
    out = self.module['convblock2'](out,input[1])
    out = self.module['upsample'](out)
    out = self.module['convblock3'](out,input[2])
    out = self.module['upsample'](out)
    out = self.module['convblock4'](out,input[3])
    return out 
class AdaINTransfer(nn.Module): 
    def __init__(self,vgg_net : str = 'vgg16' 
                      ): 
        super().__init__() 
        #create generator network
        self.Decoder = RealTimeDecoder()
        #create AdaModule
        self.ada_module = AdaModule() 
        #create feature extraction network
        assert vgg_net in ['vgg19', 'vgg16', 'vgg19_bn'] 
        if vgg_net == 'vgg19': 
            style_layers = [3,8,15,18,24]
            vgg_model = m.vgg19(pretrained = True)
        elif vgg_net == 'vgg19_bn': 
            style_layers = [2,9,26,23,30]
            vgg_model = m.vgg19_bn(pretrained = True) 
        elif vgg_net == 'vgg16':
            style_layers = [1,6,11,20]
            vgg_model = m.vgg16(pretrained = True)
            
        self.Encoder = FeatureNetwork(list(vgg_model.features), style_layers)
        for p in self.Encoder.parameters(): 
            p.requires_grad = False
        self.Encoder.eval() #especially necessary if vgg19_bn is selected
        #initialization weights
        self.Decoder.apply(init_weights_xavier)
    
    def forward(self,input, target_means, target_devs): 
        _,input_styles = self.Encoder(input)
        modulated_styles = self.ada_module(input_styles, target_means, target_devs)[::-1]
        generated_output = self.Decoder(modulated_styles)
        _,styles = self.Encoder(generated_output)
        return styles, modulated_styles

class LITAdaIN(pl.LightningModule): 
    def __init__(self,batch_size : int = 8,
                      vgg_net : str = 'vgg19', 
                      optimizer : str = 'Adam', 
                      learning_rate : float = .1, 
                      betas : tuple = (.5,.99),
                      style_weight = 1., 
                      content_weight = 1. 
                      ):  
        super().__init__() 
        #initialize model
        self.texture_net = AdaINTransfer(vgg_net = vgg_net)
        self.lr = learning_rate
        self.optimizer = optimizer 
        self.betas = betas
        self.style_weight = style_weight 
        self.content_weight = content_weight
        
        self.loss = StyleLoss() 
        
    def forward(self,input, means, devs): 
        styles, mod_styles = self.texture_net(input,means, devs)
        return styles,mod_styles
    
    def configure_optimizers(self): 
        optimizer = getattr(optim, self.optimizer)(self.texture_net.Decoder.parameters(), lr = self.lr, betas = self.betas) 
        return optimizer
    
    def training_step(self, batch, batch_idx): 
        input, means, devs = batch
        s , ms = self(input,means,devs) 
        style_loss = self.Loss(s, means, devs)*self.style_weight
        #loss against the feature maps
        content_loss = contentLoss(s[::-1], ms) *self.content_weight
        self.log('content loss', content_loss.detach())
        self.log('style loss', style_loss.detach()) 
        return style_loss + content_loss
        
    # def validation_step(self, batch,batch_idx): 
    #     input,means,devs = batch
    #     output = self.texture_net.Transfer_network(input) 
    #     _,input_styles = self.Encoder(input)
    #     modulated_styles = self.ada_module(input_styles, means, devs)[::-1]
    #     output = self.Decoder(modulated_styles)
    #     return output

##_____________________________________________________________________________________________________
##                          VGG BASED FEATURE EXTRACTION ARCHITECTURES
##_____________________________________________________________________________________________________
'''
    Discriminator: this is shared all models
'''
class Normalization(nn.Module): 
  def __init__(self,mean,std): 
    super().__init__() 
    self.mean = mean.view(1,-1,1,1)
    self.std = std.view(1,-1,1,1) 
  def forward(self,x):  
    return (x-self.mean.to(x.device))/self.std.to(x.device)
'''
    feature extractor
'''
class FeatureNetwork(nn.Module): 
  def __init__(self, vgg_net, style_layers = [], content_layers = []): 
    super().__init__()
    assert (style_layers is not None) or (content_layers is not None)
    '''
        for reference
    '''
    #feature layers of a vgg19_bn network
    #self.content_layers = [32] relu4_2
    #self.style_layers = [2,9,26,23,30] relu1_1 relu2_1 relu3_1 relu4_1 relu5_1
    #layers of vgg19 net
    #self.content_layers = [22] relu4_2
    #self.style_layers = [3,8,15,18,24] relu1_1 relu2_1 relu3_1 relu4_1 relu5_1
    #layers of vgg16
    # self.content_layers = [8] relu2_2
    # self.style_layers = [1,6,11,20] relu1_1 relu2_1 relu3_1 relu4_1
    self.content_layers = content_layers 
    self.style_layers = style_layers
    
    if self.style_layers != [] : 
        max_layers = self.style_layers[-1] 
    else: 
        max_layers = 0
    if self.content_layers != [] : 
        max_layers = max(max_layers, self.content_layers[-1]) 
    

    #only extract necessary features    
    self.features = nn.ModuleList([feature for i,feature in enumerate(vgg_net) if i <=max_layers])


    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    self.norm = Normalization(mean,std)
    
  def forward(self,input): 
    input = self.norm(input)
    styles = [] 
    content = [] 
    for i,module in enumerate(self.features): 
      input = module(input) 
      if i in self.content_layers: 
          content.append(input)
      if i in self.style_layers: 
        styles.append(input)
        
    return  content, styles
    

##_____________________________________________________________________________________________________
##                          Style extraction module to preprocess and extract the styles and features
##                          from a set of images                    
##_____________________________________________________________________________________________________

#helper function to preprocess a set of style images and create a pickled training file containing output tensor maps as well 
def extract_features(data_dir, output_fname,vgg_net : str = 'vgg16', im_size = 256): 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    assert vgg_net in ['vgg19', 'vgg16', 'vgg19_bn'] 
    if vgg_net == 'vgg19': 
        content_layers = [22]
        style_layers = [1,6,11,20]
        vgg_model = m.vgg19(pretrained = True)
    elif vgg_net == 'vgg19_bn': 
        content_layers = [32]
        style_layers = [2,9,26,23,30]
        vgg_model = m.vgg19_bn(pretrained = True) 
    elif vgg_net == 'vgg16':
        #content_layers = [8]
        content_layers = []
        style_layers = [3,8,15,18,24]
        vgg_model = m.vgg16(pretrained = True)
    f_net = FeatureNetwork(list(vgg_model.features), style_layers, content_layers).to(device)
    for p in f_net.parameters():
        p.requires_grad = False
    pwd = os.getcwd() 
    print(pwd)
    os.chdir(data_dir)#<---- change working directory 
    paths = [] 
    contents  = [] 
    styles = []
    transform = t.Compose([t.Resize(im_size), t.CenterCrop(im_size)]) 
    for file in glob.glob('*'): #assuming every file here is in the file
        try: 
            img = read_image(file).to(device)/255 
            img = transform(img)
            im_content,im_styles = f_net(img)
            c = [map.detach().cpu() for map in im_content]
            s = [style.detach().cpu() for style in im_styles] 
            contents.append([c])
            styles.append([s]) 
            paths.append(file) 
        except: 
            pass
    print('Successfully processed {} images'.format(len(paths)))
    
    df = pd.DataFrame(paths,columns = ['paths']) 
    if style_layers != []:
        for i in range(len(styles[0][0])): 
            colname = 'style_'+str(i+1)
            f = pd.DataFrame(columns = [colname])
            f[colname] = f[colname].astype(object) 
            for j in range(len(styles)): 
                f.loc[j,colname] = styles[j][0][i].cpu()
            df = pd.concat([df,f], axis = 1) 
    if content_layers != []:
        for i in range(len(contents[0][0])):
            colname = 'content_'+str(i+1)
            f = pd.DataFrame(columns = [colname]) 
            f[colname] = f[colname].astype(object)
            for j in range(len(contents)): 
                f.loc[j,colname] = contents[j][0][i].cpu()
            df = pd.concate([df,f], axis = 1) 
    
    os.chdir(pwd)
    df.to_pickle(output_fname)
    print('dataframe saved to: {}'.format(os.path.join(pwd,output_fname)))
def unzip_data(output_dirs, zip_locs): 
    assert len(output_dirs)==len(zip_locs)
    print('unzipping data')
    for dir, loc in zip(output_dirs, zip_locs):
        with ZipFile(loc, 'r') as zf: 
            zf.extractall(dir)