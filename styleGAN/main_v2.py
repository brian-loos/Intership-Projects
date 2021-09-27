
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.models as visionmodels
import torchvision.transforms as transforms
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
from torch.utils.checkpoint import checkpoint_sequential
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable,grad
from torchvision import datasets
from collections import *
import os
from torchvision.io import read_image
from progress.bar import Bar
from datetime import datetime
import glob
from zipfile import ZipFile
from models_v2 import *
from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity
'''
    custom files
'''

class customDataset(Dataset.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None,target_transform = None,dtype = torch.float32):
        '''
            we just want an unlabelled image dataset
        '''
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.dtype = dtype
    def __len__(self):
        return (len(self.img_labels))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        if '.webp' in img_path:
            try:
                image = Image.open(img_path).convert('RGB')
                #image = np.array(im).transpose(2,0,1)
                image = transforms.ToTensor()(image)
            except:
                print('could not load image')
                img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx+1, 1])
                image = Image.open(img_path).convert('RGB')
                #image = np.array(im).transpose(2,0,1)
                image = transforms.ToTensor()(image)
        else:
            image = read_image(img_path)/255

        if self.transform:
            image = self.transform(image).to(self.dtype)

        return image

def _initialize_dataset(data_path = None, label_path = None, data_transform = None,target_transform = None,dtype = torch.float32):
    dataset = customDataset(label_path, data_path, transform=data_transform,target_transform = target_transform, dtype = dtype)
    training_set = torch.utils.data.DataLoader(
         dataset, batch_size=training_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return training_set,dataset


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA = .5, device = torch.device('cpu'),dtype  = torch.float32):
    alpha = torch.rand(training_batch_size, 1,1,1,device = device,dtype = dtype).to(memory_format = torch.channels_last)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device = device,dtype = dtype),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty



def _train(num_epochs = 100,G = None, D =None,
            gOptim = None, dOptim = None,
            gScheduler = None,dScheduler = None,
            Loss_func = None, Loader = None, val_image = None, device = torch.device('cpu'),dtype = torch.float32):
    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = one * -1
    train_gen_iter = 10
    wgan_gap = 0
    r1_reg = R1()

    for epoch in range(num_epochs):
        print('Processing epoch {}/{}'.format(epoch+1, num_epochs))
        mean_loss = 0
        gen_mean_loss = 0
        mean_gp = 0
        with Bar('Processing epoch: ',max = len(Loader)) as bar:
            for i, image in enumerate(Loader):
                input_image = Variable(image).to(device).to(memory_format = torch.channels_last)

                batch_size = len(input_image)
                real_label = torch.ones(batch_size, device = device, dtype = dtype)
                fake_label = torch.zeros(batch_size, device = device, dtype = dtype)
                if batch_size != training_batch_size:
                    break


                for p in D.parameters():
                    p.requires_grad = True
                for p in G.parameters():
                    p.requires_grad = False

                dOptim.zero_grad(set_to_none = True)
                latents = Variable(torch.randn(batch_size, latent_dim, device = device,dtype = dtype)).detach()
                fake_imgs = Variable(G(latents))
                if profile:
                    with torch.autograd.profiler.profile(use_cuda = True,profile_memory = True, with_stack = True) as prof:
                        real_d = D(input_image).view(-1)
                    print(prof.key_averages().table())
                else:
                    real_d =D(input_image).view(-1)
                real_d = real_d.mean()
                real_d.backward(mone)
                # real_loss = nn.BCEWithLogitsLoss()(real_d, real_label)
                # real_loss.backward()
                '''
                    generate fakes
                '''
                if profile:
                    with torch.autograd.profiler.profile(use_cuda = True,profile_memory = True, with_stack = True) as prof:
                        fake_d = D(fake_imgs).view(-1)
                    print(prof.key_averages().table())
                else:
                    fake_d = D(fake_imgs).view(-1)

                fake_d = fake_d.mean()
                fake_d.backward(one)
                # fake_loss = nn.BCEWithLogitsLoss()(fake_d, fake_label)
                # fake_loss.backward()

                '''
                    gradient penalty
                '''

                gp = calc_gradient_penalty(D, input_image.detach(), fake_imgs.detach(),device = device, dtype = dtype)
                gp.backward()

                # if (i%10) == 0:
                #     input_image.requires_grad = True
                #     real_d = D(input_image).view(-1)
                #     r1 = r1_reg(prediction_real = real_d, real_sample = input_image)*10
                #     r1.backward()

                dOptim.step()
                # mean_loss += (real_loss.detach() + fake_loss.detach() + r1.detach())

                mean_loss += real_d.detach() - fake_d.detach()

                mean_gp += gp.detach()
                '''
                    train generator
                '''

                real_label = torch.ones(batch_size//2,device = device, dtype = dtype)
                fake_label = torch.zeros(batch_size//2, device = device, dtype = dtype)
                if  (i+1)%train_gen_iter ==  0:
                    for k in range(2*train_gen_iter):
                        for p in D.parameters():
                            p.requires_grad = False
                        for p in G.parameters():
                            p.requires_grad = True
                        gOptim.zero_grad(set_to_none = True)
                        latents = Variable(torch.randn(batch_size//2, latent_dim, device = device,dtype = dtype))
                        if profile:
                            with torch.autograd.profiler.profile(use_cuda = True,profile_memory = True, record_shapes = True, with_stack = True) as prof:
                                fake_out = G(latents)
                            print(prof.key_averages().table())
                        else:
                            fake_out = G(latents)

                        outputs = D(fake_out).view(-1)
                        outputs = outputs.mean()
                        outputs.backward(mone)
                        gen_mean_loss += outputs.detach()

                    gOptim.step()
                if (i+1) % 150 == 0:
                    print('\nRunning discriminator loss current epoch: ', mean_loss/(i+1))
                    print('Running gp loss at current epoch: ',mean_gp/(i+1))
                    img_comparision_fname = 'test_{}_epoch_{}.png'.format(i,epoch)
                    fake_im = fake_out.detach()[0]
                    test_out = torch.reshape(fake_im,(3,target_size,target_size))
                    fig = plt.figure(figsize=(5,5), dpi = 100)
                    ax = fig.subplots()
                    ax.imshow(test_out.cpu().detach().permute(1,2,0))
                    os.chdir('outputs')
                    plt.savefig(img_comparision_fname)
                    plt.show(block = False)
                    plt.close()
                    os.chdir('..')
                # if (i+1) %10 == 0 :
                #     print('processing iteration: {}/{}'.format(i+1, len(Loader)))
                #     fake_im = fake_out.detach()[0]
                #     test_out = torch.reshape(fake_im,(3,target_size,target_size))
                #     fig = plt.figure(figsize=(5,5), dpi = 100)
                #     ax = fig.subplots()
                #     ax.imshow(test_out.cpu().detach().permute(1,2,0))
                #     plt.show(block = True)

                bar.next()
                if (i+1) %10000 == 0:
                    print('-----------------------------')
                    print('saving model')
                    #then save model
                    os.chdir('models')
                    torch.save(G.state_dict(), 'style_G_{}_{}.model'.format(i,epoch))
                    torch.save(D.state_dict(), 'style_D_{}_{}.model'.format(i,epoch))
                    os.chdir('..')
                if (i+1)%10 == 0:
                    gScheduler.step()
                    dScheduler.step()
            bar.finish()


        if epoch == 0:
            print('Zipping old outputs')
            date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            zip_name_imgs = 'outputs_images_{}.zip'.format(date_time)
            os.chdir('outputs')
            if glob.glob('*.png') != []:
                with ZipFile(zip_name_imgs, 'w') as zip:
                    for img in glob.glob("*.png"):
                        zip.write(img)

            os.system('rm *.png')
            os.chdir('..')

        print('--------------------------------------------------------------')
        print('Losses at epoch {}/{}'.format(epoch +1, num_epochs+1))
        #print('Discriminator loss: {} \n Generator Loss: {} \n'.format(dLoss, gLoss))
        #G.eval()
        #print(fixed_latent.shape)


        if (epoch+1) %1 == 0:
            print('-----------------------------')
            print('saving model')
            #then save model
            os.chdir('models')
            torch.save(G.state_dict(), 'style_G_{}.model'.format(epoch))
            torch.save(D.state_dict(), 'style_D_{}.model'.format(epoch))
            os.chdir('..')

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    target_size = 64
    latent_dim = 256
    learning_rate = .0002
    betas = (.5,.99)
    weight_decay = 1e-8
    training_batch_size = 32
    dtype = torch.float32
    profile = False

    if ('pretrained_D.model' in glob.glob('*.model')) and('pretrained_G.model' in glob.glob('*.model')):
        use_pretrained = True
    else:
        use_pretrained = False
    generator = styleGenerator(target_im_size = target_size,latent_dim = latent_dim, device = device,dtype = dtype ).to(device)
    discriminator = styleDiscriminator(target_im_size = target_size, latent_size = latent_dim, device = device,dtype = dtype).to(device)
    if use_pretrained:
        print('loading pretrained models ')
        generator.load_state_dict(torch.load('pretrained_G.model'))
        discriminator.load_state_dict(torch.load('pretrained_D.model'))
    generator = generator.to(memory_format = torch.channels_last)
    discriminator = discriminator.to(memory_format = torch.channels_last)
    gOptim = torch.optim.Adam(generator.parameters(),lr= learning_rate*10, weight_decay=weight_decay, betas = betas)
    dOptim = torch.optim.Adam(discriminator.parameters(),lr= learning_rate, weight_decay=weight_decay, betas = betas)
    def lmbda(epoch):
        if (epoch+1%1000) == 0:
            return .1
        else:
            return 1


    gScheduler = torch.optim.lr_scheduler.MultiplicativeLR(
       gOptim, lr_lambda=lmbda, verbose=True)
    dScheduler = torch.optim.lr_scheduler.MultiplicativeLR(
       dOptim, lr_lambda=lmbda, verbose=True)
    #gScheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(gOptim, 100, 2, eta_min = 1e-8,verbose = True)
    #dScheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(dOptim, 100, 2, eta_min = 1e-8,verbose = True)
    print('preparing dataset')
    #path_2_data = '..\Landmark Detection\data\celeba\Img\img_align_celeba\img_align_celeba'
    #im_path_fname = '..\Landmark Detection\data\celeba\Eval\cats_traing.csv'
    #path_2_data = '..\data'
    #im_path_fname = '..\data\cats_2.csv'
    path_2_data = 'F:\cat\cat'
    im_path_fname = 'F:\cat\cats_2.csv'
    transform = transforms.Compose([transforms.Resize(target_size + 16),transforms.RandomCrop(target_size)])
    train_loader,dataset = _initialize_dataset(data_path = path_2_data, label_path = im_path_fname,\
                                data_transform = transform, dtype = dtype)

    _train(num_epochs = 100,G = generator, D =discriminator, gOptim = gOptim, dOptim = dOptim,
                gScheduler = gScheduler,dScheduler = dScheduler, Loader = train_loader, device = device, dtype = dtype)
