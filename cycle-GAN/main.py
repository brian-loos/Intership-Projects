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
import models, os
from torchvision.io import read_image
from progress.bar import Bar




class customDataset(Dataset.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None,target_transform = None):
        '''
            we just want an unlabelled image dataset
        '''
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return (len(self.img_labels))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 2])
        image = read_image(img_path)/255
        label_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 3])
        im_label = read_image(label_path)/255

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            im_label = self.target_transform(im_label)
        return image, im_label

def _initialize_dataset(data_path = None, label_path = None, data_transform = None,target_transform = None):
    dataset = customDataset(label_path, data_path, transform=data_transform,target_transform = target_transform)
    training_set = torch.utils.data.DataLoader(
         dataset, batch_size=training_batch_size, shuffle=True, num_workers=1, pin_memory=True)
    return training_set,dataset

def _train(num_epochs = 1, G = None, F = None,Dx = None, Dy = None, loss_net = None,
            gOptim = None, fOptim = None, dxOptim = None, dyOptim = None,
            gScheduler = None,fScheduler = None,dxScheduler = None, dyScheduler = None,
            Loss_func = None,Cycle_loss  = None,Loader = None, val_image = None):
    len_loader = len(Loader)
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1/0.229, 1/0.224, 1/0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.])
                                   ])
    l1 =10
    REGULARIZATION = 1/(3*512*512)
    l2 = 10
    for epoch in range(num_epochs):
        for i, (source_image, target_image) in enumerate(Loader):
            gen_input = Variable(source_image).to(device)
            target_image = Variable(target_image).to(device)
            batch_size = len(gen_input)
            real_label = torch.from_numpy(np.ones(batch_size).astype(np.float32)).to(device)
            fake_label = torch.from_numpy(np.zeros(batch_size).astype(np.float32)).to(device)
            G.train()
            F.train()
            Dx.train()
            Dy.train()
            '''
                train the discriminators first
            '''
            Dx.zero_grad()
            Dy.zero_grad()
            # on real data
            output_x = Dx(gen_input)
            output_y = Dy(target_image)
            real_loss_dx = Loss_func(output_x, real_label)
            real_loss_dy = Loss_func(output_y, real_label)
            real_loss_dx.backward()
            real_loss_dy.backward()
            # on fake data

            gen_data_y = G(gen_input)
            gen_data_x = F(target_image)
            fake_output_x = Dx(gen_data_x)
            fake_output_y = Dy(gen_data_y)
            fake_loss_x = Loss_func(fake_output_x, fake_label)
            fake_loss_y = Loss_func(fake_output_y, fake_label)
            fake_loss_x.backward(retain_graph = True)
            fake_loss_y.backward(retain_graph = True)
            dxOptim.step()
            dyOptim.step()

            '''
                need to train the generators
            '''
            G.zero_grad()
            F.zero_grad()
            fake_output_x = Dx(gen_data_x)
            fake_output_y = Dy(gen_data_y)
            loss_G = Loss_func(fake_output_y, real_label)
            loss_F = Loss_func(fake_output_x, real_label)
            '''
                before stepping back, need to add the cycle losses
            '''

            yi = gen_data_y
            xi = gen_data_x
            hat_x = F(yi)
            hat_y = G(xi)
            idt_loss_G = Cycle_loss(target_image,G(target_image))
            idt_loss_F = Cycle_loss(gen_input, F(gen_input))
            forward_cycle_loss = Cycle_loss(gen_input, hat_x)
            backward_cycle_loss = Cycle_loss(target_image,hat_y)
            loss_G +=forward_cycle_loss*l1 + backward_cycle_loss*l2 + l1*idt_loss_G
            loss_G += REGULARIZATION * (
                torch.sum(torch.abs(xi[:, :, :, :-1] - xi[:, :, :, 1:])) +
                torch.sum(torch.abs(xi[:, :, :-1, :] - xi[:, :, 1:, :]))
            )
            loss_F +=forward_cycle_loss*l1 + backward_cycle_loss*l2+ l1*idt_loss_F
            loss_F += REGULARIZATION * (
                torch.sum(torch.abs(yi[:, :, :, :-1] - yi[:, :, :, 1:])) +
                torch.sum(torch.abs(yi[:, :, :-1, :] - yi[:, :, 1:, :]))
            )
            loss_G.backward(retain_graph = True)
            loss_F.backward()
            gOptim.step()
            fOptim.step()

            if i % 10 == 0:
                G.eval()
                print('creating test output on iteration {}/{}'.format(i+epoch*len_loader, num_epochs*len_loader))
                test_out = G(val_image)
                test_out = torch.reshape(test_out,(3,512,512))

                fig = plt.figure(figsize=(8,8), dpi = 100)
                ax = fig.subplots()
                print(test_out.max(), test_out.min())
                ax.imshow(test_out.cpu().detach().permute(1,2,0))

                #plt.show(block = True)
                img_fname = 'last_pretrained_test_{}_{}.png'.format(epoch,i)
                os.chdir('outputs')
                plt.savefig(img_fname)
                os.chdir('..')

        gScheduler.step()
        fScheduler.step()
        dxScheduler.step()
        dyScheduler.step()
        if epoch%10 == 0:
            torch.save(G.state_dict(), 'last_pretrained_cycle_G_{}.model'.format(epoch))
            torch.save(F.state_dict(), 'last_pretrained_cycle_F_{}.model'.format(epoch))
            torch.save(Dx.state_dict(), 'last_pretrained_cycle_Dx_{}.model'.format(epoch))
            torch.save(Dy.state_dict(), 'last_pretrained_cycle_Dy_{}.model'.format(epoch))



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    im_channels = 3
    target_im_size = 512
    training_batch_size = 1
    learning_rate = .00002
    weight_decay = 1e-8
    betas = (.5,.99)
    num_blocks = 7 #for 128 x 128images 9 for 256 x 256 and higher
    load_model = True
    print('initializing network')
    G_XY = models.cycleGAN_Generator(im_channels,num_blocks)
    F_YX = models.cycleGAN_Generator(im_channels,num_blocks)
    D_X = models.Patch_discriminator(im_channels, target_im_size)
    D_Y = models.Patch_discriminator(im_channels, target_im_size)
    if load_model:
        print('loading pretrained models')
        G_XY.load_state_dict(torch.load('pretrained_cycle_G_20.model'))
        F_YX.load_state_dict(torch.load('pretrained_cycle_F_20.model'))
        D_X.load_state_dict(torch.load('pretrained_cycle_Dx_20.model'))
        D_Y.load_state_dict(torch.load('pretrained_cycle_Dy_20.model'))
    #vgg = visionmodels.vgg16(pretrained = True)
    #loss_net = models.LossNetwork(vgg)
    #loss_net.to(device)
    #loss_net.eval()
    G_XY.to(device)
    F_YX.to(device)
    D_X.to(device)
    D_Y.to(device)
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    gOptim = torch.optim.Adam(G_XY.parameters(),lr= learning_rate, weight_decay=weight_decay, betas = betas)
    fOptim = torch.optim.Adam(F_YX.parameters(),lr= learning_rate, weight_decay=weight_decay, betas = betas)
    dxOptim = torch.optim.Adam(D_X.parameters(),lr= learning_rate, weight_decay=weight_decay, betas = betas)
    dyOptim = torch.optim.Adam(D_Y.parameters(),lr= learning_rate, weight_decay=weight_decay, betas = betas)

    # gOptim = torch.optim.SGD(G_XY.parameters(), lr = learning_rate)
    # fOptim = torch.optim.SGD(F_YX.parameters(), lr = learning_rate)
    # dxOptim = torch.optim.SGD(D_X.parameters(), lr = learning_rate)
    # dyOptim = torch.optim.SGD(D_Y.parameters(), lr = learning_rate)
    def lmbda(epoch):
        if epoch == 30 or epoch == 50 or epoch == 75:
            return .1
        else:
            return 1


    gScheduler = torch.optim.lr_scheduler.MultiplicativeLR(
       gOptim, lr_lambda=lmbda, verbose=True)
    fScheduler = torch.optim.lr_scheduler.MultiplicativeLR(
       fOptim, lr_lambda=lmbda, verbose=True)
    dxScheduler = torch.optim.lr_scheduler.MultiplicativeLR(
       dxOptim, lr_lambda=lmbda, verbose=True)
    dyScheduler = torch.optim.lr_scheduler.MultiplicativeLR(
       dyOptim, lr_lambda=lmbda, verbose=True)

    print('preparing dataset')

    path_2_data = '..\Landmark Detection\data\celeba\Img\img_align_celeba\img_align_celeba'
    im_path_fname = '..\Landmark Detection\data\celeba\Eval\cats_traing.csv'
    transform = transforms.Compose([transforms.Resize(target_im_size + 50),transforms.RandomCrop(target_im_size)])
    target_transform = transforms.Compose([transforms.RandomCrop(target_im_size,pad_if_needed = True ),transforms.Resize(target_im_size)])
    #, transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_loader,dataset = _initialize_dataset(data_path = path_2_data, label_path = im_path_fname,\
                                data_transform = transform,target_transform=target_transform)
    val_image = torch.reshape(torch.Tensor(dataset[25][0]), (1,im_channels, target_im_size, target_im_size)).to(device)


    _train(num_epochs = 100,G = G_XY, F = F_YX,Dx = D_X, Dy = D_Y, loss_net = None,
                gOptim = gOptim, fOptim = fOptim, dxOptim = dxOptim, dyOptim = dyOptim,
                gScheduler = gScheduler,fScheduler = fScheduler,dxScheduler = dxScheduler, dyScheduler = dyScheduler,
                Loss_func = mse_loss,Cycle_loss = l1_loss, Loader = train_loader, val_image = val_image)
