##These are my personal machine learning import statements, they cover most of what I regularly need for CV
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
import dcgan
import time
from PIL import Image
from IPython.display import display
#we can use this to set our device. If we have a GPU available we should probably use it.
#currently the code is not written to make use of mulitple GPUs and will default to the
#first available GPU
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #number of epochs to train for
    num_epochs = 100 #found sufficient convergence after 1 epoch of training

    #batch sizes for training and validation
    training_batch_size = 128 #I found that a batch size of 64 led to sufficient convergence quickly
    validation_batch_size = 128 #validating over larger swatches at once

    #int, number of times to iterate through the test set of data for validation
    #     we can iterate through it more times to have more confidence in the
    #     trained model
    num_val_tests = 10

    #transform to send PIL images to tensors and resize
    # 28 x 28 --> 32 x 32

    transform = transforms.Compose([transforms.Resize(64),transforms.RandomAffine(10, translate=(.1,.1)),transforms.ToTensor(),transforms.Normalize(0,1)])
    train_data = datasets.MNIST(
        root = 'data',
        train = True,
        transform = transform,
        download = True
    )
    test_data = datasets.MNIST(
        root = 'data',
        train = False,
        transform = transform
    )

    #define a dictionary of loaders on train set and test set
    # if using GPU want to enable pin_memory -->
    #    it allows preloading of datasets into GPU memory which reduces training overhead.
    #    in my experiments disabling pin memory when GPU is used increased training times by
    #    50%
    if (device == torch.device('cuda:0')):
        loaders = {'train': torch.utils.data.DataLoader(train_data, batch_size = training_batch_size, shuffle = True,num_workers = 1, pin_memory = True),
              'test' : torch.utils.data.DataLoader(test_data, batch_size = validation_batch_size, shuffle = True,num_workers = 1, pin_memory = True)}
    else:
        loaders = {'train': torch.utils.data.DataLoader(train_data, batch_size = training_batch_size, shuffle = True,num_workers = 1),
              'test' : torch.utils.data.DataLoader(test_data, batch_size = validation_batch_size, shuffle = True,num_workers = 1)}

    '''
        Setting up the network and training loop
    '''
    torch.cuda.empty_cache()

    noise_dim = 256

    generator = dcgan.Generator(featmap_dim = 32, noise_dim = noise_dim)
    discriminator = dcgan.Discriminator(featmap_dim = 32)

    generator.to(device)
    discriminator.to(device)

    generator.apply(dcgan.weights_init)
    discriminator.apply(dcgan.weights_init)

    #we want to use Cross entropy loss for multiple output classification
    loss_func = nn.BCELoss()
    #we can use any gradient descent method. This model trains easily
    #so we want to accuracy of SGD over the speed of Adams gradient
    #descent
    init_learning_rate = .0001
    optimizer_D = optim.Adam(discriminator.parameters(),lr = init_learning_rate, betas = (.5,.99))
    optimizer_G = optim.Adam(generator.parameters(),lr = init_learning_rate, betas = (.5,.99))
    #optimizer_D = optim.SGD(discriminator.parameters(), lr = init_learning_rate)
    #optimizer_G = optim.SGD(generator.parameters(), lr = init_learning_rate)

    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size = 100, gamma=.1, verbose=False)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size = 100, gamma =.1, verbose = False)

    def train_GAN(epochs, discriminator, generator, loaders,
                  optimizer_D, optimizer_G, Loss_func, scheduler_D,
                  scheduler_G,noise_dim=100,batch_size = 100):
        torch.cuda.empty_cache()

        total_step = len(loaders['train'])

        #fixed noise for generating samples at fixed iteration
        dim_fixed_noise = 5

        fixed_noise = torch.randn(10,dim_fixed_noise,noise_dim,1,1,device = device)

        for epoch in range(epochs):
            for i , data in enumerate(loaders['train'],0):
                #loads true images
                generator.train()
                real_images, _ = data
                real_images = real_images.to(device)
                #get actual batch size (dataset might not be cut evenly)
                act_batch_size = len(real_images)


                '''
                    create labels for the real and fake targets
                '''
                real_label = torch.from_numpy(np.ones(act_batch_size).astype(np.float32)).to(device)
                fake_label = torch.from_numpy(np.zeros(act_batch_size).astype(np.float32)).to(device)



                '''
                train the discriminator first

                '''
                discriminator.zero_grad()

                #real inputs

                real_output =   torch.flatten(discriminator(real_images))
                real_error_D = Loss_func(real_output, real_label)
                real_error_D.backward()
                D_x = real_output.mean().item() #average value given to real image

                #fake images
                noise = torch.randn(act_batch_size,noise_dim,1,1,device = device)
                fake_images = generator(noise)#+torch.randn(act_batch_size,1,32,32,device = device)*.
                #train discriminator on fake images
                fake_output = torch.flatten(discriminator(fake_images.detach()))
                fake_error_D = Loss_func(fake_output, fake_label)
                fake_error_D.backward(retain_graph = True)
                D_G_z1 = fake_output.mean().item()
                #total D error
                discriminator_error = real_error_D + fake_error_D
                '''
                    Now need to train the generator network
                '''

                generator.zero_grad()

                fake_output = torch.flatten(discriminator(fake_images))
                generator_error = Loss_func(fake_output, real_label)
                generator_error.backward()
                D_G_z2 = fake_output.mean().item()
                #update G and D

                optimizer_D.step()
                optimizer_G.step()



                if (i+1) % int(total_step//3) == 0:
                    generator.eval()
                    print('Epoch [{}/{}], Step [{}/{}], Discriminator Loss: {:.8f}, Generator Loss: {:.8f}, Pass Rates (real/fake): {:.8f} {:.8f}'
                           .format(epoch + 1, epochs, i + 1, total_step, discriminator_error.item(),generator_error.item(), D_x,D_G_z2))
                    resize_transform = transforms.Resize(64)

                    full_mat = np.zeros([64*dim_fixed_noise,640])


                    for j in range(10):
                        test_output = generator(fixed_noise[j])
                        #test_output = resize_transform(torch.reshape(test_output,(dim_fixed_noise, 32,32)))#resize output
                        full_mat[:,j*64:(j+1)*64] = test_output.view(dim_fixed_noise*64,64).cpu().detach().numpy()
                    #full_mat = np.where(full_mat>0, full_mat, 0)
                    img = Image.fromarray(np.uint8(full_mat * 255) , 'L')
                    print('Saving Test image\n')
                    img = img.save( 'sample_output_%5.1f.png'%(i + total_step*epoch))

            scheduler_D.step()
            scheduler_G.step()
        return generator, discriminator

    generator, discriminator = train_GAN(num_epochs, discriminator, generator, loaders,
                  optimizer_D, optimizer_G, loss_func, scheduler_D,
                  scheduler_G,noise_dim=noise_dim,batch_size = training_batch_size)

    torch.save(generator.state_dict(), 'generator.model')
    torch.save(discriminator.state_dict(), 'discriminator.model')
    #compute the validation
    #validate(resnet, loss_func,1,validation_batch_size)
