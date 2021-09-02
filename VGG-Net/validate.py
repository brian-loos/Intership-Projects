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
import resnet
#we can use this to set our device. If we have a GPU available we should probably use it.
#currently the code is not written to make use of mulitple GPUs and will default to the
#first available GPU
#first available GPU
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    #number of epochs to train for
    num_epochs = 1 #found sufficient convergence after 1 epoch of training

    #batch sizes for training and validation
    training_batch_size = 128 #I found that a batch size of 64 led to sufficient convergence quickly
    validation_batch_size = 64 #validating over larger swatches at once

    #boolean, whether or not to plot loss vs iterations
    plot_loss = True

    #int, number of times to iterate through the test set of data for validation
    #     we can iterate through it more times to have more confidence in the
    #     trained model
    num_val_tests = 10

    #transform to send PIL images to tensors and resize
    # 28 x 28 --> 32 x 32
    transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
    train_data = datasets.CIFAR100(
        root = 'data',
        train = True,
        transform = transform,
        download = True
    )
    test_data = datasets.CIFAR100(
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
    vgg19 = resnet.VGG(type = '19', num_classes = 100)
    vgg19.state_dict = torch.load('vgg19.model')
    vgg19.to(device)


    def validate(model, num_tests,val_batch_size):
        top_1 = np.zeros([len(loaders['test'])*(num_tests),val_batch_size])
        act_out = np.zeros([len(loaders['test'])*(num_tests),val_batch_size])
        val_loader_size = len(loaders['test'])
        for j in range(num_tests):
            print(torch.cuda.memory_summary(device=device, abbreviated=False))
            for i ,(images,labels) in enumerate(loaders['test']):

                b_x = Variable(images).to(device)
                b_y = Variable(labels).to(device)
                output = model(b_x)
                top_1[i + j*val_loader_size,:len(b_y)]  = np.argmax(output.detach().cpu().numpy(),axis = 1)
                act_out[i + j*val_loader_size,:len(b_y)] = b_y.detach().cpu().numpy()
        print((len(np.where(top_1 - act_out != 0)[1])))
        print(len(loaders['test'].dataset))
        pass_rate = 100 -((len(np.where(top_1 - act_out != 0)[1])/(len(loaders['test'].dataset)*num_tests)))*100
        results = pass_rate
        print('pass rate for trained network = {:3.4f}%'.format(results))



    validate(vgg19,1,validation_batch_size)
