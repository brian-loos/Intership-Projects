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
import vgg
#we can use this to set our device. If we have a GPU available we should probably use it.
#currently the code is not written to make use of mulitple GPUs and will default to the
#first available GPU
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #number of epochs to train for
    num_epochs = 1 #found sufficient convergence after 1 epoch of training

    #batch sizes for training and validation
    training_batch_size = 64 #I found that a batch size of 64 led to sufficient convergence quickly
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
    vgg19 = vgg.VGG( type = '16',num_classes = 100,in_channels = 3)
    vgg19.to(device)


    #we want to use Cross entropy loss for multiple output classification
    loss_func = nn.CrossEntropyLoss()
    #we can use any gradient descent method. This model trains easily
    #so we want to accuracy of SGD over the speed of Adams gradient
    #descent
    init_learning_rate = .01
    optimizer = optim.SGD(vgg19.parameters(),lr = init_learning_rate, momentum = .9)
    step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=.1, verbose=False)

    def train(num_epochs, model ,loaders,optimizer,
              loss_func,prints_per_epoch,lr_step, output_loss = False
             ):
        total_step = len(loaders['train'])#parameter for printing output
        model.train()
        #empty GPU cache before training if using
        if (device == torch.device('cuda:0')):
            torch.cuda.empty_cache()

        #create numpy array to store running errors in if requested
        #ie for creating plot of losses after training etc...
        if output_loss:
            err = np.zeros(total_step*num_epochs)
        for epoch in range(num_epochs):
            for i ,(images,labels) in enumerate(loaders['train']):

                #send our images and training labels to either the
                #cpu or gpu depending on what we are using
                b_x = Variable(images).to(device)
                b_y = Variable(labels).to(device)
                #compute the model's output
                output = model(b_x)

                #compute the loss
                loss = loss_func(output,b_y)

                #zero out the gradient and update our weights
                #and biases
                model.zero_grad()
                loss.backward()
                optimizer.step()

                #print out progress at rate of prints_per_epoch approximately
                if i == 0 & epoch == 0:
                    print('Initial loss : {:.4f}'.format(loss.item()))
                if (i+1) % int(total_step//prints_per_epoch) == 0:
                    print('Epoch [{}/{}], Step [{}/{}], test Loss: {:.4f}'
                            .format(epoch + 1, num_epochs, i + 1, total_step, loss.mean().item()))
                pass

                #create loss array if requested, default does not return
                if output_loss:
                    err[i+epoch*total_step] = loss.mean().item()
            lr_step.step()
        pass

        if output_loss:
            return model, err
        else:
            return model

    #function to plot losses vs. iterations
    def plot_lossses(loss):
        fig,ax = plt.subplots()
        ax.plot(err,lw = .5,label = 'training loss')
        ax.set_xlabel('iterations')
        ax.set_ylabel('Cross Entropy Loss')
        ax.legend()
        ax.set_title('Loss over training')
        plt.show(block = False)
        plt.savefig('resnet_plot_%3.1e_%2.1e.png'%(num_epochs, init_learning_rate))
        plt.close()


        #run the training loop and update the model with trained parameters
    if plot_loss:
        #run training loop
        vgg19, err = train(num_epochs,vgg19,loaders,optimizer, loss_func, 100,step_lr,plot_loss)

        #create output figure
        plot_lossses(err)
    else:
        vgg19 = train(num_epochs,vgg19,loaders,optimizer, loss_func,2,step_lr,plot_loss)
    torch.save(vgg19.state_dict(), 'vgg19.model')
    #compute the validation
    #validate(resnet, loss_func,1,validation_batch_size)
