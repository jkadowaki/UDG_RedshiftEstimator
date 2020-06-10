#!/usr/bin/env python

from __future__ import printfunction

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

import torch as th
import torch.nn as nn
import torch.utils.data
import torchvision.datasets

################################################################################


mnist = torchvision.datasets.MNIST('./datasets/',download=True)
mnist.data.shape



mnist = torchvision.datasets.MNIST('./datasets/',download=True)
mnist.data.shape


train_features = mnist.data.reshape(-1,1,28,28)

train_labels = mnist.targets

# Note: Features need to be in floating-point data type for entering the network.
# The default one is `th.float32`
train_features = train_features.to(th.float32)

mu = train_features.mean()
sigma = train_features.std()
train_features = (train_features-mu)/sigma

n_train = 50000
n_test = len(train_features)-n_train

test_features = train_features[-n_test:]
test_labels = train_labels[-n_test:]

train_features = train_features[:n_train]
train_labels = train_labels[:n_train]
dataset = torch.utils.data.TensorDataset(train_features,train_labels)
test_dataset = torch.utils.data.TensorDataset(test_features,test_labels)
# Okay, now we have some data. 
# It has 60,000 datapoints.
# There are 784 features and 10 classes:

# Note that the inputs structured with shape (n_examples, n_features)
# 
print(train_features.shape, train_labels.shape)
print(test_features.shape, test_labels.shape)


################################################################################

class MyConvNet(nn.Module):
    def __init__(self,n_hidden,out_channels=10):
        """The initialization method used when building the network."""
        super().__init__()
        # Fully Connected Layers
        
        nh = n_hidden  # number of hidden layers
        ks = 3         # kernel size
        ps = (ks-1)//2 # padding size
        
        # Start with a 1x28x28 image
        
        # Pool it over 2x2 regions
        # Shape 1x14x14
        self.pool0 = nn.AvgPool2d(kernel_size=2)
        
        # Apply 3x3 convolution
        # Shape n_hidden x 14 x 14
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=nh,
                               kernel_size=ks,
                               padding=ps)
        # Batch Normalization -- does not effect the size of the image.
        self.bn1 = nn.BatchNorm2d(nh)
        
        # Pool it over 2x2 regions
        # Shape n_hidden x 7 x 7
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution with less padding,
        # Shape n_hidden x 5 x 5
        self.conv2 = nn.Conv2d(in_channels=nh,
                               out_channels=nh,
                               kernel_size=ks,
                               padding=(ps-1))
        self.bn2 = nn.BatchNorm2d(nh)

        # Convolution with less padding,
        # Shape n_hidden x 3 x 3
        self.conv3 = nn.Conv2d(in_channels=nh,
                               out_channels=nh,
                               kernel_size=ks,
                               padding=(ps-1))
        self.bn3 = nn.BatchNorm2d(nh)

        # Convolution with less padding to 10 features, 1 for each class.
        # Shape out_channels x 1 x 1
        self.conv4 = nn.Conv2d(in_channels=nh,
                               out_channels=out_channels, 
                               kernel_size=ks,
                               padding=(ps-1))

        
    def forward(self,x):
        x = self.pool0(x)

        x = self.conv1(x)
        x = self.bn1(x).relu()
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x).relu()
        
        x = self.conv3(x)
        x = self.bn3(x).relu()

        x = self.conv4(x)

        return x.reshape(-1,10)


################################################################################



net = MyConvNet(n_hidden=16)

# The only purpose of this was to see if the shapes
# in the conv net work right while I was debugging.
with torch.no_grad():
    net(train_features[:2])

n_params = sum(p.nelement() for p in net.parameters())
print("Nunber of parameters:",n_params)
print(net)

for name, m in net.named_modules():
    if isinstance(m,nn.Conv2d):
        print("Initializing",name)
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    else:
        print("Not initializing",name)



list(net.bn1.named_parameters())

list(net.bn1.buffers())



################################################################################



%%time 

batch_size = 50
n_epochs = 10
learning_rate = 1e-3

dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size)

opt = torch.optim.Adam(net.parameters(),learning_rate)

loss_fn = th.nn.CrossEntropyLoss()

net.train() # <-- Set the batchnorm layers to training mode
# This works by sending a signal to this module
# and all submodules to go into training mode

epoch_loss_list = []
for epoch in range(n_epochs):
    loss_list = []
    print("Epoch",epoch,end='\t')
    for x,ytrue in dataloader:
        yhat = net(x)
        loss = loss_fn(yhat,ytrue)
        opt.zero_grad()
        loss.backward()
        opt.step()    
        loss_list.append(loss.item())

    epoch_loss_list.append(np.mean(loss_list))
    print("Loss:",epoch_loss_list[-1])
    
print("Done!")

plt.plot(epoch_loss_list)
plt.xlabel("Epoch number")
plt.ylabel("Loss")
plt.show()


################################################################################


# This call sets the network to 'evaluation' mode, which effects
# batchnorm layers slightly. In training mode, the network can't
# operate on a single example at a time, it needs batches to be
# well defined.
net.eval()
with torch.no_grad(): # `torch.no_grad` explained in automatic differentiation
    all_outs = net(test_features)
test_prediction = all_outs.argmax(dim=1)

accuracy = sklearn.metrics.accuracy_score(test_labels,test_prediction)
print("CONVOLUTIONAL NETWORK")
print("Overall accuracy:",accuracy*100,"%")
print("Classification Report:")
print(sklearn.metrics.classification_report(test_labels,test_prediction,digits=4))

print("Parameter count:", sum(p.nelement() for p in net.parameters()))
for name,param in net.named_parameters():
    print(list(param.shape),name)


################################################################################



%%time 

lrnet = nn.Linear(784,10)

batch_size = 50
n_epochs = 10
learning_rate = 1e-3

fc_dataset = torch.utils.data.TensorDataset(train_features.reshape(-1,784),train_labels)
dataloader = torch.utils.data.DataLoader(fc_dataset,batch_size=batch_size)

opt = torch.optim.Adam(lrnet.parameters(),learning_rate) #lrnet

loss_fn = th.nn.CrossEntropyLoss() #This is the loss for logistic regression

epoch_loss_list = []
for epoch in range(n_epochs):
    loss_list = []
    print("Epoch",epoch)
    for x,ytrue in dataloader:
        yhat = lrnet(x)   # lrnet
        loss = loss_fn(yhat,ytrue)
        opt.zero_grad()
        loss.backward()
        opt.step()    
        loss_list.append(loss.item())

    epoch_loss_list.append(np.mean(loss_list))
    print("\tLoss:",epoch_loss_list[-1])
    
print("Done!")

plt.plot(epoch_loss_list)
plt.xlabel("Epoch number")
plt.ylabel("Loss")
plt.show()

net.eval()
with torch.no_grad():
    all_outs = lrnet(test_features.reshape(-1,784)) #FCNET

test_prediction = all_outs.argmax(dim=1) # Find the prediction with highest probability

accuracy = sklearn.metrics.accuracy_score(test_labels,test_prediction)
print("LOGISTIC REGRESSION")
print("Overall accuracy:",accuracy*100,"%")
print("Classification Report:")
print(sklearn.metrics.classification_report(test_labels,test_prediction,digits=4))


################################################################################

################################################################################
