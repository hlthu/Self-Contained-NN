# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms

########################################################################
# 1. Loading and normalizing CIFAR10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


########################################################################
# 2. Define a Convolution Neural Network

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

## This is the define of the pretrained CONV net
class NewNet(nn.Module):
    def __init__(self):
        super(NewNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        ### This layer is going to be replaced by self-contain
        self.hidden = nn.Linear(256 * 2 * 2, 256 * 2 * 2)
        ####
        self.batchnorm = nn.BatchNorm1d(256 * 2 * 2)
        self.new_fc2 = nn.Linear(256 * 2 * 2, 128)
        self.new_fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 2 * 2)
        x = F.relu(self.hidden(x))
        ###batchnorm
        x = self.batchnorm(x)
        ###
        x = F.relu(self.new_fc2(x))
        x = self.new_fc3(x)
        return x

########################################################################
## load the saved conv model 
net = NewNet()
checkpoint = torch.load('models/conv_base_200epochs.tar')
pretrain_dict = checkpoint['model']

## Let the conv layers doesn't update
net.conv1.requires_grad = False
net.conv2.requires_grad = False
net.conv3.requires_grad = False
net.conv4.requires_grad = False

# Sent the model to GPU
net.cuda()
# parallel
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
model_dict = net.state_dict()
pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
model_dict.update(pretrain_dict)
net.load_state_dict(model_dict)




########################################################################
# 3. Define a Loss function and optimizer

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
lr_decay = 0.995
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

########################################################################
# 4. Train the network
epochs = 200
# epochs=checkpoint['epochs']
for epoch in range(epochs):  # loop over the dataset multiple times, 10 epochs there

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable, and sent them to GPU
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
    # print every epochs
    if epoch % 10 == 9:
        print('Epoch: %d,  loss: %.3f' %(epoch + 1, running_loss))
    learning_rate *= lr_decay
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

print('Finished Training')

########################################################################
# 5. Test the network on the test data


correct = 0
total = 0
for data in testloader:
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

########################################################################
# what are the classes that performed well

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


'''
### the training log

### using the model of 200 epochs and retrain 200 epochs

Epoch: 10,  loss: 66.747
Epoch: 20,  loss: 23.483
Epoch: 30,  loss: 10.506
Epoch: 40,  loss: 4.645
Epoch: 50,  loss: 3.136
Epoch: 60,  loss: 2.002
Epoch: 70,  loss: 1.432
Epoch: 80,  loss: 1.125
Epoch: 90,  loss: 1.005
Epoch: 100,  loss: 0.804
Epoch: 110,  loss: 0.715
Epoch: 120,  loss: 0.721
Epoch: 130,  loss: 0.512
Epoch: 140,  loss: 0.576
Epoch: 150,  loss: 0.602
Epoch: 160,  loss: 0.532
Epoch: 170,  loss: 0.476
Epoch: 180,  loss: 0.474
Epoch: 190,  loss: 0.494
Epoch: 200,  loss: 0.484
Accuracy of the network on the 10000 test images: 73 %
Accuracy of plane : 81 %
Accuracy of   car : 84 %
Accuracy of  bird : 76 %
Accuracy of   cat : 45 %
Accuracy of  deer : 38 %
Accuracy of   dog : 60 %
Accuracy of  frog : 61 %
Accuracy of horse : 66 %
Accuracy of  ship : 80 %
Accuracy of truck : 82 %




####### using the model of 300 epochs and retrain 100 epochs
Epoch: 10,  loss: 73.077
Epoch: 20,  loss: 22.779
Epoch: 30,  loss: 10.092
Epoch: 40,  loss: 4.786
Epoch: 50,  loss: 2.695
Epoch: 60,  loss: 2.078
Epoch: 70,  loss: 1.594
Epoch: 80,  loss: 1.170
Epoch: 90,  loss: 0.954
Epoch: 100,  loss: 0.799
Finished Training
Accuracy of the network on the 10000 test images: 72 %
Accuracy of plane : 68 %
Accuracy of   car : 92 %
Accuracy of  bird : 92 %
Accuracy of   cat : 31 %
Accuracy of  deer : 38 %
Accuracy of   dog : 40 %
Accuracy of  frog : 66 %
Accuracy of horse : 83 %
Accuracy of  ship : 80 %
Accuracy of truck : 88 %

 ######## using the model of 300 epochs and retrain 200 epochs

Epoch: 10,  loss: 72.022
Epoch: 20,  loss: 23.131
Epoch: 30,  loss: 9.501
Epoch: 40,  loss: 4.737
Epoch: 50,  loss: 2.921
Epoch: 60,  loss: 1.976
Epoch: 70,  loss: 1.469
Epoch: 80,  loss: 1.084
Epoch: 90,  loss: 0.854
Epoch: 100,  loss: 0.849
Epoch: 110,  loss: 0.652
Epoch: 120,  loss: 0.599
Epoch: 130,  loss: 0.579
Epoch: 140,  loss: 0.665
Epoch: 150,  loss: 0.467
Epoch: 160,  loss: 0.460
Epoch: 170,  loss: 0.461
Epoch: 180,  loss: 0.498
Epoch: 190,  loss: 0.477
Epoch: 200,  loss: 0.404
Finished Training
Accuracy of the network on the 10000 test images: 73 %
Accuracy of plane : 62 %
Accuracy of   car : 84 %
Accuracy of  bird : 84 %
Accuracy of   cat : 40 %
Accuracy of  deer : 46 %
Accuracy of   dog : 53 %
Accuracy of  frog : 61 %
Accuracy of horse : 75 %
Accuracy of  ship : 80 %
Accuracy of truck : 94 %


'''