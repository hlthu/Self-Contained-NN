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
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        ### This layer is going to be replaced by self-contain
        self.hidden = nn.Linear(256 * 2 * 2, 256 * 2 * 2)
        ####
        self.new_fc2 = nn.Linear(256 * 2 * 2, 128)
        self.new_fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.pool(self.bn4(F.relu(self.conv4(x))))
        x = x.view(-1, 256 * 2 * 2)
        ####
        x = F.relu(self.hidden(x))
        ####
        x = F.relu(self.new_fc2(x))
        x = self.new_fc3(x)
        return x

########################################################################
## load the saved conv model 
net = NewNet()
checkpoint = torch.load('models/conv_100epochs.mdl')
pretrain_dict = checkpoint['model']

## Let the conv layers not update
for param in net.parameters():
    param.requires_grad = False
net.hidden = nn.Linear(256 * 2 * 2, 256 * 2 * 2)
net.new_fc2 = nn.Linear(256 * 2 * 2, 128)
net.new_fc3 = nn.Linear(128, 10)

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
learning_rate = 0.1
lr_decay = 0.96
optimizer = optim.SGD([
    {'params': net.module.hidden.parameters() },
    {'params': net.module.new_fc2.parameters() },
    {'params': net.module.new_fc3.parameters() }
    ], lr=learning_rate, momentum=0.9, weight_decay=5e-4)

########################################################################
# 4. Train the network
epochs = 100
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
    optimizer = optim.SGD([
        {'params': net.module.hidden.parameters() },
        {'params': net.module.new_fc2.parameters() },
        {'params': net.module.new_fc3.parameters() }
        ], lr=learning_rate, momentum=0.9, weight_decay=5e-4)

print('Finished Training')

########################################################################
# 5. Test the network on the test data
correct = 0.0
total = 0.0
for data in testloader:
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %.2f %%' % (
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
    print('Accuracy of %5s : %.2f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


'''
################ 
# Training Log
Epoch: 10,  loss: 0.325
Epoch: 20,  loss: 0.323
Epoch: 30,  loss: 0.326
Epoch: 40,  loss: 0.325
Epoch: 50,  loss: 0.316
Epoch: 60,  loss: 0.317
Epoch: 70,  loss: 0.310
Epoch: 80,  loss: 0.318
Epoch: 90,  loss: 0.318
Epoch: 100,  loss: 0.321
Finished Training
Accuracy of the network on the 10000 test images: 82.30 %
Accuracy of plane : 75.00 %
Accuracy of   car : 92.31 %
Accuracy of  bird : 76.92 %
Accuracy of   cat : 59.09 %
Accuracy of  deer : 61.54 %
Accuracy of   dog : 73.33 %
Accuracy of  frog : 72.22 %
Accuracy of horse : 91.67 %
Accuracy of  ship : 90.48 %
Accuracy of truck : 82.35 %
'''