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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
# Sent the model to GPU
net.cuda()
# parallel
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

########################################################################
# 3. Define a Loss function and optimizer

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
lr_decay = 0.995
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

########################################################################
# 4. Train the network
epochs=200
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


## save the model
print('---Saving Model---')
torch.save({'model': net.state_dict(),
            'epochs': epochs, 
            }, 'models/conv_base_{0}epochs.tar'.format(epochs))

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
# Epoch: 10,  loss: 447.060
# Epoch: 20,  loss: 331.624
# Epoch: 30,  loss: 282.397
# Epoch: 40,  loss: 251.856
# Epoch: 50,  loss: 222.285
# Epoch: 60,  loss: 201.380
# Epoch: 70,  loss: 178.609
# Epoch: 80,  loss: 157.810
# Epoch: 90,  loss: 138.466
# Epoch: 100,  loss: 121.376
# Epoch: 110,  loss: 105.195
# Epoch: 120,  loss: 90.178
# Epoch: 130,  loss: 71.308
# Epoch: 140,  loss: 54.969
# Epoch: 150,  loss: 41.211
# Epoch: 160,  loss: 27.562
# Epoch: 170,  loss: 18.831
# Epoch: 180,  loss: 12.333
# Epoch: 190,  loss: 7.691
# Epoch: 200,  loss: 4.914

### accuracy report
# Accuracy of the network on the 10000 test images: 72 %
# Accuracy of plane : 62 %
# Accuracy of   car : 76 %
# Accuracy of  bird : 76 %
# Accuracy of   cat : 40 %
# Accuracy of  deer : 53 %
# Accuracy of   dog : 46 %
# Accuracy of  frog : 61 %
# Accuracy of horse : 83 %
# Accuracy of  ship : 85 %
# Accuracy of truck : 64 %


#########################
# For 300 epochs
Epoch: 10,  loss: 449.185
Epoch: 20,  loss: 341.252
Epoch: 30,  loss: 290.084
Epoch: 40,  loss: 256.611
Epoch: 50,  loss: 227.288
Epoch: 60,  loss: 204.560
Epoch: 70,  loss: 179.661
Epoch: 80,  loss: 160.357
Epoch: 90,  loss: 142.084
Epoch: 100,  loss: 123.785
Epoch: 110,  loss: 106.753
Epoch: 120,  loss: 91.248
Epoch: 130,  loss: 73.023
Epoch: 140,  loss: 57.974
Epoch: 150,  loss: 40.626
Epoch: 160,  loss: 28.594
Epoch: 170,  loss: 18.084
Epoch: 180,  loss: 11.281
Epoch: 190,  loss: 6.815
Epoch: 200,  loss: 4.499
Epoch: 210,  loss: 3.258
Epoch: 220,  loss: 2.487
Epoch: 230,  loss: 2.060
Epoch: 240,  loss: 1.735
Epoch: 250,  loss: 1.511
Epoch: 260,  loss: 1.358
Epoch: 270,  loss: 1.233
Epoch: 280,  loss: 1.144
Epoch: 290,  loss: 1.072
Epoch: 300,  loss: 1.019
Finished Training
---Saving Model---
Accuracy of the network on the 10000 test images: 72 %
Accuracy of plane : 68 %
Accuracy of   car : 100 %
Accuracy of  bird : 84 %
Accuracy of   cat : 40 %
Accuracy of  deer : 46 %
Accuracy of   dog : 53 %
Accuracy of  frog : 61 %
Accuracy of horse : 83 %
Accuracy of  ship : 76 %
Accuracy of truck : 82 %







##########################
# for 100 epochs
Epoch: 10,  loss: 445.848
Epoch: 20,  loss: 342.962
Epoch: 30,  loss: 289.368
Epoch: 40,  loss: 257.188
Epoch: 50,  loss: 229.948
Epoch: 60,  loss: 206.554
Epoch: 70,  loss: 183.773
Epoch: 80,  loss: 163.606
Epoch: 90,  loss: 145.235
Epoch: 100,  loss: 128.619
Finished Training
---Saving Model---
Accuracy of the network on the 10000 test images: 69 %
Accuracy of plane : 68 %
Accuracy of   car : 84 %
Accuracy of  bird : 92 %
Accuracy of   cat : 50 %
Accuracy of  deer : 61 %
Accuracy of   dog : 20 %
Accuracy of  frog : 61 %
Accuracy of horse : 66 %
Accuracy of  ship : 80 %
Accuracy of truck : 70 %

'''