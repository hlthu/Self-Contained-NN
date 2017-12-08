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
import torch.backends.cudnn as cudnn


class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.mpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, 10)
        
    def forward(self, x):
        x = self.mpool(F.relu(self.bn1(self.conv1(x))))
        x = self.mpool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mpool(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.mpool(F.relu(self.bn6(self.conv6(x))))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.mpool(F.relu(self.bn8(self.conv8(x))))
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


net = VGG11()
# Sent the model to GPU
net.cuda()
# parallel
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
# cudnn
cudnn.benchmark = True


########################################################################
# 3. Define a Loss function and optimizer

import torch.optim as optim

epochs = 300
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
if epochs == 200:
    lr_decay = 0.976
elif epochs == 300:
    lr_decay = 0.984
else:
    lr_decay = 0.95
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

########################################################################
# 4. Train the network
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

torch.save({'model': net.state_dict(),
            'epochs': epochs, 
            }, 'models/vgg11_{0}epochs.mdl'.format(epochs))

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
    print('Accuracy of %5s : %2.2f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


'''
############### 200 epochs ##################
Epoch: 10,  loss: 61.900
Epoch: 20,  loss: 24.288
Epoch: 30,  loss: 15.763
Epoch: 40,  loss: 13.008
Epoch: 50,  loss: 11.069
Epoch: 60,  loss: 6.402
Epoch: 70,  loss: 3.485
Epoch: 80,  loss: 2.490
Epoch: 90,  loss: 0.151
Epoch: 100,  loss: 0.211
Epoch: 110,  loss: 0.218
Epoch: 120,  loss: 0.215
Epoch: 130,  loss: 0.215
Epoch: 140,  loss: 0.210
Epoch: 150,  loss: 0.209
Epoch: 160,  loss: 0.210
Epoch: 170,  loss: 0.205
Epoch: 180,  loss: 0.206
Epoch: 190,  loss: 0.207
Epoch: 200,  loss: 0.204
Epoch: 210,  loss: 0.208
Epoch: 220,  loss: 0.206
Epoch: 230,  loss: 0.204
Epoch: 240,  loss: 0.206
Epoch: 250,  loss: 0.206
Epoch: 260,  loss: 0.203
Epoch: 270,  loss: 0.205
Epoch: 280,  loss: 0.203
Epoch: 290,  loss: 0.204
Epoch: 300,  loss: 0.205
Finished Training
Accuracy of the network on the 10000 test images: 86.39 %
Accuracy of plane : 87.50 %
Accuracy of   car : 100.00 %
Accuracy of  bird : 100.00 %
Accuracy of   cat : 68.18 %
Accuracy of  deer : 92.31 %
Accuracy of   dog : 80.00 %
Accuracy of  frog : 72.22 %
Accuracy of horse : 91.67 %
Accuracy of  ship : 100.00 %
Accuracy of truck : 88.24 %
'''