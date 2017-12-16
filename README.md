# Self-Contained-NN

## Python Files

Training logs and accuracy reports are attached in the tail of the files.

### 4-Layers Convolution Networks

* `01_cifar10_conv.py`: Train and save a conv network with 4 conv layers, which will be used as a feature extractor. The final accuracy is about 88.70% for 200 epochs.

* `01_cifar10_conv_base.py`: Define a new model with the same conv layers as `01_cifar10_conv.py`, but with new FC layers. The conv layers are fixed and not updated when training, for 50 epochs, the result is about 87.44%.

### VGG-11 with 8 Conv Layers

* `02_cifar10_vgg11.py`: The same as `01_cifar10_conv.py`, but using VGG11's conv layers, which have 8 conv layers. When trained 300 epochs, the accuracy is about 90.79%。

* `02_cifar10_vgg11_base.py`: The same as `01_cifar10_conv_base.py`, but using the above 8-layer conv networks as a feature extractor. The accuracy is 90.73%.

### VGG-16 with 13 Conv Layers

* `03_cifar10_vgg16.py`: The same as `02_cifar10_vgg11.py`, but using VGG-16, which has 13 convolution layers. The accuracy is 92.41%.

* `03_cifar10_vgg16_base.py`: The same as `02_cifar10_vgg11_base.py`, but using VGG-16's conv layers as a feature extractor. The accuracy is also 92.41%%.


## To Do

- [x] Fixed the Conv Layers as feature extractor.
- [x] Finetune the Conv Layers (VGG16 should be about 92%).
- [ ] Add self-contained connections.
- [ ] Test some update rules of self-contained layer.

## Reference

* [1] [Train CIFAR10 with PyTorch](https://github.com/hlthu/pytorch-cifar)
