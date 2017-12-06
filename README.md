# Self-Contained-NN

## Python Files

Training logs and accuracy report is attached in the final of the files.

* `01_cifar10_conv.py`: Train and save a conv network, which will be used as a feature extractor. The final accuracy is 72% for 200 epochs.

* `02_cifar10_base.py`: Using the conv layers of the above conv network as a feature extractor, which means their values are fixed, only the new defined fully connected layers are updated when training. Using the model trained with 200 epochs above, then again train 200 epochs, the accuracy is 73%.

## Models Dir

In `models/`, there will be some pretrained models.