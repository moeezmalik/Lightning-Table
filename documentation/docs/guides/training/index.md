# How to Train the Networks
The repository will contain the code to train multiple networks on the dataset. The `trainers.py` file can be used for the purposes of training the available models. Currently only the default [PyTorch RetinaNet](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.retinanet_resnet50_fpn.html#torchvision.models.detection.retinanet_resnet50_fpn) is available for the training purposes. 


## Known Issues
Following are some known issues with the code:

- Negative examples do not work while training. They increase the loss to infinity from which the network never recovers.

## Tested On
The code has been tested to run on:

- macOS 12.4 Monterey on Intel Macbook Pro 2019
- Google Colab


