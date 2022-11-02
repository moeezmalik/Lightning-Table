---
hide:
    - navigation
---

# Detecting Tables using Pytorch and Lightning
This repository contains the code for detecting tables in PDF documents by using the PyTorch and the Lightning framework. 

!!! warning
    This repository is still a work in progress and things might change drastically.

## Requirements
The requirements for this repository can be found in the `requirements.txt` file in the root of respository. The requirements can be installed using the following command using pip:

```
pip install -r requirements.txt
```

## Folder Structure of the Repository
The repository folders are structured in the following way.

```
- network
	- datamodules.py
	- models.py
	- trainers.py
	- transforms.py
	- utilities.py
- utils
	- visualisers.py
- requirements.txt
```

- The `network` folder contains all the files that deal for the creation, configuration, training and inference of PyTorch based models.
	- The `datamodules.py` file contains everything related to the management of dataset for the training of the models.
	- The `models.py` file contains all the PyTorch based models e.g. the RetinaNet.
	- The `trainers.py` contains all the necessary assembly of functions to initialise the training of the models. More information about the training of the models is provided below.
	- Since we are dealing with object detection models, the runtime transforms that might be required to apply to the images need to be applied to the annotations i.e. bounding boxes as well. These special transforms are present in the `transforms.py` file.
	- The `utilities.py` file contains helper utilities for the setup of the models.
- The `utils` folder provides some utilities that might be needed to interface with the network. 
	- `visualisers.py` provides some functions for the purposes of visualising the detections made by the network.
- `requirements.txt` file contains the required Python packages to run the code in this repository.
