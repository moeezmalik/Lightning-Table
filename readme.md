---
---

# Detecting Tables using Pytorch and Lightning
This repository contains the code for detecting tables in PDF documents by using the PyTorch and the Lightning framework. 

> **WARNING:** This repository is still a work in progress and things might change drastically.

## 1. Folder Structure of the Repository
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


## 2. How to Train the Networks
The repository will contain the code to train multiple networks on the dataset. The `trainers.py` file can be used for the purposes of training the available models. Currently only the default [PyTorch RetinaNet](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.retinanet_resnet50_fpn.html#torchvision.models.detection.retinanet_resnet50_fpn) is available for the training purposes. 


### 2.1. Breakdown of the Training Command
The `trainer.py` file accepts the following arguments when initialising the training:

- `dataset_path`: 
  *Type*: String
  *Default*: `./`
  This specifies the path to the dataset. More information about the format of the dataset folder is mentioned in the following sections.
- `checkpoint_path`: 
  *Type*: String
  *Default*: `./`
  This is used to specify the path where the checkpoints should be saved. More information about this is mentioned in the following sections.
- `epochs`:
  *Type*: Postive Integer
  *Default*: `50`
  This specifies the number of epochs to train for.
- `subset`:
  *Type*: Positive Integer
  *Default*: `None`
  This specifies if there is a need to take a random subset of the dataset. This can be useful initially when building the model or debugging to remove the bugs. It will be much faster to train on a small subset of the data to detect bugs. Specify `None` to not create subset of the data.
- `train_eval_split`:
  *Type*: Positive Float
  *Default*: `0.8`
  This is a float number between the range of 0 - 1 that specifies the split of the dataset between training and evaluation set. The number specified is for the size of the training set. For example: `0.8` will mean 80% of the dataset to use as training set and 20% of the dataset to use as evaluation set.
- `log_name`:
  *Type*: String
  *Default*: `None`
  This specifies the name of the project for logging metrics to the [Weights and Biases](https://wandb.ai/site) account. More information about metrics and logging are mentioned in the following sections. Specify `None` to not log anything to Weights and Biases.
- `num_classes`:
  *Type*: Positive Integer
  *Default*: `2`
  This specifies the number of classes in the dataset. For this particular example since there is only one object to be detected this parameter should be specified as 2. Note that 2 classes are specified instead of 1 because that is the PyTorch convention. One class represents the background and one class represents the object to detect i.e. Table.


### 2.2. Example of the Training Command
An example of training command is as follows. It trains the RetinaNet model for 50 epochs, with only 10 images randomly selected from the dataset, 8 of which will be used as the training images and 2 as evaluation images. It also provides the path for the dataset and the path to save the checkpoint. 

```
python trainers.py --dataset_path "../dataset/" --epochs 50 --subset 10 --checkpoint_path "../checkpoint/"
```


## 3. About the Dataset
The way the code has been setup, it requires the dataset folder to be in a very specific format. The format is mentioned below.

### 3.1. Folder Structure of Dataset
The folder structure should be the following.

> **IMPORTANT:** The name of the files and the folder should be exactly like mentioned here. This is because the names are hard coded so that only the path to root of the dataset folder needs to supplied when training.

```
- dataset
	- images
	- annotations.csv
	- classes.csv
```

- `images`: A folder that contains all the images.
- `annotations.csv`: Annotations CSV file that contains the annotations in the following format: 
  `image_id,x1,y1,x2,y2,class_name`
  Note that the `image_id` should just be name of the file in the images folder and not a path to the image. The dataset implementation in the code takes care of the path automatically. `x1,y1` are the coordinates for the top-left of the box. `x2,y2` are the coordinates for the bottom-right of the box. 
- `classes.csv`: Class list CSV file that contains the list of class in the following format: 
  `class_name,class_id`
  The object classes should start from 1 and not 0, 0 is reserved for background classes, if the dataset contains examples for the background class then 0 class_id can be used.

### 3.2. Some Notes
There are some particularities about the current implementation of dataset that need to be noted:
- There is no separate annotation file for the validation and training set. All the annotations should be mentioned in the `annotations.csv` file. Instead the Evaluation and Training split should be specified as an argument for the `trainer.py` file (See the 'How to Train the Networks' section).
- Negative examples are not allowed. These are images where there is no object to be detected. If such images are present, the loss is drived to infinity. This is a known issue currently for the code.


## 4. About the Checkpoints
The checkpoint is a terminology taken from the PyTorch Lightning framework. It is a file that contains the state of the model. Checkpoints are saved successively throughout the training process. The benefit of keeping checkpoints is that we can keep track and save the model state that performed the best since training of the network is an iterative process. Following checkpoints are saved through the training process:

- Checkpoint for the last epoch
- Checkpoint for the best Evaluation Average IoU
- Checkpoint for the best Evalutaion Precision (at 75% Confidence Score and 90% IoU)
- Checkpoint for the best Evaluation Recall (at 75% Confidence Score and 90% IoU)

## 5. About Logging and Evaluation Metrics
Evaluation metrics are critical in order to evaluate the model performance. It is necessary to keep track if these metrics throughout the training process as well in order make decisions about hyperparameters for example.

### 5.1. Weights and Biases Integration
In order to keep track of the model performance throughout the training process, there is [Weights and Biases](https://wandb.ai/site) integration built into the code. Weights and biases is an MLOps platform that allows us to keep track of Machine Learning experiements.

In order to log values to the Weights and Biases dashboard, a project name needs to be specified when running the training script, see the 'How to Train the Networks' section for more information. In addition to specifying the name of the projects, an account at Weights and Biases is needed. After logging in, the secret key will be available at [https://wandb.ai/authorize](https://wandb.ai/authorize). Enter this key when prompted while running the training script. 

### 5.2. Logged Metrics
Different metrics are logged during the training and the validation phases. These can be visualised on the Weights and Biases dashboard. 

#### 5.2.1. During training
During the training process, following quantities are logged.
- **Classification Loss:** This is loss that the network calculates when classifying the region of objects into classes. For our case there is only one class of interest i.e. Table hence this loss goes down fairly quickly during the training phase.
- **Regression Loss:** This represents the loss that is calculated when the network tries to draw the bounding boxes around the tables. For us this quantity is more representative of the network training success.
- **Total Loss:** This is just the sum of both the classification and regression loss and provides the overview of the network training performance.
- **Mean Total Loss:** This is just the total loss averaged over the entire epoch.
  
#### 5.2.2. During Evaluation
During the evaluation of the dataset, following quantities are logged.
- **Averaged IoU:** During the evaluation phase, the network is asked to make bounding box predictions over an image. Those box predictions are taken and compared with the ground truth bounding boxes. An overlap is calculated using a measure called IoU (Intersection of Union). 
- **Precision:** During the evaluation phase, the network tries its best to make predictions of where the tables are. The predictions are the bounding boxes. Each of these bounding boxes come with their confidence scores. We select some of these boxes with a certain confidence score threshold. In addition to the confidence score, we also compute the overlap of these boxes with the ground truths and only select the boxes with a certain overlap threshold. So after both of these thresholds applied, we will have final set of detected tables. Precision value will tell us how many of these detected tables were actually tables. 
- **Recall:** The process of calculating the recall metric is very similar to precision as described above. After applying both the confidence score and IoU thresholds, we get a set of predicted detections for the table. Recall tells us how many of the tables were actually detected by the network. 

## 6. Known Issues
Following are some known issues with the code:
- Negative examples do not work while training. They increase the loss to infinity from which the network never recovers.


## 7. Changelog
- October 11, 2022 - The ability to train RetinaNet on the table dataset has been added.

## 8. Requirements
The requirements for this repository can be found in the `requirements.txt` file in the root of respository. The requirements can be installed using the following command using pip:

```
pip install -r requirements.txt
```

## 9. Tested On
The code has been tested to run on
- macOS 12.4 Monterey on Intel Macbook Pro
- Google Colab