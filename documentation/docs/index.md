---
hide:
    - navigation
---

# Detecting Tables using Pytorch and Lightning

!!! warning
    This repository is still a work in progress and things might change drastically.

This repository contains the code for detecting tables in PDF documents by using the PyTorch and the Lightning framework. The following image is just an example of passing a PDF through one of the networks in this repository that is trained on detecting the tables. The red bounding-boxes show the areas in the image that the model has predicted as a table.

![Main Table Photo](assets/main-table-photo.png)


## Setting up Dependencies

The requirements for this repository can be found in the `requirements.txt` file in the root of respository. The requirements can be installed using the following command using pip:

```
pip install -r requirements.txt
```

In addition, a dockerfile is provided that can setup the repository in the correct fashion and take care of all the dependecies of the code. The image can be generated using the following command.
```
docker build -t image_name .
```

## Evaluation

This repository and all the code produced in it were designed to create an end-to-end pipeline for extracting tabular information from PDF documents. As evaluation is a critical part of any experimentation, specific code has been created that can perform evaluations of different aspects of the pipeline and reproduce the reesults. If you are interested in running the evaluations for yourself please follow the guide on [Reproducing Evaluation](guides/evaluation/index.md).

## Information about Files

The repository contains many Python files in multiple folders that help make the complete pipeline for extracting tables from PDF documents. This is done in order to make the code more organised and modular and only some of these files are designed to be used by the end-user. For more information about the utilities that are designed to be exposed to the end-user, please check the [Guides](guides/index.md) or the [Tools](tools/index.md) section.
