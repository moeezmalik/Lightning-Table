# Detecting Tables using Pytorch and Lightning

> **WARNING:** This repository is still a work in progress and things might change drastically.

This repository contains the code for detecting tables in PDF documents by using the PyTorch and the Lightning framework. The following image is just an example of passing a PDF through one of the networks in this repository that is trained on detecting the tables. The red bounding-boxes show the areas in the image that the model has predicted as a table.

![Image title](documentation/docs/assets/main-table-photo.png)

## More Information

Please find more information about this repository in the [documentation](https://moeezmalik.github.io/Lightning-Table/).

## Folder Structure of the Repository
The repository folders are structured in the following way.

```
- network
	- datamodules.py
    - files.py
    - infer.py
    - inferencers.py
    - misc.py
	- models.py
    - test.py
    - train.py
	- trainers.py
	- transforms.py
	- utilities.py
    - visualise.py
    - visualisers.py

- utils
	- visualisers.py
    - data-house-keeper.py
    - pascalvoc-to-csv.py
- requirements.txt
- experiment-configs.py
```

More information about the files and folders in the repository:

- The `network` folder contains all the files that deal for the creation, configuration, training and inference of PyTorch based models.
	- The `datamodules.py` file contains everything related to the management of dataset for the training of the models.
    - The `files.py` file contains all the functions required by other scripts to interact with the filesystem.
    - The `infer.py` implements the command-line utility to infer on a single or a folder of PDF files.
    - The `inferencers.py` file contains all the functions required to perform inference using the models provided.
    - The `misc.py` contains miscellaneous functions required by other scripts.
	- The `models.py` file contains all the PyTorch based models e.g. the RetinaNet.
    - The `test.py` is just there to test out the scripts, it does not implement any useful functionality.
    - The `train.py` file implements the command-line utility to train the models.
	- The `trainers.py` contains all the necessary assembly of functions to initialise the training of the models. More information about the training of the models is provided below.
	- Since we are dealing with object detection models, the runtime transforms that might be required to apply to the images need to be applied to the annotations i.e. bounding boxes as well. These special transforms are present in the `transforms.py` file.
	- The `utilities.py` file contains helper utilities for the setup of the models.
    - The `visualise.py` file implements the command-line utility to visualise pdf and image files using the trained models.
    - The `visualisers.py` file contains the functions that enables the visualiation of PDF and image files using the trained models.

- The `utils` folder provides some utilities that might be needed to interface with the network. 
    - The `data-house-keeper.py` file provides functionality to clean up the PascalVOC annotation directory and rename the files.
    - The `files.py` file contains all the functions required by other scripts to interact with the filesystem.
    - The `misc.py` contains miscellaneous functions required by other scripts.
    - The `pascalvoc-to-csv.py` file provides functionality to convert the PascalVOC annotations to CSV annotations.
- `experiment-configs` provides model, datamodule and hyperparameter configurations for the networks in this repository.
- `requirements.txt` file contains the required Python packages to run the code in this repository.

The above was just a summary of all the main files and folders that are in the repository. However, only some of them are designed to be used by the end-user. For more information about the utilities that are designed to be exposed to the end-user, please check the full documentation.
