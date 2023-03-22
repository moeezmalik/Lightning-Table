# Running the Scripts to Reproduce Results

After setting up the repository and the environment for the code, the scipts to generate the evaluation results can be run. In total there are 4 scripts included, the details of which will be mentioned below.

## Table Detection Evaluation

This script will run the deep learning models on the evaluation images, compile the results and show them. Since, for this task the deep learning model will be used on above 400 images, this process might take a long time on CPU. You can perform this evaluation in two different ways.

### Using Docker

The docker image needs to be setup before running this command. Please check the [Setup](setup.md) section for that.
```
docker run image_name make evaluation-detection
```

### On Your Own Computer

The requirements need to be installed before running this command. Please check the [Setup](setup.md) section for that.
```
make evaluation-detection
```

## Table Classification Evaluation

This script will train the machine learning models for the table classification task, compile the results and show them. This evaluation can be performed in two different ways.

### Using Docker

The docker image needs to be setup before running this command. Please check the [Setup](setup.md) section for that.
```
docker run image_name make evaluation-classification
```

### On Your Own Computer

The requirements need to be installed before running this command. Please check the [Setup](setup.md) section for that.
```
make evaluation-classification
```

## Complete Pipeline Evaluation
This script will generate the evaluation results for the complete pipeline. Here, the deep learning model will also be used but the number of images will be a lot less as computer to Table Detection so it will not take as long to complete this evaluation. This evaluation can be performed in two different ways.

### Using Docker

The docker image needs to be setup before running this command. Please check the [Setup](setup.md) section for that.
```
docker run image_name make evaluation-complete
```

### On Your Own Computer

The requirements need to be installed before running this command. Please check the [Setup](setup.md) section for that.
```
make evaluation-complete
```

## Run All Evaluations Above

This will run all three evaluations mentioned above one-by-one and show the results from them all collectively. As the deep learning model will be used twice, this evaluation might take very long to run on CPU. As with other evaluations mentioned above, this can be run in two different ways.

### Using Docker

The docker image needs to be setup before running this command. Please check the [Setup](setup.md) section for that.
```
docker run image_name make evaluation-complete
```

### On Your Own Computer

The requirements need to be installed before running this command. Please check the [Setup](setup.md) section for that.
```
make evaluation-complete
```