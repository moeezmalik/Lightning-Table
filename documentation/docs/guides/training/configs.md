# Experiment Configurations

This section will give information about the usage of the experiment configurations
that is required for training of the networks.

## Why do we need an experiment configuration file?

While developing a machine learning application, it can be very hard
to keep track of all the parameters that we change trying to attain
higher accuracy. This case is no different. There are a number of
object detection models that can be used for detection of tables in 
the PDF documents, each of these models will have their own set of
hyperparameters that need to be tuned to achieve optimal performance.

Keeping this in mind, an separate yaml file must be created (provided 
with this repository) that includes the parameters, models and data
modules that were used while training. This will also help other
people in reproducing the work that was done previously. 

## Breakdown of an Experiment Configuration

One example of such configuration will be mentioned below, the rest
can be developed easily by following the same pattern.

```
vanilla:
  name: "vanilla"
  model: "VanillaRetinaNet"
  datamodule: "TableDatasetModule"
  learning_rate: 1e-5
  batch_size: 2
  train_eval_split: 0.8
```

As it can be seen from the configuration file, the formatting is in YAML.
The header of the configuration (in this case "vanilla") specifies the name
of the configuration. This name will need to be specified in the [training command](command.md) when starting a trainig run. 

Following are the parameters that need to supplied in an experiment configuration:

- `name`: This is the name of the configuration, this name will also be appended as a tag in the Weights and Biases run. See the [Metrics](metrics.md) for more information.

- `model`: This specifies the name of the model to use. The name of the model should exactly the same as name of the class that implements the model in the `models.py` file in the networks folder.

- `datamodule`: This specifies the datamodule to use for the training run. As for the models, the name should match the name of the class that implements the required datamodule in the `datamodules.py` file in the networks folder.

- `learning_rate`: This will specify the learning rate in the training run.

- `batch_size`: This parameter specifies the number of images that will be fed to the model in each iteration during the training phase.

- `train_eval_split`: This is a float number between the range of 0 - 1 that specifies the split of the dataset between training and evaluation set. The number specified is for the size of the training set. For example: 0.8 will mean 80% of the dataset to use as training set and 20% of the dataset to use as evaluation set.

All of these configurations will be printed on the console when the training run starts so that it can be validated if the correct values were read.