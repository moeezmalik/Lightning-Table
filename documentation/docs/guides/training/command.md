# Training Command

This section will show how to initiate the training and what parameters are
offered in the training command.

## Breakdown of the Training Command
The `trainer.py` file accepts the following arguments when initialising the training:

- `dataset_path`:
  (*Type*: String,
   *Default*: `./`)
  This specifies the path to the dataset. More information about the format of the dataset folder is mentioned in the following sections.

- `checkpoint_path`: 
  (*Type*: String,
   *Default*: `./`)
  This is used to specify the path where the checkpoints should be saved. More information about this is mentioned in the following sections.

- `epochs`:
  (*Type*: Postive Integer,
   *Default*: `50`)
  This specifies the number of epochs to train for.

- `config_path`:
  (*Type*: String,
   *Default*: `None`)
  This is the path to file that contains all of the training configuration settings.
  Please refer to the [Experiment Configurations](configs.md) for more information.

- `config_name`:
  (*Type*: String,
   *Default*: `None`)
  This specifies the name of configuration setting to use from the file the file that
  contains all of the training configuration settings specified above. Please refer to the [Experiment Configurations](configs.md) for more information.

- `subset`:
  (*Type*: Positive Integer,
   *Default*: `None`)
  This specifies if there is a need to take a random subset of the dataset. This can be useful initially when building the model or debugging to remove the bugs. It will be much faster to train on a small subset of the data to detect bugs. Specify `None` to not create subset of the data.

- `log_name`:
  (*Type*: String,
   *Default*: `None`)
  This specifies the name of the project for logging metrics to the [Weights and Biases](https://wandb.ai/site) account. More information about metrics and logging are mentioned in the following sections. Specify `None` to not log anything to Weights and Biases.

- `num_classes`:
  (*Type*: Positive Integer,
   *Default*: `2`)
  This specifies the number of classes in the dataset. For this particular example since there is only one object to be detected this parameter should be specified as 2. Note that 2 classes are specified instead of 1 because that is the PyTorch convention. One class represents the background and one class represents the object to detect i.e. Table.


## Example of the Training Command
An example of training command is as follows. It trains the RetinaNet model for 50 epochs, with only 10 images randomly selected from the dataset, 8 of which will be used as the training images and 2 as evaluation images. It also provides the path for the dataset and the path to save the checkpoint. 

```
python network/trainers.py --dataset_path "datasets/original/" --epochs 50 --checkpoint_path "checkpoint/" --config_path "experiment-configs.yml" --config_name "vanilla_fasterrcnn_v2_1" --log_name "test-test"
```