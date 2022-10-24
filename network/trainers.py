"""
Name:
    trainers.py

Description:
    This file contains a selection of Trainers that setup
    the their respective neural network models and data
    modules. Then they train those models.

Author: 
    Name:
        M. Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

from distutils.command.config import config
from typing import Sequence, Dict
from datamodules import TableDatasetModule
from models import VanillaRetinaNet, VanillaRetinaNetV2

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml

import argparse

class ExperimentConfig():
    """
    This class will help in loading the experiment configurations from the
    YAML file. A YAML file with the experiment configurations will help
    in keeping track of models and their relevant configuations that were
    performing well throughout the experiment process.

    Parameters:
        config_file_path (str):
            Path where the configuration file is saved.

        config_name (str):
            This will be the name of the configuration that needs to be
            loaded from the YAML file.
    """
    def __init__(
        self,
        config_file_path: str = None,
        config_name: str = None
        ) -> None:
        
        self.name = None
        self.model = None
        self.datamodule = None
        self.learning_rate = None
        self.batch_size = None
        self.train_eval_split = None

        # Load and error check the configurations
        if config_file_path is not None:

            dataloaded = self._load_config_file(config_file_path)
            if dataloaded is None:
                return None

        else:
            print("Error: No experiment config file specified!")
            return None

        # Select and error check the specific configuration requested
        if config_name is not None:
            
            config_dict = dataloaded.get(config_name, None)

            # Load the individual elements
            if config_dict is not None:
                experiment_name = config_dict.get("name", "default_config")
                model_name = config_dict.get("model", "VanillaRetinaNet")
                datamodule_name = config_dict.get("datamodule", "TableDatasetModule")
                learning_rate = config_dict.get("learning_rate", 1e-5)
                batch_size = config_dict.get("batch_size", 2)
                train_eval_split = config_dict.get("train_eval_split", 0.8)

            # Throw an error if the configuration is not found
            else:
                print("Configuration name specified not found")
                return None

        else:
            print("Error: No experiment config name specified!")
            return None

        model = self._validate_model(model_name)

        if model is not None:
            self.model = model
        else:
            return None

        datamodule = self._validate_datamodule(datamodule_name)

        if datamodule is not None:
            self.datamodule = datamodule
        else:
            return None

        self.name = experiment_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_eval_split = train_eval_split
        


    def _load_config_file(self, path_to_config: str = None) -> Dict:
        """
        This function will read the YAML configurations from the file and
        return the Python object with all the read configurations.

        Parameters:
            path_to_config (str):
                This is the path to the configuration file that contains
                all the experiment configurations.

        Returns:
            yaml_config (Object):
                A python object that contains the read configurations
                from the YAML file.
        """

        data_loaded = None

        # Open the file and try to read the content, in case
        # of an error, return None

        with open(path_to_config, 'r') as stream:
            try:
                data_loaded = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        return data_loaded

    def _validate_model(self, model_name: str = None) -> LightningModule:
        """
        This is an internal function that will take in the model name
        specified in the configuration file and try to match and return
        the available models to train.

        Parameters:
            model_name (str):
                This is the name of the model in the string format. This
                parameter will be used to match with the models available
                for training.

        Returns:
            model (LightningModule):
                Returns the model that was requested in the configuration
                file. In case of an error, none is returned.
        """
        
        if model_name == "VanillaRetinaNet":
            return VanillaRetinaNet

        elif model_name == "VanillaRetinaNetV2":
            return VanillaRetinaNetV2
        
        else:
            return None

    def _validate_datamodule(self, datamodule_name: str = None) -> LightningDataModule:
        """
        This is an internal function that will take in the datamodule name
        and return the appropriate implementation. This function is very
        similar to the validate model function.

        Parameters:
            datamodule_name (str):
                Name of the Lightning Data Module to get.
        
        Returns:
            datamodule (LightningDataModule):
                If datamodule implementation is found then it will be returned
                otherwise a None type is returned.
        """


        if datamodule_name == "TableDatasetModule":
            return TableDatasetModule
        
        else:
            return None

        

class VanillaRetinaNetTrainer():
    """
    This class implements the trainer for the Vanilla RetinaNet.
    It is the RetinaNet that is available directly from PyTorch.
    The only change made was to reduce the number of classes to
    2. Since that is all we need. One class for background and
    one class for the tables.

    Args:
        path_to_dataset (str):
            Path to the directory that contains the dataset on
            which to train the model. The folder should contain
            the following.
            Note: The folder and filenames should be exactly as
            mentioned.

            images:
                A folder that contains all the images

            annotations.csv:
                Annotations CSV file that contains in the following
                format:
                image_id,x1,y1,x2,y2,class_name

            classes.csv:
                Class list CSV file that contains the list of class
                in the following format:
                class_name,class_id
                
                The object classes should start from 1 and not 0, 0
                is reserved for background classes, if the dataset
                contains examples for the background class then 0
                class_id can be used.

        path_to_save_ckpt (str):
            This is the folder where the checkpoints regarding the
            models will be saved.

        num_epochs (int):
            The number of epochs to train.
        
        subset (int):
            The subset of data to train on. Do not specify this if
            a subset of the data is not required.

        train_eval_split (float):
            This is the argument that will denote the split of the full
            dataset between training and evaluation. It should be a value
            between 0 and 1. For example 0.8 denotes 80% split whereby
            80% of the images will go to the training set and 20% will
            go the evaluation set.

        num_classes (int):
            The number of classes the dataset has.

        log_name (str):
            The project name that will be used for logging to Weights
            and Biases. Do not specify if logging is not required.
    """

    def __init__(
        self,
        path_to_dataset: str = None,
        path_to_save_ckpt: str = None,
        num_epochs: int = 50,
        subset: int = None,
        train_eval_split: float = 0.8,
        num_classes: int = 2,
        log_name: str = None
    ) -> None:
        

            self.datamodule = TableDatasetModule(
                path=path_to_dataset,
                batch_size=2,
                train_eval_split=train_eval_split,
                subset=subset,
                num_workers=8
            )

            self.model = VanillaRetinaNet(
                lr=1e-5,
                num_classes=num_classes,
                pretrained=True,
                batch_size=2
            )

            # Save the best checkpoint according to the evaluation average IoU
            chkpnt_best_avgiou = ModelCheckpoint(
                monitor="val/epoch/avg_iou",
                mode="max",
                filename="chkpnt-best-avgiou-{epoch}",
                dirpath=path_to_save_ckpt
            )

            # Save the best checkpoint according to the evaluation precision
            chkpnt_best_precision = ModelCheckpoint(
            monitor="val/epoch/precision_75_90",
            mode="max",
            filename="chkpnt-best-precision-{epoch}",
            dirpath=path_to_save_ckpt
            )

            # Save the best checkpoint according to the evaluation precision
            chkpnt_best_recall = ModelCheckpoint(
            monitor="val/epoch/recall_75_90",
            mode="max",
            filename="chkpnt-best-recall-{epoch}",
            dirpath=path_to_save_ckpt
            )

            # Save the last checkpoint so that it can be resumed
            chkpnt_last_epoch = ModelCheckpoint(
                filename="chkpnt-last-{epoch}",
                dirpath=path_to_save_ckpt
            )

            callbacks_assembled = [chkpnt_last_epoch, chkpnt_best_avgiou, chkpnt_best_precision, chkpnt_best_recall]

            if log_name is not None:

                wandb_logger = WandbLogger(
                    project=log_name
                )

                self.trainer = Trainer(
                    max_epochs=num_epochs,
                    accelerator='auto',
                    devices='auto',
                    log_every_n_steps=1,
                    callbacks=callbacks_assembled,
                    logger=wandb_logger
                )

            else:

                self.trainer = Trainer(
                    max_epochs=num_epochs,
                    accelerator='auto',
                    devices='auto',
                    callbacks=callbacks_assembled,
                    log_every_n_steps=1
                )

    def start_training(self):
        """
        This is a very simple function that takes all the setup components
        from the constructor function and starts the training process.
        """
        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule
        )

class GenericTrainer():
    """
    This class implements a generic trainer class for the models
    that are available for training. The difference between this
    trainer and the VanillaRetinaNetTrainer is that it takes
    additional parameter for the experiment configuration.

    Parameters:
        path_to_dataset (str):
            Path to the directory that contains the dataset on
            which to train the model. The folder should contain
            the following.
            Note: The folder and filenames should be exactly as
            mentioned.

            images:
                A folder that contains all the images

            annotations.csv:
                Annotations CSV file that contains in the following
                format:
                image_id,x1,y1,x2,y2,class_name

            classes.csv:
                Class list CSV file that contains the list of class
                in the following format:
                class_name,class_id
                
                The object classes should start from 1 and not 0, 0
                is reserved for background classes, if the dataset
                contains examples for the background class then 0
                class_id can be used.

        path_to_save_ckpt (str):
            This is the folder where the checkpoints regarding the
            models will be saved.

        num_epochs (int):
            The number of epochs to train.
        
        subset (int):
            The subset of data to train on. Do not specify this if
            a subset of the data is not required.

        train_eval_split (float):
            This is the argument that will denote the split of the full
            dataset between training and evaluation. It should be a value
            between 0 and 1. For example 0.8 denotes 80% split whereby
            80% of the images will go to the training set and 20% will
            go the evaluation set.

        num_classes (int):
            The number of classes the dataset has.

        log_name (str):
            The project name that will be used for logging to Weights
            and Biases. Do not specify if logging is not required.
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig = None,
        path_to_dataset: str = None,
        path_to_save_ckpt: str = None,
        num_epochs: int = 50,
        subset: int = None,
        num_classes: int = 2,
        log_name: str = None
    ) -> None:

            Model = experiment_config.model
            DataModule = experiment_config.datamodule

            self.datamodule = DataModule(
                path=path_to_dataset,
                batch_size=int(experiment_config.batch_size),
                train_eval_split=float(experiment_config.train_eval_split),
                subset=subset,
                num_workers=2
            )

            self.model = Model(
                lr=float(experiment_config.learning_rate),
                num_classes=num_classes,
                batch_size=int(experiment_config.batch_size)
            )

            # Save the best checkpoint according to the evaluation average IoU
            chkpnt_best_avgiou = ModelCheckpoint(
                monitor="val/epoch/avg_iou",
                mode="max",
                filename="chkpnt-best-avgiou-{epoch}",
                dirpath=path_to_save_ckpt
            )

            # Save the best checkpoint according to the evaluation precision
            chkpnt_best_precision = ModelCheckpoint(
            monitor="val/epoch/precision_75_90",
            mode="max",
            filename="chkpnt-best-precision-{epoch}",
            dirpath=path_to_save_ckpt
            )

            # Save the best checkpoint according to the evaluation precision
            chkpnt_best_recall = ModelCheckpoint(
            monitor="val/epoch/recall_75_90",
            mode="max",
            filename="chkpnt-best-recall-{epoch}",
            dirpath=path_to_save_ckpt
            )

            # Save the last checkpoint so that it can be resumed
            chkpnt_last_epoch = ModelCheckpoint(
                filename="chkpnt-last-{epoch}",
                dirpath=path_to_save_ckpt
            )

            callbacks_assembled = [chkpnt_last_epoch, chkpnt_best_avgiou, chkpnt_best_precision, chkpnt_best_recall]

            if log_name is not None:

                wandb_logger = WandbLogger(
                    project=log_name,
                    tags = [experiment_config.name]
                )

                self.trainer = Trainer(
                    max_epochs=num_epochs,
                    accelerator='auto',
                    devices='auto',
                    log_every_n_steps=1,
                    callbacks=callbacks_assembled,
                    logger=wandb_logger
                )

            else:

                self.trainer = Trainer(
                    max_epochs=num_epochs,
                    accelerator='auto',
                    devices='auto',
                    callbacks=callbacks_assembled,
                    log_every_n_steps=1
                )

    def start_training(self):
        """
        This is a very simple function that takes all the setup components
        from the constructor function and starts the training process.
        """
        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule
        )


def main(args: Sequence = None) -> None:

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Training Script for Training the Networks')

    parser.add_argument('--dataset_path', help='Path to the dataset.', type=str, default="./")
    parser.add_argument('--checkpoint_path', help='Location to save the checkpoints.', type=str, default="./")
    parser.add_argument('--epochs', help='Number of epochs.', type=int, default=50)
    parser.add_argument('--subset', help='Specify a value if a smaller subset of data is required.', type=int, default=None)
    parser.add_argument('--log_name', help='The name that will be used to log to Weights and Biases.', type=str, default=None)
    parser.add_argument('--num_classes', help='Number of classes for the network in the given dataset.', type=int, default=2)
    parser.add_argument('--config_path', help = 'Path to the file that contains all experiment configurations', type=str, default=None)
    parser.add_argument('--config_name', help='Name of the experiment configuration to use', type=str, default=None)

    parser = parser.parse_args(args)

    config_path = parser.config_path

    if config_path is None:
        print("Configuration Path not specified!")
        return None

    config_name = parser.config_name

    if config_name is None:
        print("Configuration Name not specified!")
        return None

    ec = ExperimentConfig(config_file_path=config_path, config_name=config_name)

    if (ec is None):
        print("Error: Something went wrong with Experiment Configurations")
        return None
    

    print()
    print("Experiment Configuration: ")
    print("Experiment Name: " + str(ec.name))
    print("Model: " + str(ec.model))
    print("Datamodule: " + str(ec.datamodule))
    print("Learning Rate: " + str(ec.learning_rate))
    print("Batch Size: " + str(ec.batch_size))
    print("Train Eval Split: " + str(ec.train_eval_split))
    print()

    # Setup the required trainer
    trainer = GenericTrainer(
        experiment_config=ec,
        path_to_dataset=parser.dataset_path,
        path_to_save_ckpt=parser.checkpoint_path,
        num_epochs=parser.epochs,
        subset=parser.subset,
        num_classes=parser.num_classes,
        log_name=parser.log_name
    )

    # Start the training
    trainer.start_training()

    # # Setup the required trainer
    # trainer = VanillaRetinaNetTrainer(
    #     path_to_dataset=parser.dataset_path,
    #     path_to_save_ckpt=parser.checkpoint_path,
    #     num_epochs=parser.epochs,
    #     subset=parser.subset,
    #     train_eval_split=parser.train_eval_split,
    #     num_classes=parser.num_classes,
    #     log_name=parser.log_name
    # )

    # # Start the training
    # trainer.start_training()

    return None

if __name__ == '__main__':
    main()
