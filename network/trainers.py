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

from typing import Sequence
from datamodules import TableDatasetModule
from models import VanillaRetinaNet

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

import argparse

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
                
                wandb.login()

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
    parser.add_argument('--train_eval_split', help='Specify what amount of dataset to use as training set.', type=float, default=0.8)
    parser.add_argument('--log_name', help='The name that will be used to log to Weights and Biases.', type=str, default=None)
    parser.add_argument('--num_classes', help='Number of classes for the network in the given dataset.', type=int, default=2)

    parser = parser.parse_args(args)

    # Setup the required trainer
    trainer = VanillaRetinaNetTrainer(
        path_to_dataset=parser.dataset_path,
        path_to_save_ckpt=parser.checkpoint_path,
        num_epochs=parser.epochs,
        subset=parser.subset,
        train_eval_split=parser.train_eval_split,
        num_classes=parser.num_classes,
        log_name=parser.log_name
    )

    # Start the training
    trainer.start_training()

    return None

if __name__ == '__main__':
    main()