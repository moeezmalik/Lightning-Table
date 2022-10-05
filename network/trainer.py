from ast import parse
from typing import Sequence
from datamodules import TableDatasetModule
from models import RetinaNet

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb

import argparse


def main(args: Sequence = None) -> None:

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Training Script for Training the Networks')

    parser.add_argument('--dataset', help='Path to the dataset.')
    parser.add_argument('--log_name', help='The name that will be used to log to Weights and Biases.', type=str, default=None)
    parser.add_argument('--num_classes', help='Number of classes for the network in the given dataset', type=int, default=2)
    parser.add_argument('--subset', help='Specify a value if a smaller subset of data is required', type=int, default=None)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=50)

    parser = parser.parse_args(args)

    path_to_dataset_folder = parser.dataset
    num_classes = parser.num_classes
    num_epochs = parser.epochs
    log_name = parser.log_name
    subset = parser.subset


    table_datamodule = TableDatasetModule(
        path=path_to_dataset_folder,
        batch_size=2,
        train_eval_split=0.8,
        subset=subset,
        num_workers=2
    )

    model = RetinaNet(
        lr=1e-5,
        num_classes=num_classes,
        pretrained=True,
        batch_size=2
    )

    if log_name is not None:

        wandb_logger = WandbLogger(
            project=log_name
        )

        trainer = Trainer(
            max_epochs=num_epochs,
            accelerator='auto',
            devices='auto',
            log_every_n_steps=1,
            logger=wandb_logger
        )
        
        wandb.login()

    else:

        trainer = Trainer(
            max_epochs=num_epochs,
            accelerator='auto',
            devices='auto',
            log_every_n_steps=1
        )


    trainer.fit(model=model, datamodule=table_datamodule)


    return None

if __name__ == '__main__':
    main()