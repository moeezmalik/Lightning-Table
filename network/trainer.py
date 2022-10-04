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

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--log_name', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--num_classes', help='Number of epochs', type=int, default=2)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=50)

    parser = parser.parse_args(args)

    path_to_dataset_folder = parser.dataset
    num_classes = parser.num_classes
    num_epochs = parser.epochs
    log_name = parser.log_name


    wandb_logger = WandbLogger(
        project=log_name
    )

    table_datamodule = TableDatasetModule(
        path=path_to_dataset_folder,
        batch_size=2,
        train_eval_split=0.8,
        subset=30,
        num_workers=2
    )

    model = RetinaNet(
        lr=1e-2,
        num_classes=num_classes,
        pretrained=True,
        batch_size=2
    )

    trainer = Trainer(
        max_epochs=num_epochs,
        accelerator='auto',
        devices='auto',
        overfit_batches=1,
        log_every_n_steps=1,
        logger=wandb_logger
    )


    wandb.login()
    trainer.fit(model=model, datamodule=table_datamodule)


    return None

if __name__ == '__main__':
    main()