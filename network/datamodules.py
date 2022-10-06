"""
Name:
    datamodules.py

Description:
    This file was created in order to establish data
    and organise different data modules that will be
    required by PyTorch and PyTorch Lightning for
    loading in the data required to train the models.

Author: 
    Name:
        M. Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

### - IMPORTS - ###

# PyTorch and Lightning
from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import random_split, DataLoader, Subset

# Others
from transforms import Compose, PILToTensor, ConvertImageDtype
from data import CSVDataset
from utils import collate_fn
import random

# Misc
from typing import Optional


### - CLASSES - ###

# Full Table Dataset
class TableDatasetModule(LightningDataModule):
    """
    This class loads the full Table Detection dataset
    that can be used for the training of the PyTorch
    detection networks.

    Args:
        path (str): 
            Path to the directory that contains the following.
            Note: The folder and filenames should be exactly as
            mentioned.

            images:
                A folder that contains all the images.

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

        batch_size (int):
            This is the batch size that will be used when sampling the
            images from the dataset.

        train_eval_split (float):
            This is the argument that will denote the split of the full
            dataset between training and evaluation. It should be a value
            between 0 and 1. For example 0.8 denotes 80% split whereby
            80% of the images will go to the training set and 20% will
            go the evaluation set.

        subset (int):
            If specified, this parameter will take out a subset of data
            from the provided dataset.

        num_workers (int):
            This sets the number of workers to use for the dataloading
            purposes.

        
    """
    def __init__(
        self,
        path: str='./',
        batch_size: int=2,
        train_eval_split: float=0.8,
        subset: int=None,
        num_workers: int=2
    ) -> None:
        
        super().__init__()

        self.path = path
        self.batch_size = batch_size
        self.transforms = Compose([
            PILToTensor(),
            ConvertImageDtype(torch.float)
        ])
        self.train_eval_split = train_eval_split
        self.subset = subset
        self.num_workers = num_workers

        return None

    def prepare_data(self) -> None:
        original_dataset = CSVDataset(self.path, tranforms=self.transforms)
        len_original_data = original_dataset.__len__()

        if self.subset is None:
            # If we dont want a subset of our data

            self.full_dataset = original_dataset
        
        else:
            # If we want a subset of our data then we can generate the required number of
            # random indices and then select the files at those indices

            # Generate Random Indicees in range
            random_indices = random.sample(
                population=range(0, len_original_data),
                k=self.subset
            )

            # Select those random indices
            self.full_dataset = Subset(
                dataset=original_dataset,
                indices=random_indices
            )

    def setup(
        self,
        stage: Optional[str] = None
    ) -> None:
        
        full_set = self.full_dataset
        full_set_length = full_set.__len__()

        train_set_length = round(self.train_eval_split * full_set_length)
        eval_set_length = full_set_length - train_set_length

        print()
        print("Dataset Information")
        print("Total Images in Dataset: " + str(full_set_length))
        print("Number of Training Images: " + str(train_set_length))
        print("Number of Evaluation Images: " + str(eval_set_length))
        print()

        if stage == 'fit' or stage is None:
            self.train_set, self.eval_set = random_split(
                dataset=self.full_dataset,
                lengths=[train_set_length, eval_set_length]
                )

        if stage == 'test' or stage is None:
            self.test_set = full_set

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.eval_set,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers
        )





# Toy Table Dataset