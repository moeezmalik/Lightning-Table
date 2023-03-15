"""
This is the file that implements the command-line script
for training the models.

Author:
    Name:
        Muhammad Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

from datamodules import TableDatasetModule
from models import get_model
import torch
from pytorch_lightning import Trainer

print("Hello, World")

