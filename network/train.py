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

path_to_dataset = "../datasets/cellmodules-1/csv"
path_to_ckpt = "../inference-checkpoints/best-fasterrcnn.ckpt"

model_name = "VanillaFasterRCNN"

dm = TableDatasetModule(
    path=path_to_dataset
)

vanilla_model = get_model(model_name=model_name)
    
# Check if the correct model name was specified
if vanilla_model is None:
    print("Model with the name: " + model_name + " not found.")
    

# Load the model with the checkpoint weights
model = vanilla_model.load_from_checkpoint(path_to_ckpt)

# Move the model to appropriate storage i.e. RAM if only CPU is
# available and VRAM if only GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Loaded Successfully")