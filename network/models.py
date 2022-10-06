"""
Name:
    models.py

Description:
    This file was created to house all of the models that will
    be trained and utilised for the purposes of detecting
    tables in the PDF files.

Author: 
    Name:
        M. Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

### - IMPORTS - ###

# PyTorch and Lightning
from pytorch_lightning import LightningModule

import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2, RetinaNetHead
from torchvision.ops import box_iou
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Others
from utilities import evaluate_iou

# Misc
from typing import Any

### - CLASSES - ###

# RetinaNet
class VanillaRetinaNet(LightningModule):
    """
    This class implements the RetinaNet using PyTorch and the higher
    level wrapper PyTorch Lighting modules

    The network implementation follows the paper:
        https://arxiv.org/abs/1708.02002

    The code is heavily borrowed from the PyTorch Lightning Bolt Implementations
    which can be found here:
        https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/detection/retinanet/retinanet_module.py

    During training, the model expects:
        images (List of Tensors [N, C, H, W]):
            List of tensors, each of shape [C, H, W], one for each image, and should be in 0-1 range.
            Different images can have different sizes.
        targets (List of Dictionaries):
            boxes (FloatTensor[N, 4]):
                The ground truth boxes in `[x1, y1, x2, y2]` format.
            labels (Int64Tensor[N]):
                The class label for each ground truh box.

    Args:
        lr (float):
            This is the learning rate that will be used when training the model
        num_classes (int):
            These are the number of classes that the data has
        pretrained (bool):
            If set to true, RetinaNet will be generated with pretrained weights
        batch_size (int):
            This is the batch size that is being used with the data
        

    """
    def __init__(
        self,
        lr: float = 0.0001,
        num_classes: int = 91,
        pretrained: bool = False,
        batch_size: int = 2
    ) -> None:

        super().__init__()

        print()
        print("Vanilla RetinaNet Object Created")
        print()

        # Either load weights or not depending upon the pretrained flag specified
        # in the arguments and create the RetinaNet
        weights = "DEFAULT" if pretrained else None
        self.model = retinanet_resnet50_fpn(weights=weights, weights_backbone="DEFAULT")

        # Replace the head based on the number of classes that we have.
        self.model.head = RetinaNetHead(
            in_channels=self.model.backbone.out_channels,
            num_anchors=self.model.head.classification_head.num_anchors,
            num_classes=num_classes,
        )

        self.lr = lr
        self.batch_size = batch_size

        self.save_hyperparameters()
            

        return None

    def forward(self, x) -> Any:
        """
        This is one of the default function for a PyTorch module. Whenever
        we call model(x), basically this function gets called.

        Args: 
            images (List of Tensors [N, C, H, W]):
                List of tensors, each of shape [C, H, W], one for each image, and should be in 0-1 range.
                Different images can have different sizes.

        Returrns:
            The output of the model. Which in this case would be predictions made
            by the model.

            boxes (FloatTensor[N, 4]):
                The predicted boxes in [x1, y1, x2, y2] format,
                with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
            labels (Int64Tensor[N]):
                The predicted labels for each detection
            scores (Tensor[N]):
                The scores of each detection
        """

        # Setting the model in evaluation mode, don't understand why this
        # done automatically
        self.model.eval()

        # Passing the input through the model
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        This function is one of the hooks for the PyTorch Lightning Module.
        This is the main training step. One batch of input tensors are
        passed in by the training dataloaders and then we have to compute
        the losses on it. The lightning framework would then take that
        loss and compute the gradients and backpropagate using the 
        optimiser automatically.
        """

        # Get the images and targets from the batch loader
        images, targets = batch

        # The model takes both images and targets as in inputs in the
        # training mode and returns the a dictionary that contains
        # both the classification loss and the regression loss. 
        # We need to sum that up so that PyTorch lightning perfoms a 
        # backward on both of them
        loss_dict = self.model(images, targets)

        classification_loss = float(loss_dict['classification'])
        regression_loss = float(loss_dict['bbox_regression'])

        loss = sum(loss for loss in loss_dict.values())

        # Log all the metrics for one training step
        self.log("train/step/total_loss", loss, prog_bar=True)
        self.log("train/step/classification_loss", classification_loss)
        self.log("train/step/regression_loss", regression_loss)

        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        """
        This function is one of the hooks for the PyTorch Lightning Modules.
        This function is called once the training epoch is complete. Thus we
        can use this function to compute metrics that are interesting to us.
        """
        
        # Get list of losses from the outputs
        list_of_losses = [o["loss"] for o in outputs]

        # Get the means of the losses, we can use
        # the stack functionality of torch tensors
        # to do that

        mean_epoch_loss = float(torch.stack(list_of_losses).mean())

        self.log("train/epoch/mean_total_loss", mean_epoch_loss)

        return None

    def validation_step(self, batch, batch_idx):
        """
        This function is one of the hooks for the PyTorch Lightning Modules.
        This function is very similar to the training step except it is
        performed on the validation set passed on by the Lightning Data
        Module.
        """
        
        # Get the images and targets from the batch
        images, targets = batch

        # In evaluation mode, the model only takes in the image
        # and produces the box outputs, on which we can compute
        # our own metrics
        preds = self.model(images)
        
        # Calculate Intersection over Union for the predicted boxes
        iou = torch.stack([evaluate_iou(p, t) for p, t in zip(preds, targets)]).mean()

        return {"val_iou": iou}

    def validation_epoch_end(self, outs):
        """
        This function is one of the hooks for the PyTorch Lightning Modules.
        This function is called at the end of the validation epoch. This can
        be utilised to compute mean metrics that can guage the model performance.
        """

        # Calculate the average IoU over the validation set
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        self.log("val/epoch/avg_iou", avg_iou)
        
        return None

    def configure_optimizers(self):
        """
        This function is one of the hooks for the PyTorch Lightning Modules.
        This function is used to configure the optimisers that the Lightning
        framework will use to optimise the network. 
        """

        # This is the optimiser that we want to use
        optimiser = Adam(
            self.parameters(),
            lr=self.lr
        )

        # This is the LR scheduler we want to use
        lr_scheduler = ReduceLROnPlateau(
            optimiser,
            patience=3,
            verbose=True
        )

        # Configuration for LR schedular
        lr_scheduler_config = {
            # This is the scheduler that we want to use
            "scheduler": lr_scheduler,

            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",

            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,

            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "train/epoch/mean_total_loss",

            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True
        }

        # The dictionary to return to the Lightning Module
        to_return = {
            "optimizer": optimiser,
            "lr_scheduler": lr_scheduler_config
        }

        return to_return

