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
from torch.optim import SGD


# Others
from typing import Any

### - CLASSES - ###

# RetinaNet
class RetinaNet(LightningModule):
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
        print("Hello, World! from RetinaNet")

        self.lr = lr
        
        # Either load weights or not depending upon the pretrained flag specified
        # in the arguments
        weights = "DEFAULT" if pretrained else None
        self.model = retinanet_resnet50_fpn(weights=weights, weights_backbone="DEFAULT")

        self.model.head = RetinaNetHead(
            in_channels=self.model.backbone.out_channels,
            num_anchors=self.model.head.classification_head.num_anchors,
            num_classes=num_classes,
        )
        


        # self.model = fasterrcnn_resnet50_fpn(
        #     pretrained=pretrained,
        #     pretrained_backbone=True,
        #     trainable_backbone_layers=3,
        # )

        # in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)





        self.batch_size = batch_size

        self.save_hyperparameters()
            

        return None

    def forward(self, x) -> Any:

        # Setting the model in evaluation mode, don't understand why this
        # done automatically
        self.model.eval()

        # Passing the input through the model
        return self.model(x)

    def training_step(self, batch, batch_idx):

        # Get the images and targets from the batch loader
        images, targets = batch

        # The batchloader returns the targets as a list of tuples,
        # the following operation coverts those tuples to a list
        # targets = [{k: v for k, v in t.items()} for t in targets]

        # The model takes both images and targets as in inputs in the
        # training mode and returns the a dictionary that contains
        # both the classification loss and the regression loss. 
        # We need to sum that up so that PyTorch lightning perfoms a 
        # backward on both of them
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        
        # Get the images and targets from the batch
        images, targets = batch

        # In evaluation mode, the model only takes in the image
        # and produces the box outputs, on which we can compute
        # our own metrics
        preds = self.model(images)
        
        # Calculate Intersection over Union for the predicted boxes
        iou = torch.stack([self._evaluate_iou(p, t) for p, t in zip(preds, targets)]).mean()

        self.log("val/iou", iou, prog_bar=True)

        return {"val_iou": iou}

    def validation_epoch_end(self, outs):

        # Calculate the average IoU over the validation set
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        self.log("val/avg_iou", avg_iou)
        
        return None

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr
        )

    def _evaluate_iou(self, preds, targets):
        """
        Evaluate intersection over union (IOU) for target from dataset and output prediction from model.
        """
        # no box detected, 0 IOU
        if preds["boxes"].shape[0] == 0:
            return torch.tensor(0.0, device=preds["boxes"].device)
        
        return box_iou(preds["boxes"], targets["boxes"]).diag().mean()

