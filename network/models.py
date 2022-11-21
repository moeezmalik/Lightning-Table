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

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, FastRCNNPredictor
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2, RetinaNetHead, RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from torchvision.models import ResNet18_Weights

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Others
from utilities import evaluate_iou, OneClassPrecisionRecall

# Misc
from typing import Any

# ##########################################################
# Classes that implement the models
# ##########################################################

# The main SuperNet from which the other models with inherit

class SuperNet(LightningModule):
    """
    This class serves to gather together the most common pieces in all of the
    detection networks that we are going to use. This is done so that we dont
    have to write the same code again and again. All the actual networks will
    then derive the methods from this class and modify them. This class actually
    itself inherits from the PyTorch Lightning Module.

    The code is heavily borrowed from the PyTorch Lightning Bolt Implementations
    which can be found here:
        https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/detection/retinanet/retinanet_module.py

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
        pretrained: bool = True,
        batch_size: int = 2,
        verbose: bool = False
    ) -> None:

        super().__init__()

        self.verbose = verbose

        if self.verbose:
            print()
            print("SuperNet Object Created")

        self.lr = lr
        self.batch_size = batch_size

        self.pr_metric_75_50 = OneClassPrecisionRecall(score_threshold=0.75, iou_threshold=0.5)
        self.pr_metric_75_75 = OneClassPrecisionRecall(score_threshold=0.75, iou_threshold=0.75)
        self.pr_metric_75_90 = OneClassPrecisionRecall(score_threshold=0.75, iou_threshold=0.9)

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

        self.pr_metric_75_50.update(preds=preds, targets=targets)
        self.pr_metric_75_75.update(preds=preds, targets=targets)
        self.pr_metric_75_90.update(preds=preds, targets=targets)

        return {"val_iou": iou}

    def validation_epoch_end(self, outs):
        """
        This function is one of the hooks for the PyTorch Lightning Modules.
        This function is called at the end of the validation epoch. This can
        be utilised to compute mean metrics that can guage the model performance.
        """

        # Calculate the average IoU over the validation set
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()

        precision_75_50 = self.pr_metric_75_50.compute()["precision"]
        precision_75_75 = self.pr_metric_75_75.compute()["precision"]
        precision_75_90 = self.pr_metric_75_90.compute()["precision"]

        recall_75_50 = self.pr_metric_75_50.compute()["recall"]
        recall_75_75 = self.pr_metric_75_75.compute()["recall"]
        recall_75_90 = self.pr_metric_75_90.compute()["recall"]

        # Log Everything
        self.log("val/epoch/avg_iou", avg_iou)

        self.log("val/epoch/precision_75_50", precision_75_50)
        self.log("val/epoch/precision_75_75", precision_75_75)
        self.log("val/epoch/precision_75_90", precision_75_90)
        
        self.log("val/epoch/recall_75_50", recall_75_50)
        self.log("val/epoch/recall_75_75", recall_75_75)
        self.log("val/epoch/recall_75_90", recall_75_90)

        # Reset the metrics
        self.pr_metric_75_50.reset()
        self.pr_metric_75_75.reset()
        self.pr_metric_75_90.reset()
        
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

# RetinaNets

class VanillaRetinaNet(SuperNet):
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
        pretrained: bool = True,
        batch_size: int = 2,
        verbose: bool = False
    ) -> None:

        super().__init__(lr=lr, num_classes=num_classes, pretrained=True, batch_size=batch_size, verbose=verbose)

        if self.verbose:
            print("Vanilla RetinaNet Object Created")

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

        return None

class VanillaRetinaNetV2(SuperNet):
    """
    This class implements the RetinaNet V2 using PyTorch and the higher
    level wrapper PyTorch Lighting modules. The implementation is one
    that is directly available from PyTorch.

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
        pretrained: bool = True,
        batch_size: int = 2,
        verbose: bool = False
    ) -> None:

        super().__init__(lr=lr, num_classes=num_classes, pretrained=True, batch_size=batch_size, verbose=verbose)

        if self.verbose:
            print("Vanilla RetinaNet V2 Object Created")

        # Either load weights or not depending upon the pretrained flag specified
        # in the arguments and create the RetinaNet
        weights = "DEFAULT" if pretrained else None
        self.model = retinanet_resnet50_fpn_v2(weights=weights, weights_backbone="DEFAULT")

        # Replace the head based on the number of classes that we have.
        self.model.head = RetinaNetHead(
            in_channels=self.model.backbone.out_channels,
            num_anchors=self.model.head.classification_head.num_anchors,
            num_classes=num_classes,
        )

        return None

class RetinaNetResnet18FPN(SuperNet):
    """
    This class implements the RetinaNet with a much smaller ResNet18 FPN
    Backbone with 3 trainable layers.

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
        pretrained: bool = True,
        batch_size: int = 2,
        verbose: bool = False
    ) -> None:

        super().__init__(lr=lr, num_classes=num_classes, pretrained=True, batch_size=batch_size, verbose=verbose)

        if self.verbose:
            print("RetinaNet ResNet 18 FPN Object Created")

        # Create the backbone that will extract the features
        backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=ResNet18_Weights.DEFAULT, trainable_layers=5)

        # Either load weights or not depending upon the pretrained flag specified
        # in the arguments and create the RetinaNet
        self.model = RetinaNet(backbone=backbone, num_classes=num_classes)

        return None

# Faster RCNNs

class VanillaFasterRCNN(SuperNet):
    """
    This class implements the FasterRCNN network that is available
    directly from PyTorch. The link to the model builder is:
    https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn

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
        pretrained: bool = True,
        batch_size: int = 2,
        verbose: bool = False
    ) -> None:

        super().__init__(lr=lr, num_classes=num_classes, pretrained=True, batch_size=batch_size, verbose=verbose)

        if self.verbose:
            print("Vanilla FasterRCNN Object Created")

        # Either load weights or not depending upon the pretrained flag specified
        # in the arguments and create the RetinaNet
        weights = "DEFAULT" if pretrained else None
        self.model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone="DEFAULT", num_classes=91)

        # Replace the head for custom classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)

        return None

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

        classification_loss = float(loss_dict['loss_classifier'])
        regression_loss = float(loss_dict['loss_box_reg'])

        loss = sum(loss for loss in loss_dict.values())

        # Log all the metrics for one training step
        self.log("train/step/total_loss", loss, prog_bar=True)
        self.log("train/step/classification_loss", classification_loss)
        self.log("train/step/regression_loss", regression_loss)

        return {"loss": loss}

class VanillaFasterRCNNV2(SuperNet):
    """
    This class implements the FasterRCNN network that is available
    directly from PyTorch. The link to the model builder is:
    https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2

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
        pretrained: bool = True,
        batch_size: int = 2,
        verbose: bool = False
    ) -> None:

        super().__init__(lr=lr, num_classes=num_classes, pretrained=True, batch_size=batch_size, verbose=verbose)

        if self.verbose:
            print("Vanilla FasterRCNN V2 Object Created")

        # Either load weights or not depending upon the pretrained flag specified
        # in the arguments and create the RetinaNet
        weights = "DEFAULT" if pretrained else None
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone="DEFAULT", num_classes=91)

        # Replace the head for custom classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)

        return None

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

        classification_loss = float(loss_dict['loss_classifier'])
        regression_loss = float(loss_dict['loss_box_reg'])

        loss = sum(loss for loss in loss_dict.values())

        # Log all the metrics for one training step
        self.log("train/step/total_loss", loss, prog_bar=True)
        self.log("train/step/classification_loss", classification_loss)
        self.log("train/step/regression_loss", regression_loss)

        return {"loss": loss}


# ##########################################################
# Functions
# ##########################################################

def get_model(model_name: str) -> SuperNet:
    """
    In other scripts that are part of this repository and this
    network as a whole, there are crucial functions that will
    be using these models. For examples the trainers.py will
    use these models to train them on the provided dataset.
    For that purpose we need to return the appropriate class
    based on the name that is provided, that is what this function
    does on the most basic level.

    Parameters:
        model_name (str):
            This is the model name according to which the appropriate
            model will be returned, the names that are used are exactly
            the same as the name of the classes that implement these
            models.

    Returns:
        model (SuperNet):
            The function will return the appropriate model class
            according to the name provided. In case of a mismatch, the
            function will just a return that can be used for error
            checking purposes.
    """
        
    if model_name == "VanillaRetinaNet":
        return VanillaRetinaNet

    elif model_name == "VanillaRetinaNetV2":
        return VanillaRetinaNetV2

    elif model_name == "RetinaNetResnet18FPN":
        return RetinaNetResnet18FPN

    elif model_name == "VanillaFasterRCNN":
        return VanillaFasterRCNN

    elif model_name == "VanillaFasterRCNNV2":
        return VanillaFasterRCNNV2
    
    else:
        return None