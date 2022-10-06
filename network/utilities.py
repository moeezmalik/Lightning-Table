import torch
from torchvision.ops import box_iou

def evaluate_iou(preds, targets):
    """
    Evaluate intersection over union (IOU) for target from dataset
    and output prediction from model.

    Args:
        preds (Dict):
            This is the dictionary consisting of predictions that
            the network has made.
        targets (Dict):
            This is the target dictionary that is the provided
            ground truths.

    Returns:
        mean_iou (float):
            The mean IoU of the current set of boxes passed
            to this function.
    """
    
    # If no box is detected then the IoU is 0
    if preds["boxes"].shape[0] == 0:
        return torch.tensor(0.0, device=preds["boxes"].device)
    
    return box_iou(preds["boxes"], targets["boxes"]).diag().mean()

def collate_fn(batch):
    """
    This is a custom collate function that has been taken from
    the PyTorch implementation in order to properly collate the
    data for the object detection models. Original source can
    be found here:
        Link: https://github.com/pytorch/vision/blob/main/references/detection/utils.py
    
    Args:
        batch (List[Tensor]):
            This is the current batch of images that is passed
            to the collate function
    """
    return tuple(zip(*batch))

# Also add IoU calculation and other metrics here