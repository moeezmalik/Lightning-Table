### - IMPORTS - ###

# Pytorch and Lightning
import torch
from torchvision.ops import box_iou
from torchmetrics import Metric

# Other
import numpy as np

# Misc
from typing import Any, Dict, List, Optional



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


class OneClassPrecisionRecall(Metric):
    """
    A custom evaluation metric class that inherits from the
    TorchMetrics' Metrics class by PyTorch Lightning Framework.
    This class only computes the precision and recall metrics
    for one class which is the goal in the case of Table
    Detection.

    Args:
        score_threshold (float):
            When the network makes predictions, it also includes
            the scores of those predictions. These scores are an
            indication of how confident the network is in those
            predictions. In any image there will be detections
            where the network makes some predictions that are not
            correct but they will also have very low confidence. 
            With this threshold we can cut off those low confidence
            detections. This will get a better measure because
            we will also neglect low confidence results during
            production.

        iou_threshold (float):
            iou stand for Intersection of Union. This is a measure
            when we trying to evaluate the overlap of bounding boxes.
            Higher IoU value means that the boxes are very similar.
            This parameter can be adjusted in order to evaluate precision
            and recall at different level of IoU.

    Note: 
        This implementation is for a very specific use case of where
        there is only one object of interest that needs to be detected.
        This metric cannot be used for multiclass precision and recall
        metric calculations.
    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = True

    true_positives: torch.Tensor(0)
    false_positives: torch.Tensor(0)
    num_annotations: torch.Tensor(0)

    def __init__(
        self,
        score_threshold: float = 0.75,
        iou_threshold: float = 0.9,
        **kwargs: Any
        ) -> None:
        
        
        super().__init__(**kwargs)

        # Set up the parameters that will be used in the calculations
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        # We will add the states here that we need to cummulate across evaluation steps
        # At the end of the evaluation epoch we can use these states to calculate
        # the precision and recall metrics.

        self.add_state(
            name="true_positives",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )

        self.add_state(
            name="false_positives",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )

        self.add_state(
            name="num_annotations",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )


    def update(self, preds: List[Dict], targets: List[Dict]):
        """
        This function updates the state variables of the metric by
        taking in the predictions and targets at the current
        evaluation step.

        Args:
            preds (List[Dict]):
                This is the list of dictionaries that the network
                generates. The dictionaries are assumed to be in the
                format of PyTorch detection networks. Each dictionary
                represents one image.
            targets (List[Dict]):
                These are the ground truth values for each of the
                prediction images.

        Note:
            An assertion error is thrown if the length of the predictions
            and the targets is not the same.
        """

        assert len(preds) == len(targets)

        false_positives = 0
        true_positives = 0
        num_annotations = 0

        # Iterate over all the images in the batch
        for i in range(len(preds)):

            # Get all the predicted and ground truth boxes
            # on the current image
            annotations = targets[i]["boxes"].cpu().numpy()
            num_annotations += annotations.shape[0]
            detected_annotations = []

            # Get the boxes with scores higher than specified
            # threshold
            scores_tensor = preds[i]["scores"]
            boxes_tensor = preds[i]["boxes"]

            idxs = torch.where(scores_tensor > self.score_threshold)
            selected_boxes = boxes_tensor[idxs]
            detections = selected_boxes.cpu().numpy()


            # Start evaluation
            for d in detections:
                
                if annotations.shape[0] == 0:
                    false_positives += 1
                    true_positives  += 0
                    continue

                overlaps = self._compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= self.iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives += 0
                    true_positives  += 1
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives += 1
                    true_positives  += 0


        # Save the local state to the global states
        self.false_positives += false_positives
        self.true_positives += true_positives
        self.num_annotations += num_annotations

        # print("False Positives: " + str(false_positives))
        # print("True Positives: " + str(true_positives))
        # print("Num Annotations: " + str(num_annotations))
        # print()

        # recall = true_positives / np.maximum(num_annotations, np.finfo(np.float64).eps)
        # precision = true_positives / np.maximum((true_positives + false_positives), np.finfo(np.float64).eps)

        # print("Recall: " + str(recall))
        # print("Precision: " + str(precision))


    def compute(self) -> Dict:
        """
        This function will eventually computer the precision
        and recall for the current epoch at the end of the
        evaluation or training steps.

        Returns:
            result (Dict):
                This return type of this function is a dictionary
                that contains the values for precision and recall
                that is computed when this function is called.
        """


        # Load up the state variables to perform calculations
        result = {}
        true_positives = float(self.true_positives)
        false_positives = float(self.false_positives)
        num_annotations = float(self.num_annotations)

        # Computer Recall and Precision
        recall = true_positives / np.maximum(num_annotations, np.finfo(np.float64).eps)
        precision = true_positives / np.maximum((true_positives + false_positives), np.finfo(np.float64).eps)

        # Save the result in a dictionary
        result["recall"] = recall
        result["precision"] = precision

        # Retun the result dictionary
        return result


    def _compute_overlap(self, a, b):
        """
        This function computes the overlap of the set of boxes
        provided.

        Args:
            a (N, 4):
                First set of boxes: ndarray of float
            b (K, 4):
                Second set of boxes: ndarray of float

        Returns:
            overlaps (N, K):
                ndarray of overlap between boxes and query_boxes
        """
        
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
        ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

        ua = np.maximum(ua, np.finfo(float).eps)

        intersection = iw * ih

        return intersection / ua