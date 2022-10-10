from types import MethodDescriptorType
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype
from torchvision.utils import draw_bounding_boxes
from pytorch_lightning import LightningModule
from PIL import Image
import torch
import cv2 as opencv
from utilities import OneClassPrecisionRecall

from pprint import pprint

from models import VanillaRetinaNet

from typing import List, Dict

ckpt_path = "misc/best-chkpnt-epoch=35.ckpt"
img_path = "misc/image_42.jpg"

def show_uint8_image_tensor(uint8_image_tensor: torch.Tensor) -> None:
    """
    This function uses the OpenCV library to show the image tensor provided.
    It will also wait for an enter key press and then it will close the
    window in which the image was plotted.

    Args:
        uint8_image_tensor (torch.Tensor):
            A UINT8 type image tensor that is to be visualised
    """

    # Tensors have channels as the first dimension so we will shift them to the back
    # because OpenCV needs them as the last dimension
    channels_at_end = uint8_image_tensor.permute(1, 2, 0).numpy()

    # Show the image
    opencv.imshow(winname="Image Visualiser", mat=channels_at_end)

    # Wait for eenter key press
    opencv.waitKey(0)

    # Then close the windows
    opencv.destroyAllWindows()

    return None

def get_detections(
    Network: LightningModule,
    image: Image,
    ckpt_path: str = None
) -> List[Dict]:
    """
    The purpose of this function is to get the detections on a particular
    image using the network and its respective checkpoint provided.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Network.load_from_checkpoint(ckpt_path)
    model.to(device)

    image_tensor = pil_to_tensor(image)

    image_tensor_float = convert_image_dtype(image=image_tensor, dtype=torch.float)
    image_tensor_float.to(device=device)
    image_tensor_float = image_tensor_float.unsqueeze_(0)

    with torch.no_grad():
        out = model(image_tensor_float)

    return out


# Open the image
image = Image.open(img_path).convert('RGB')


# Get the detections
pred_dict = get_detections(VanillaRetinaNet, image=image, ckpt_path=ckpt_path)[0]

preds = [pred_dict, pred_dict]

# Mean Average Precision Testinig

gt_boxes = torch.tensor(
    [
        [147, 195, 982, 478],
        [150, 595, 992, 871],
        [147, 968, 982, 1141],
        [144, 1144, 982, 1384],
        [992, 208, 1593, 585],
        [998, 598, 1593, 1007],
        [995, 1014, 1583, 1212],
        [989, 1280, 1583, 1430]
    ]
)

gt_labels = torch.tensor(
    [1, 1, 1, 1, 1, 1, 1, 1]
)

gt_dict = {}

gt_dict['boxes'] = gt_boxes
gt_dict['labels'] = gt_labels

targets = [gt_dict, gt_dict]


metric = OneClassPrecisionRecall(score_threshold=0.05, iou_threshold=0.9)

metric.update(preds, targets)
metric.update(preds, targets)

pprint(metric.compute())










# Visualise the detections

# Get the image tensor
# image_tensor = pil_to_tensor(image)

# # Get the boxes with scores higher than 0.75
# scores_tensor = out["scores"]
# boxes_tensor = out["boxes"]

# idxs = torch.where(scores_tensor > 0.75)

# selected_boxes = boxes_tensor[idxs]

# print(selected_boxes)


# Draw bounding boxes and show the image

# with_bb = draw_bounding_boxes(image=image_tensor, boxes=selected_boxes, colors="blue", width=5)
# with_bb = draw_bounding_boxes(image=with_bb, boxes=gt_boxes, colors="red", width=5)
# show_uint8_image_tensor(uint8_image_tensor=with_bb)
