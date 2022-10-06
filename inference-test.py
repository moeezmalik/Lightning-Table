import torch
from network.models import RetinaNet
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype
from torchvision.utils import draw_bounding_boxes
import cv2 as opencv

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

def visualise_bb(path_to_ckpt, path_to_img):

    path_to_ckpt = "best-chkpnt-epoch=35.ckpt"
    path_to_img = "image_42.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RetinaNet.load_from_checkpoint(path_to_ckpt)
    model.to(device)

    image = Image.open(path_to_img).convert('RGB')
    image_tensor = pil_to_tensor(image)

    image_tensor_float = convert_image_dtype(image=image_tensor, dtype=torch.float)
    image_tensor_float.to(device=device)
    image_tensor_float = image_tensor_float.unsqueeze_(0)

    with torch.no_grad():
        out = model(image_tensor_float)[0]

    scores_tensor = out["scores"]
    boxes_tensor = out["boxes"]

    idxs = torch.where(scores_tensor > 0.75)

    selected_boxes = boxes_tensor[idxs]


    with_bb = draw_bounding_boxes(image=image_tensor, boxes=selected_boxes, colors="blue", width=5)

    show_uint8_image_tensor(uint8_image_tensor=with_bb)
