import cv2 as opencv
import torch
import torchvision.transforms.functional as TF

def float_to_uint8_image_tensor(float_image_tensor: torch.Tensor) -> torch.Tensor:
    """
    This function converts the tensors that have been scaled in float
    type to uint8 and scales the values accordingly. This function
    was written original to convert the image tensors from float to
    uint so that they can be visualised using libraries such as OpenCV

    Args:
        tensor (torch.Tensor):
            An image tensor with the data type in float to be converted
            into uint8 type

    Returns:
        tensor (torch.Tensor):
            An image tensor that has been converted to uint8 from float
            in the form of a torch tensor
    """

    return TF.convert_image_dtype(image=float_image_tensor, dtype=torch.uint8)

def show_uint8_image_tensor(uint8_image_tensor: torch.Tensor) -> None:
    """
    This function uses the OpenCV library to show the image tensor provided.
    It will also wait for an enter key press and then it will close the
    window in which the image was plotted.

    Args:
        uint8_image_tensor (torch.Tensor):
            A UINT8 type image tensor that is to be visualised
    """

    # Tensors have channels at the start so we will shift them to the back
    channels_at_end = uint8_image_tensor.permute(1, 2, 0).numpy()

    # Show the image
    opencv.imshow(winname="Image Visualiser", mat=channels_at_end)

    # Wait for eenter key press
    opencv.waitKey(0)

    # Then close the windows
    opencv.destroyAllWindows()

    return None