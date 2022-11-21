"""
This file implements the functions for visualising the
outputs of the models that are available as part of
this library.

Author:
    Name:
        Muhammad Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

# ##########################################################
# Imports
# ##########################################################

# Installed Packages
import torch
import cv2 as opencv
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype
from torchvision.utils import draw_bounding_boxes

# Custom Files
from inferencers import infer_on_pil_image
from files import load_image_to_pil, get_list_of_files_with_ext, PDFLoader

# ##########################################################
# Core Functions
# ##########################################################

def show_uint8_image_tensor(
    uint8_image_tensor: torch.Tensor,
    win_name: str = "Image Visualiser"
    ) -> None:
    """
    This function uses the OpenCV library to show the image tensor provided.
    It will also wait for an enter key press and then it will close the
    window in which the image was plotted.

    Parameters:
        uint8_image_tensor:
            A UINT8 type image tensor that is to be visualised

        win_name:
            This is the name of the window that will be shown when
            the image is displayed.
    """

    # Tensors have channels as the first dimension so we will shift them to the back
    # because OpenCV needs them as the last dimension
    channels_at_end = uint8_image_tensor.permute(1, 2, 0).numpy()

    # Show the image
    opencv.imshow(winname=win_name, mat=channels_at_end)

    # Wait for Enter Key press
    opencv.waitKey(0)

    # Then close the windows
    opencv.destroyAllWindows()

    return None

def show_pil_with_bb(
    pil_image: Image,
    bb: torch.Tensor,
    title: str = "Image",
    color: str = "blue",
    width: int = 5
    ) -> None:
    """
    This function will draw the bounding boxes on the PIL image provided
    and then use OpenCV to show the image with bounding boxes on the
    screen. To close the image shows, press enter.

    Parameters:
        pil_image:
            This is the PIL image object that contains the image to
            show.

        bb:
            These are the bounding boxes that the network had produced.

        title:
            This is the title of the window that will be shown when the
            image is displayed.

        color:
            This is the color of the bounding boxes that will be drawn.

        width:
            This is the width of the bounding boxes that will be drawn.
    """

    # Convert the image to a tensor put in appropriate memory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = pil_to_tensor(pil_image)

    # Apply appropriate transformations
    image_tensor_float = convert_image_dtype(image=image_tensor, dtype=torch.float)
    image_tensor_float.to(device=device)
    image_tensor_float = image_tensor_float.unsqueeze_(0)

    # Get the image tensor with the bounding boxes drawn
    with_bb = draw_bounding_boxes(
        image=image_tensor,
        boxes=bb,
        colors=color,
        width=width
        )

    # Show the tensor on screen
    show_uint8_image_tensor(uint8_image_tensor=with_bb, win_name=title)

def visualise_single_image(
    path_to_image: str,
    model_name: str,
    ckpt_path: str,
    conf_thresh: float,
    verbose: bool = False
    ) -> None:
    """
    This function will run a single image through the model with
    the checkpoint provided and show it on the screen.

    Parameters:
        path_to_image:
            This is the path to the image that needs to be
            visualised.
        
        model_name:
            This is the name of the model that will be used for
            inferencing on the image.

        ckpt_path:
            This is the path to the checkpoint that contains the
            trained weights for the specified model.

        conf_thresh:
            This specifies the confidence threshold for the
            predictions made by the model.

        verbose:
            If this flag is set, the name of the image being used
            will also be displayed on the command line.
    """

    # Load the image
    image, image_name = load_image_to_pil(path_to_image=path_to_image)

    # Show the name of the image on the command line if asked for

    if verbose:
        print("Evaluating: " + image_name)

    # Return the selected boxes according to the confidence threshold provided
    # and using the model provided
    bounding_boxes = infer_on_pil_image(
        pil_image=image,
        model_name=model_name,
        ckpt_path=ckpt_path,
        conf_thresh=conf_thresh
        )

    # Show the PIL image with the name of the image as the title of the image
    show_pil_with_bb(
        pil_image=image,
        bb=bounding_boxes,
        title=image_name
        )


    return None

def visualise_folder_of_images(
    path_to_folder: str,
    model_name: str,
    ckpt_path: str,
    conf_thresh: float,
    verbose: bool = False
    ) -> None:
    """
    This function will run a single image through the model with
    the checkpoint provided and show it on the screen.

    Parameters:
        path_to_folder:
            This is the path to the folder that contains all of the
            images that need to be visualised.
        
        model_name:
            This is the name of the model that will be used for
            inferencing on the images.

        ckpt_path:
            This is the path to the checkpoint that contains the
            trained weights for the specified model.

        conf_thresh:
            This specifies the confidence threshold for the
            predictions made by the model.
    """

    # Get the paths to images in the directory specified
    list_of_images = get_list_of_files_with_ext(
        path_to_folder=path_to_folder,
        ext=".jpg",
        verbose=verbose
    )

    print()

    # Visualise each image one by one
    for image in list_of_images:
        visualise_single_image(
            path_to_image=image,
            model_name=model_name,
            ckpt_path=ckpt_path,
            conf_thresh=conf_thresh,
            verbose=verbose
        )

    return None

def visualise_single_pdf(
    path_to_pdf: str,
    model_name: str,
    ckpt_path: str,
    conf_thresh: float,
    dpi: int = 600,
    verbose: bool = False
    ) -> None:
    """
    This function will run a single PDF through the model with
    the checkpoint provided and show its individual pages with
    inferred bounding boxes on the screen.

    Parameters:
        path_to_pdf:
            This is the path to the PDF that needs to be
            visualised.
        
        model_name:
            This is the name of the model that will be used for
            inferencing on the pdf.

        ckpt_path:
            This is the path to the checkpoint that contains the
            trained weights for the specified model.

        conf_thresh:
            This specifies the confidence threshold for the
            predictions made by the model.

        dpi:
            This is the DPI number that will be used to convert
            the PDF to images before perfoming the inference.

        verbose:
            If this flag is set, the name of the image being used
            will also be displayed on the command line.
    """

    pdf_doc = PDFLoader(
        path_to_pdf=path_to_pdf,
        verbose=verbose
    )

    pdf_name = pdf_doc.pdf_name

    if verbose:
        print("Evaluating: " + pdf_name)

    # Get the number of pages
    page_count = pdf_doc.page_count

    # Loop over all available pages, show them with their bounding boxes
    for i in range(page_count):
        
        pil_image = pdf_doc.get_page_in_pil(
            pg_no=i,
            dpi=dpi
        )

        bounding_boxes = infer_on_pil_image(
            pil_image=pil_image,
            model_name=model_name,
            ckpt_path=ckpt_path,
            conf_thresh=conf_thresh
        )

        show_pil_with_bb(
            pil_image=pil_image,
            bb=bounding_boxes,
            title=pdf_name + " - Page " + str(i + 1)
        )


def visualise_folder_of_pdfs(
    path_to_folder: str,
    model_name: str,
    ckpt_path: str,
    conf_thresh: float,
    dpi: int = 600,
    verbose: bool = False
    ) -> None:
    """
    This function will run a single PDF through the model with
    the checkpoint provided and show its individual pages with
    inferred bounding boxes on the screen.

    Parameters:
        path_to_folder:
            This is the path to the folder that contains all
            of the PDF files.
        
        model_name:
            This is the name of the model that will be used for
            inferencing on the pdf.

        ckpt_path:
            This is the path to the checkpoint that contains the
            trained weights for the specified model.

        conf_thresh:
            This specifies the confidence threshold for the
            predictions made by the model.

        dpi:
            This is the DPI number that will be used to convert
            the PDF to images before perfoming the inference.

        verbose:
            If this flag is set, the name of the image being used
            will also be displayed on the command line.
    """

    list_of_pdfs = get_list_of_files_with_ext(
        path_to_folder=path_to_folder,
        ext=".pdf",
        verbose=verbose
    )

    for pdf in list_of_pdfs:
        visualise_single_pdf(
            path_to_pdf=pdf,
            model_name=model_name,
            ckpt_path=ckpt_path,
            conf_thresh=conf_thresh,
            dpi=dpi,
            verbose=verbose
        )

    return None
