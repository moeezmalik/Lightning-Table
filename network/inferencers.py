"""
This file implements the functions to do inference using the
models that are available as part of the models.py file.

Author:
    Name:
        Muhammad Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

# #################################################################
# All of the imports
# #################################################################

# All the utilities
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype
from pytorch_lightning import LightningModule
import cv2 as opencv

# Imports from custom scripts
from models import get_model
from files import PDFLoader, get_list_of_files_with_ext

# #################################################################
# The following functions are the main building blocks of inference
# #################################################################


def get_model_with_ckpt(
    model_name: str,
    ckpt_path: str
    ) -> LightningModule:
    """
    This function will get the model from the models.py file and
    then load the trained weights from the checkpoint that was
    provided. The returned entity will be the trained model.
    Care needs to be taken to make sure the checkpoint belongs to
    the model that is being requested.
    """

    # Get the model without any weights loaded from the models file
    vanilla_model = get_model(model_name=model_name)

    # Check if the correct model name was specified
    if vanilla_model is None:
        print("Model with the name: " + model_name + "not found.")
        return None
    
    # Load the model with the checkpoint weights
    model = vanilla_model.load_from_checkpoint(ckpt_path)

    # Move the model to appropriate storage i.e. RAM if only CPU is
    # available and VRAM if only GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model

def infer_on_pil_image(
    pil_image: Image,
    model_name: str,
    ckpt_path: str,
    conf_thresh: float
    ) -> torch.Tensor:
    """
    This function will perform the inference on the provided image
    and return the resulting bounding boxes in a tensor format.

    Parameters:
        pil_image (Image):
            This is the image in PIL format that needs to be inferred
            on.
        
        model (LightningModule):
            This is the name of the model that will be used for inferencing
            on the pil image.

        ckpt_path (str):
            This is the path to the checkpoint that contains the trained
            weights for the model.

        conf_thresh (float):
            This is the confidence threshold above which the bounding
            boxes will be kept, the rest will be ignored.

    Returns:
        selected_boxes (torch.Tensor):
            These are the bounding boxes that were detected on this image.
    """

    # Convert the PIL image to a tensor and put it in appropriate memory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the model and load the weights from the checkpoint provided
    model = get_model_with_ckpt(model_name=model_name, ckpt_path=ckpt_path)
    
    # Convert the PIL image to an image tensor
    image_tensor = pil_to_tensor(pil_image)

    # Apply transformations on the image tensor so that it can be used to
    # infer with the help of the model
    image_tensor_float = convert_image_dtype(image=image_tensor, dtype=torch.float)
    image_tensor_float.to(device=device)
    image_tensor_float = image_tensor_float.unsqueeze_(0)

    # Perform the inferences and get all the bounding boxes
    with torch.no_grad():
        out = model(image_tensor_float)[0]

    # Get the boxes and their respective scores
    boxes_tensor = out["boxes"]
    scores_tensor = out["scores"]

    # Get the boxes that have the confidence above the provided threshold
    idxs = torch.where(scores_tensor > conf_thresh)
    selected_boxes = boxes_tensor[idxs]

    return selected_boxes

def pil_to_list_of_bb(
    pil_image: Image,
    model_name: str,
    ckpt_path: str,
    conf_thresh: float
    ) -> list:
    """
    This function will perform the inference on the provided image
    using the provided model and return the resulting bounding boxes
    in a list. The format of the list is shown in the returns section.

    Parameters:
        pil_image:
            This is the image in PIL format that needs to be inferred
            on.
        
        model:
            This is the name of the model that will be used for inferencing
            on the pil image.

        ckpt_path:
            This is the path to the checkpoint that contains the trained
            weights for the model.

        conf_thresh:
            This is the confidence threshold above which the bounding
            boxes will be kept, the rest will be ignored.

    Returns:
        list_of_boxes:
            The list of bounding boxes detected on the current image.
    """

    # Get the bounding boxes in the tensor format
    bb = infer_on_pil_image(
        pil_image=pil_image,
        model_name=model_name,
        ckpt_path=ckpt_path,
        conf_thresh=conf_thresh
    )

    # Conver the tensor of bounding boxes to a list
    # return it
    return bb.tolist()


def pdf_to_list_of_bb(
    path_to_pdf: str,
    model_name: str,
    ckpt_path: str,
    conf_thresh: str,
    dpi: int = 600,
    verbose: bool = True
    ) -> list:
    """
    This function will use the model provided to infer on the the PDFs file
    that was provided.

    Parameters:
        path_to_pdf:
            This is the path to PDF that on which inference will be performed
            and the detected bounding boxes will be saved in a CSV file.

        model_name:
            This is the name of the model that will be used for performing the
            inference.

        ckpt_path:
            This is the path to the checkpoint that will be used for loading
            weights into the model specified.

        conf_thresh:
            This specifies the confidence threshold for the bounding boxes
            detected by the model.

        dpi:
            This is the DPI that will be used to render the individual pages
            of PDF to images.

        verbose:
            If set to True, this will produce output on the command line tool
            while the tool is performing operations.
    """

    # Load the specified PDF file
    pdf_file = PDFLoader(
        path_to_pdf=path_to_pdf,
        verbose=verbose
    )

    # Initialise the list that will contain the bounding
    # with respect to the page numbers
    curr_pdf_bb_with_pgno = []

    if not pdf_file.is_text_based:

        print("Skipping: PDF is not text-based")
        return curr_pdf_bb_with_pgno

    page_count = pdf_file.page_count

    # Loop over all the pages and get the bounding boxes
    for pg_no in range(page_count):

        # Render the current page to a PIL object
        scale_factor, pil_image = pdf_file.get_page_in_pil(
            pg_no=pg_no,
            dpi=dpi
            )

        # Infer on the rendered image to get
        # the list of bounding boxes
        curr_pg_bbs = pil_to_list_of_bb(
            pil_image=pil_image,
            model_name=model_name,
            ckpt_path=ckpt_path,
            conf_thresh=conf_thresh
        )

        # Skip the page if no table was found
        if len(curr_pg_bbs) == 0:
            continue
        
        # Transform the boxes to the PDF coordinate space
        curr_pg_tranformed_bb = pdf_file.transform_bb_to_pdf_space(
                bbs=curr_pg_bbs,
                pg_no=pg_no,
                scale_factor=scale_factor
                )

        # Add the page number with each of the bounding box
        # Also add the name of the PDF file
        for box in curr_pg_tranformed_bb:

            curr_pdf_bb_with_pgno.append(
                    [
                        pdf_file.pdf_name,
                        pg_no+1,
                        int(box[0]),
                        int(box[1]),
                        int(box[2]),
                        int(box[3])
                    ]
                )


    return curr_pdf_bb_with_pgno



# #################################################################
# The following functions serve the purpose of the command line
# application
# #################################################################


def pdf_to_csv(
    path_to_pdf: str,
    model_name: str,
    ckpt_path: str,
    conf_thresh: str,
    output_path: str,
    dpi: int = 600,
    verbose: bool = True
    ) -> None:
    """
    This function will use the model provided to infer on the the PDFs file
    that was provided.

    Parameters:
        path_to_pdf:
            This is the path to PDF that on which inference will be performed
            and the detected bounding boxes will be saved in a CSV file.

        model_name:
            This is the name of the model that will be used for performing the
            inference.

        ckpt_path:
            This is the path to the checkpoint that will be used for loading
            weights into the model specified.

        conf_thresh:
            This specifies the confidence threshold for the bounding boxes
            detected by the model.

        output_path:
            This is the path where the output CSV will be saved.

        dpi:
            This is the DPI that will be used to render the individual pages
            of PDF to images.

        verbose:
            If set to True, this will produce output on the command line tool
            while the tool is performing operations.
    """

    # Get the list of bounding boxes on the current PDF
    list_of_bb = pdf_to_list_of_bb(
        path_to_pdf=path_to_pdf,
        model_name=model_name,
        ckpt_path=ckpt_path,
        conf_thresh=conf_thresh,
        dpi=dpi,
        verbose=verbose
    )

    # Convert the list to a data frame
    df = pd.DataFrame(
        list_of_bb,
        columns=[
            'filename',
            'pageno',
            'x1',
            'y1',
            'x2',
            'y2'
        ]
    )

    # Save the DataFrame to a CSV file
    df.to_csv(
        path_or_buf=output_path,
        header=True,
        index=False
    )

    # Show the message the CSV has been saved
    print()
    print("Saved CSV file to: " + output_path)

    return None

def folder_of_pdf_to_csv(
    path_to_folder: str,
    model_name: str,
    ckpt_path: str,
    conf_thresh: str,
    output_path: str,
    dpi: int = 600,
    verbose: bool = True
    ) -> None:
    """
    This function will use the model provided to infer on the the PDFs file
    in the folder that was entereds. It does not expect a special folder structure,
    just a folder with all the PDF files.

    Parameters:
        path_to_folder:
            This is the path to the directory/folder that contains all of the
            PDF files what need to be evaluated.

        model_name:
            This is the name of the model that will be used for performing the
            inference.

        ckpt_path:
            This is the path to the checkpoint that will be used for loading
            weights into the model specified.

        conf_thresh:
            This specifies the confidence threshold for the bounding boxes
            detected by the model.

        output_path:
            This is the path where the output CSV will be saved.

        dpi:
            This is the DPI that will be used to render the individual pages
            of PDF to images.

        verbose:
            If set to True, this will produce output on the command line tool
            while the tool is performing operations.
    """

    # Get a list of all the PDF files
    list_of_pdfs = get_list_of_files_with_ext(
        path_to_folder=path_to_folder,
        ext='.pdf',
        verbose=verbose
    )

    # Create a template for the list of bouding boxes
    all_bbs = []

    # Set the count for the PDF files
    count = 1
    total = len(list_of_pdfs)
    
    # Go over the PDFs one by one
    for pdf in list_of_pdfs:

        curr_pdf_bbs = pdf_to_list_of_bb(
            path_to_pdf=pdf,
            model_name=model_name,
            ckpt_path=ckpt_path,
            conf_thresh=conf_thresh,
            dpi=dpi,
            verbose=verbose
        )

        # Update the bounding boxes list
        all_bbs.extend(curr_pdf_bbs)

        # Print the progress update
        print("---")
        print("Progress: {}/{} PDFs Evaluated".format(count, total))
        print("---")
        count += 1

    
    # Convert the list to a data frame
    df = pd.DataFrame(
        all_bbs,
        columns=[
            'filename',
            'pageno',
            'x1',
            'y1',
            'x2',
            'y2'
        ]
    )

    # Save the DataFrame to a CSV file
    df.to_csv(
        path_or_buf=output_path,
        header=True,
        index=False
    )

    # Show the message the CSV has been saved
    print()

    print("---")
    print("Saved CSV file to: " + output_path)
    print("---")

    



    

