"""
Name:
    PDF to Image

Description:
    This file contains the functions necessary to convert the PDF
    files to images. This conversion is necessary because the
    object detectors implemented using PyTorch can only read and
    process image files. This would be a preprocess before the PDF
    files are fed into the network.

Author:
    Name:
        M. Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

import fitz
import argparse
from os import listdir
from os.path import join
from pathlib import Path

from typing import Sequence


def convert_pdf_to_images(
    pdf_folder: str,    
    pdf_file_name: str,
    image_save_folder: str,
    dpi: int
) -> None:
    """
    This function will convert a single PDF file to set of its images. It will
    take the name of the PDF file, create a folder with the same name. Within
    that folder, it will put in the converted images, one for each page.

    Parameters:
        pdf_folder (str):
            This is folder that contains the PDF file.

        pdf_file_name (str):
            This is the name of the PDF file that needs to be converted. It is
            assumed that this filename includes the extention '.pdf'

        image_save_folder (str):
            This is the path to the folder in which the images should be saved.
            This function will create a new folder inside this specified folder
            with the same name as the PDF file and put the images inside.

        dpi (int):
            This is the DPI for the conversion of PDF to images.
    """

    # Get the name of the PDF file without the extention '.pdf'
    file_name_without_ext = pdf_file_name.rsplit('.', 1)[0]

    # Make a folder using the name extracted above
    path_to_pdf_images = join(image_save_folder, file_name_without_ext)
    Path(path_to_pdf_images).mkdir(parents=True, exist_ok=True)

    # Open the current PDF file using the library
    path_to_pdf_file = join(pdf_folder, pdf_file_name)
    doc = fitz.open(path_to_pdf_file)

    # Get the number of pages
    page_count = doc.page_count

    # Loop over all available pages and convert them to images
    for i in range(page_count):
        
        # Load the page
        page = doc.load_page(i)

        # Convert to an image
        pix = page.get_pixmap(dpi=dpi)

        # Set a unique name for the current page
        image_name = str(i + 1) + ".jpg"
        output_name = join(path_to_pdf_images, image_name)

        # Save the page as image
        pix.save(output_name)


    return None

class PDFToImage():
    """
    This class provides the main structure to the code that will be reponsible
    for generating images from PDFs. 

    Parameters:
        pdf_folder (str):
            This is the path to the folder that contains all of the PDF files.

        images_folder (str):
            The folder where to save the images.

        dpi (int):
            This is the DPI setting that will be used when converting the PDF
            files to images. Larger DPI setting will mean larger image sizes.
    """

    def __init__(
            self,
            pdf_folder: str,
            images_folder: str,
            dpi: int = 600
        ) -> None:
        """
        This is the main constructor function.

        Parameters:
            See the parameters of the class for more information on what
            parameters can be provided.
        """

        print()
        print("PDF to Image Utility")

        # Assign parameters
        self.pdf_folder = pdf_folder
        self.images_folder = images_folder
        self.dpi = dpi

        # Declare class variables
        self.list_pdf_files = []

        # Do an initial survey of the folder and get
        # the list of PDF files.
        self.checkout_pdf_folder()

    def checkout_pdf_folder(self) -> list:
        """
        This function will explore the PDF folder provided and get the list
        of PDF files in the provided folder.
        """

        # Go through the folder
        # get the files ending with extension
        # PDF and put them in the list

        for file in listdir(self.pdf_folder):
            if file.endswith(".pdf"):
                self.list_pdf_files.append(file)
        
        print()
        print("Number of PDF files found: " + str(len(self.list_pdf_files)))

    def convert(self) -> None:
        """
        This function will go through the list of PDF files one by one
        and convert them to images that are required using the specified
        DPI setting.
        """

        # Go through all the files in the list and convert them

        for file in self.list_pdf_files:
            convert_pdf_to_images(
                pdf_folder=self.pdf_folder,
                pdf_file_name=file,
                image_save_folder=self.images_folder,
                dpi=self.dpi
        )

        return None
        

def main(args: Sequence = None) -> None:

    # Setup the argument parser to get the required parameters

    parser = argparse.ArgumentParser(description='Utility script for converting PDFs to images.')

    parser.add_argument('--pdf_folder', help='Path to the folder with PDF files.', type=str, default=None)
    parser.add_argument('--images_folder', help='Path to the folder where you want to save the images.', type=str, default=None)
    parser.add_argument('--dpi', help='Path to the folder where you want to save the images.', type=int, default=600)

    parser = parser.parse_args(args)

    pdf_folder = parser.pdf_folder
    images_folder = parser.images_folder
    dpi = parser.dpi

    # Error check the non-optional parameters
    if(pdf_folder is None):
        print("Error: No PascalVOC path specified, cannot continue")
        return None
    
    if(images_folder is None):
        print("Error: No images folder specified, cannot continue")
        return None


    pdf_to_image = PDFToImage(
        pdf_folder=pdf_folder,
        images_folder=images_folder,
        dpi=600
        )

    pdf_to_image.convert()

    return None

if __name__ == '__main__':
    main()