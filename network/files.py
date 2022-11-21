"""
This file will contain methods for dealing with
everything related to dealings with files in
the file system.

Author:
    Name:
        Muhammad Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

# ##########################################################
# Imports
# ##########################################################

import fitz
from os import listdir
from os.path import join, split
from PIL import Image

# ##########################################################
# Functions
# ##########################################################

def get_list_of_files_with_ext(
    path_to_folder: str,
    ext: str,
    verbose: bool = True
    ) -> list:
    """
    This function will go through all the files in the given
    folder and make a list of files with the provided extension.
    This can be used, for example, to filter out thre required
    files in the folder.

    Parameters:
        path_to_folder:
            This is the path to folder that will be scanned
            for the required files.

        ext:
            This is the extension of the files that will be
            selected from the folder

        verbose:
            If this flag is set to True, then this function will
            display the information from the folder.

    Returns:
        list_of_files:
            This is the list of files in the provided
            directory (folder) that matches the extension
            provided. It contains the full path to the files
            not just the name of the files.
    """

    list_of_files = []

    for file in listdir(path_to_folder):
        if file.endswith(ext):
            full_path = join(path_to_folder, file)
            list_of_files.append(full_path)

    if verbose:
        print()
        print("Looking for " + ext + " files in folder: " + path_to_folder)
        print()
        print("Total " + ext + " files found: " + str(len(list_of_files)))

    return list_of_files

def load_image_to_pil(
    path_to_image: str
    ) -> tuple:
    """
    This function will read the image file present on the disk and
    load it into a PIL image. This PIL image file can then be used
    for inferencing with the model.

    Parameters:
        path_to_image:
            This is the path to the image that needs to be read.

    Returns:
        Tuple of:
            image:
                This is the read image in PIL format.

            image_name: 
                This is the name of the image that was read
                from disk.
    """

    # Get the image name
    dir, image_name = split(path_to_image)

    # Read the image
    image = Image.open(path_to_image).convert('RGB')

    # Read the image and convert it to RGB format
    return image, image_name



class PDFLoader():
    """
    This is the class that will be responsible for loading the PDF
    file from the disc into a Python PDF utility. It will provide
    the functions to convert the individual pages of the PDF file
    to images. This separate class is being written so that if in
    future we want to change the PDF library for Python, we can
    easily do so globally for all the rest of scripts that depend
    on the PDF loading utility.

    Parameters:
        path_to_pdf:
            This is the path to the PDF file that needs to be loaded
        verbose:
            If this flag is set to True, the entity will produce informative
            outputs on the command line.
    """

    def __init__(
        self,
        path_to_pdf: str = None,
        verbose: bool = False
        ) -> None:
        
        # Setup the global variables
        self.verbose = verbose
        self.path_to_pdf = None
        self.pdf_name = None
        self.fitz_doc = None
        self.page_count = None

        # If path to pdf is specified in the constructor
        if path_to_pdf is not None:
            self.load_pdf(
                path_to_pdf=path_to_pdf
            )

    def _load_pdf_to_fitz(
        self,
        path_to_pdf: str
        ) -> tuple:
        """
        This function will load a PDF file from the filesystem
        using the PyMuPDF library into a fitz object.

        Parameters:
            path_to_pdf:
                This is the path to the PDF that needs to be
                loaded.
        """

        # Get the name of the PDF document
        dir, pdf_name = split(path_to_pdf)

        # Read the PDF file into a fitz document and return it
        fitz_doc = fitz.open(path_to_pdf)

        return fitz_doc, pdf_name

    def load_pdf(
        self,
        path_to_pdf: str
        ) -> None:
        """
        This function will load the PDF file into the object of this
        class. This function is made to be accessed publicly i.e.
        external to this class.

        Parameters:
            path_to_pdf:
                This is the path to the PDF that will be loaded.
        """

        self.path_to_pdf = path_to_pdf

        self.fitz_doc, self.pdf_name = self._load_pdf_to_fitz(
            path_to_pdf=path_to_pdf
        )

        self.page_count = self.fitz_doc.page_count

        if self.verbose:
            print()
            print("Loaded File: " + self.pdf_name)
            print("Page Count of File: " + str(self.page_count))

        return None

    def get_page_in_pil(
        self,
        pg_no: int,
        dpi: int = 600
        ) -> Image:
        """
        This function will get the page from the document that is specified
        and convert it into a PIL Image and return that.

        Parameters:
            pg_no:
                This the page number that needs to be acquired.
            dpi:
                This the DPI at which to render the image.
        """

        if pg_no > self.page_count or pg_no < 0:
            print("Page number out of bounds for the PDF document")
            return None

        # Load the page
        page = self.fitz_doc.load_page(pg_no)

        # Convert to an image
        pix = page.get_pixmap(dpi=dpi)

        # Convert the image to PIL object and return
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
