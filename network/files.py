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
from os.path import join, split, basename
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

    # Constructor Function
    
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
        self.is_text_based = None

        # If path to pdf is specified in the constructor
        if path_to_pdf is not None:
            self.load_pdf(
                path_to_pdf=path_to_pdf
            )


    # Internal Functions
    
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

    def _is_text_based(
        self
        ) -> None:
        """
        This internal function will classify the PDF either as
        text-based PDF or an image-based PDF. Depending upon the
        decision it will set a flag in the class variables that
        can indicate to the user of the this class whether the
        PDF image that has just been loaded is either text-based
        or not.
        """

        # Start with the assumption that the PDF is not
        # text-based
        self.is_text_based = False

        # Go through page by page and if text is found
        # then change the initial assumption
        for page in self.fitz_doc:
            
            if page.get_text("text"):

                # If the text is found, change assumption    
                self.is_text_based = True

                # Break the loop because we dont need to continue
                break

        return None


    # External Functions

    def load_pdf(
        self,
        path_to_pdf: str
        ) -> None:
        """
        This function will load the PDF file into the object of this
        class. This function is made to be accessed publicly i.e.
        external to this class. It is also used by the contructor of
        the class to load the image.

        Parameters:
            path_to_pdf:
                This is the path to the PDF that will be loaded.
        """

        self.path_to_pdf = path_to_pdf

        # Load the PDF and the name of the PDF
        self.fitz_doc, self.pdf_name = self._load_pdf_to_fitz(
            path_to_pdf=path_to_pdf
        )

        # Get the page count of the PDF
        self.page_count = self.fitz_doc.page_count

        # Determine if the PDF is image based or text-based
        self._is_text_based()

        if self.verbose:
            print()
            print("Loaded File: " + self.pdf_name)
            print("Page Count of File: " + str(self.page_count))

            # Print whether text-based or not
            if self.is_text_based:
                print("PDF is text-based")

            else:
                print("PDF does not contain text")

            print()

        return None

    def get_page_in_pil(
        self,
        pg_no: int,
        dpi: int = 600
        ) -> tuple:
        """
        This function will get the page from the document that is specified
        and convert it into a PIL Image and return that.

        Parameters:
            pg_no:
                This the page number that needs to be acquired.
            dpi:
                This the DPI at which to render the image.

        Returns:
            Tuple:
                scale_factor:
                    This is the ratio of the DPI at which the PDF was rendered
                    and the original DPI setting of the PDF. This can be used
                    for example, to scale down the bounding boxes that are
                    generated on the image to the bounding boxes in the PDF
                    coordinates.

                Image:
                    This is the rendered image of the current page as a PIL
                    object.
        """

        if pg_no > self.page_count or pg_no < 0:
            print("Page number out of bounds for the PDF document")
            return None

        # Load the page
        page = self.fitz_doc.load_page(pg_no)

        _ = page.mediabox.x1
        page_height = page.mediabox.y1

        # Convert to an image
        pix = page.get_pixmap(dpi=dpi)

        # Convert the image to PIL object and return
        pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        _ = pix.width
        image_height = pix.height

        scale_factor = image_height/page_height

        return scale_factor, pil_image

    def transform_bb_to_pdf_space(
        self,
        bbs: list,
        pg_no: int,
        scale_factor: int = 1
        ) -> list:
        """
        In the regular coordinate space the (0, 0) position for x and y-axis
        respecively is at the top left of the page. However in the PDF coordinate
        space the (0, 0) position for the x and y-axis respectively is at the
        bottom-left of the page and not the top-left.

        This function will convert the bounding box coordinates from the regular
        coordinate space to the PDF coordinate space. It will also scale down
        the boxes according to the scale factor of the page of PDF provided.

        Parameters:
            bb:
                This is a list of bounding boxes that needs to be transformed.
                It is a list of list with coordinates as [x1, y1, x2, y2] where
                x1, y1 is the top-left of the box and x2, y2 is the bottom-right
                of the box in the regular coordinate system.
            
            pg_no:
                This is the page number on which the bounding box exists. The page
                number is required because different pages in the PDF might have
                different sizes and thus different transformation of the page might
                be required.
            
            scale_factor:
                This is the scale factor between the image DPI and the PDF's
                original DPI. This is calculated for each page separately,
                in the "get_page_in_pil" function of this class.

        Returns:
            transformed_bbs:
                This is the list of transformed coordinates of the bounding boxes.
                It is a list of list with coordinates as [x1, y1, x2, y2] where
                x1, y1 is the top-left of the box and x2, y2 is the bottom-right
                of the box in the regular coordinate system.
        """

        transformed_bbs = []

        # For all bounding boxes in the list of bounding boxes provided
        for bb in bbs:

            # Scale down the bounding box to fit the original PDF size
            scaled_bb = [x/scale_factor for x in bb]

            # Load the page and get the page height
            page = self.fitz_doc.load_page(pg_no)
            page_height = page.mediabox.y1

            # Transform along the y-axis
            transformed_box = scaled_bb
            transformed_box[1] = page_height - transformed_box[1]
            transformed_box[3] = page_height - transformed_box[3]

            transformed_bbs.append(transformed_box)

        return transformed_bbs
