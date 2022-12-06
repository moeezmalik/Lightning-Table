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
from os import listdir, remove, makedirs, replace
from os.path import join, split, getsize, getmtime, exists, basename, isfile
from PIL import Image

# ##########################################################
# Functions
# ##########################################################

def get_filename_from_filepath(
    filepath: str
    ) -> str:
    """
    This function returns the name of the file from the
    full path to the file that is provided.

    Parameters:
        filepath:
            This is the full path to the file.
    """

    return basename(
        p=filepath
    )

def create_dir(
    path_to_dir: str
    ) -> bool:
    """
    This function will create a new directory (folder) on the
    path this provided.

    Parameters:
        path_to_dir:
            This is the full path to the directory that needs
            to be created. It must contain the name of the
            directory that needs to be created as well.

    Returns:
        success:
            A boolean type flag is returned that indicates if
            the directory was successfully created or not.
    """

    if not exists(path_to_dir):
        makedirs(path_to_dir)
        return True
    else:
        print("Exiting: Cannot create folder, already exists.")
        return False

def remove_file(
    path_to_file: str
    ) -> None:
    """
    This function will delete the file at the path that is provided.

    Parameters:
        path_to_file:
            This is the path to the file that needs to be deleted. It
            must contain the of the file that needs to be deleted.
    """
    
    # Try to delete the file
    try:
        remove(path_to_file)

    # Catch exceptions and errors
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

    return None

def move_file(
    source_path: str,
    dest_path: str
    ) -> None:
    """
    This function is made to move the files around in the file
    system.

    Parameters:
        source_path:
            This is the path to the source file that needs to move.
            It must contain the name of the file in question.

        dest_path:
            This is the path to the destination where the file will
            be moved. It must contain the name of the file as well.
    """

    # Try to move the file
    try:
        replace(
            src=source_path,
            dst=dest_path
        )

    # Catch exceptions and errors
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

    return None

def get_filesize(
    path_to_file: str
    ) -> int:
    """
    This function will return the file size of the file of which
    the path is provided.

    Parameters:
        path_to_file:
            This is the path to the file, for which the size
            needs to be extracted.

    Returns:
        size:
            This is the size of the file in bytes.
    """

    return getsize(filename=path_to_file)

def get_file_mod_time(
    path_to_file: str,
    ) -> float:
    """
    This function will return the time when the file was last
    modified.

    Parameters:
        path_to_file:
            This is the path to the file, for which the modification
            time needs to be extracted.

    Returns:
        time:
            The result will be a floating point value that will show
            the the time when file was last modified in seconds from
            epoch.
    """

    return getmtime(filename=path_to_file)


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


# ##########################################################
# Classes
# ##########################################################

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
