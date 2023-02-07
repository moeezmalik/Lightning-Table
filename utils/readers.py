"""
This file will hold the implementations for reading tables from the
pdf files given the coordinates of the table areas. This file will
implement readers using a custom algorithm as well as off-the-shelf
algorithms such as camelot.

Note:
    - The table coordinates must be in the PDF coordinate space.
    - This utility currently only works with text-based PDF files.

Author:
    Name:
        Muhammad Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

# #################################################################
# All of the imports
# #################################################################

# Installed packges
import fitz
import pandas as pd
import camelot
import tabula

# From other files
from files import basename, isfile

# #################################################################
# Reader Functions
# #################################################################

def tab(
    filepath: str,
    pages: int,
    table_areas: list,
    flavor: str = "stream"
) -> list:
    """
    This function will use the library camelot to read the tables
    from the provided pdf.

    Parameters:
        pdf_path:
            This is the path to the PDF file.

        pg_no:
            This is the page number of which the table exists.

        table_areas:
            This is the list of all the tables on the said page.

        flavor:
            This is a camelot specific parameter that sets the mode
            for reading the tables.
    """

    # Open the PDF document using PyMuPDF
    doc = fitz.open(filepath)

    # Get the required page from the pdf
    # The first page is indicated by 1 rather than 0, so we
    # have to cater for that
    page = doc[pages - 1]

    # Get the height of the page
    page_height = page.mediabox_size[1]
    
    # Tabula requires the table areas very differently to that of
    # Camelot. The incoming table areas are in PDF coordinate space
    # (bottom right of the document is (0, 0)) and are arranged
    # like [x0, y0, x1, y1]. Tabula needs them in normal coordinate
    # space (top-left of document is (0, 0)). So we need to transform
    # them to what is acceptable by tabula.
    table_area_tab = [
        page_height - table_areas[1],
        table_areas[0],
        page_height - table_areas[3],
        table_areas[2]
    ]

    extracted_tables = []

    if flavor == "stream":

        extracted_tables = tabula.read_pdf(
            input_path=filepath,
            output_format="dataframe",
            area=table_area_tab,
            pages=pages,
            stream=True
        )

        # By default the first row in the table read will be
        # set as the column header by tabula. We do not want that,
        # hence we shift the header to a row
        extracted_tables = [extracted_tables[0].T.reset_index().T]

    elif flavor == "lattice":

        extracted_tables = tabula.read_pdf(
            input_path=filepath,
            output_format="dataframe",
            area=table_area_tab,
            pages=pages,
            lattice=True
        )
        
        # By default the first row in the table read will be
        # set as the column header by tabula. We do not want that,
        # hence we shift the header to a row
        extracted_tables = [extracted_tables[0].T.reset_index().T]

    else:
        extracted_tables = []

    return extracted_tables

def cam(
    filepath: str,
    pages: str,
    table_areas: list,
    flavor: str = "stream"
    ) -> list:
    """
    This function will use the library camelot to read the tables
    from the provided pdf.

    Parameters:
        pdf_path:
            This is the path to the PDF file.

        pg_no:
            This is the page number of which the table exists.

        table_areas:
            This is the list of all the tables on the said page.

        flavor:
            This is a camelot specific parameter that sets the mode
            for reading the tables.
    """

    return camelot.read_pdf(
        filepath=filepath,
        pages=pages,
        table_areas=table_areas,
        flavor=flavor
    )

def baseline(
    filepath: str,
    pages: int,
    table_areas: list
    ) -> list:
    """
    This function will use the library camelot to read the tables
    from the provided pdf.

    Parameters:
        pdf_path:
            This is the path to the PDF file.

        pg_no:
            This is the page number of which the table exists. This
            starts from 1 and not zero.

        table_areas:
            This is the list of all the tables on the said page.
    """
    
    # Open the PDF document using PyMuPDF
    doc = fitz.open(filepath)

    # Get the required page from the pdf
    # The first page is indicated by 1 rather than 0, so we
    # have to cater for that
    page = doc[pages - 1]

    # Get the height of the page
    page_height = page.mediabox_size[1]

    # Get all the words from the page
    words = page.get_text("words", sort=True)

    # Determine which words lie in the table region, then collect
    # those words as a list
    table_x0 = table_areas[0]
    table_y0 = table_areas[1]
    table_x1 = table_areas[2]
    table_y1 = table_areas[3]

    raw_words = []

    for item in words:
    
        item_x0 = item[0]
        item_y0 = page_height - item[1]

        item_x1 = item[2]
        item_y1 = page_height - item[3]

        if item_x0 > table_x0:
            if item_y0 < table_y0:
                if item_x1 < table_x1:
                    if item_y1 > table_y1:
                        
                        avg_y = (item_y0 + item_y1) / 2.0
                        text = item[4]
                        block_no = item[5]

                        raw_word = [avg_y, text, block_no]

                        raw_words.append(raw_word)

    
    # Determine which words have the same y-positions, this means
    # they will belong to the same line

    raw_lines = []

    raw_line = []
    prev_pos = raw_words[0][0]

    for word in raw_words:
        
        curr_pos = word[0]

        # If decided that it is the same line
        if abs(curr_pos - prev_pos) < 5:
            raw_line.append(word)
        
        # If decided that it is not the same line
        else:

            # Commit the line
            raw_lines.append(raw_line)

            # Clear the previous line
            raw_line = []

            # Append the word for new line
            raw_line.append(word)

        prev_pos = curr_pos

    # Clean up the raw lines that were determined and combine the words
    # that were in the same block

    lines = []
    line = []

    prev_block_no = raw_lines[0][0][2]

    w_string = ""

    for raw_line in raw_lines:
        
        prev_block_no = raw_line[0][2]

        for raw_word in raw_line:
            
            curr_block_no = raw_word[2]
            raw_word_string = raw_word[1]

            if curr_block_no == prev_block_no:
                
                if not w_string:
                    w_string = w_string + raw_word_string
                else:
                    w_string = w_string + " " + raw_word_string
            else:

                line.append(w_string)
                w_string = raw_word_string

            prev_block_no = curr_block_no

        line.append(w_string)
        w_string = ""

        lines.append(line)

        line = []

    table_df = pd.DataFrame(lines)

    return [table_df]