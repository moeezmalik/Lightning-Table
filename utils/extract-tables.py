"""
This is a command-line utility. Given a CSV file with name of
PDF file and the table coordinates on the current page in the
PDF coordinate space, this utility will extract the tables from
the PDF file and save it in a structured format.

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
import argparse
import pandas as pd
import camelot

# From other files
from files import basename, isfile

# #################################################################
# Other Functions
# #################################################################

def resolve_pdf_info(
    path_to_csv: str,
    pdf_folder: str
    ) -> list:
    """
    This function will go through the CSV file that is provided and
    convert the information present into something that we can
    iterate over in python.

    Paramters:
        path_to_csv:
            This is the path to the CSV file that contains information
            about where the bounding boxes are located in the PDF file.
            The CSV file should contain the information in the following
            order:
                filename,pageno,x1,y1,x2,y2
            The CSV file should not include the header mentioned above.

        pdf_folder:
            This is the path to the folder that contains the PDF files
            that are mentioned in the CSV file.

    Returns:
        info_list:
            This would be the list that contains information about the PDF
            files. Each item in the list will be one PDF.
    """

    # Read the CSV file
    info_df = pd.read_csv(
        filepath_or_buffer=path_to_csv,
        names=['filename', 'page', 'x1', 'y1', 'x2', 'y2']
        )

    # Apply transformations to get the desired data
    info_df = info_df.groupby(['filename', 'page']) \
                    .apply(lambda x: x[['x1', 'y1', 'x2', 'y2']].values.tolist()) \
                    .reset_index(name='boxes')

    info_df = info_df.groupby(['filename']) \
                    .apply(lambda x: x[['page', 'boxes']].values.tolist()) \
                    .reset_index(name='pages')

    # Change the file names to full paths
    info_df['filename'] = pdf_folder + info_df['filename'].astype(str)
    info_df.rename(columns={'filename':'filepath'}, inplace=True)

    info_list = info_df.values.tolist()

    return info_list

def single_page_to_tables(
    pdf_path: str,
    pg_no: str,
    table_areas: str
    ) -> list:
    """
    This function will extract all the tables from the page specified of the
    PDF specified.

    Parameters:
        pdf_path:
            This is the path to the PDF file.

        pg_no:
            This is the page number of which the table exists.

        table_areas:
            This is the list of all the tables on the said page.
    """

    total_tables = len(table_areas)

    extracted_tables = []

    for count, table in enumerate(table_areas, start=1):
        
        # Assemble the table string for camelot
        cam_table_string = [str(table[0]) + "," + str(table[1]) + "," + str(table[2]) + "," + str(table[3])]

        # Use Camelot to read a single table
        try:
            read_table = camelot.read_pdf(
                filepath=pdf_path,
                pages=str(pg_no),
                table_areas=cam_table_string,
                flavor='stream'
            )

        except ValueError:
            print("Page: {} | Table: {}/{} | Error: Coordinates Might be Wrong".format(pg_no, count, total_tables))

        else:
            extracted_tables.append(read_table[0])
            print("Page: {} | Table: {}/{} | Successfully Read".format(pg_no, count, total_tables))
        
    return extracted_tables

def single_pdf_to_tables(
    pdf_path: str,
    tables: list
    ) -> list:
    """
    This function will generate an excel file consisting of all tables found
    in a single PDF file.

    Parameters:
        pdf_path:
            This is the path to the PDF file from which to extract the tables.

        tables:
            This is a list that should contain information about the pages on which
            the tables are to be found and the coordinates of where the tables are
            to be found.
    """

    all_page_extracted_tables = []

    # Go page by page get the tables
    for page in tables:

        pg_no = page[0]
        table_areas = page[1]

        curr_page_extracted_tables = single_page_to_tables(
            pdf_path=pdf_path,
            pg_no=pg_no,
            table_areas=table_areas
        )

        all_page_extracted_tables.extend(curr_page_extracted_tables)

    return all_page_extracted_tables


# Functions for PDF to excel

def save_tables_to_excel(
    tables: list,
    name: str,
    output_folder: str
    ) -> None:
    """
    This function will save a list of Camelot tables to individual sheets
    in the designated excel file.

    tables:
        This is a list of tables that are read by Camelot.

    name:
        This is the name of the file by which the excel file will be saved.

    output_folder:
        This is the path to the folder where the excel file should be saved.
    """

    # Assemble path to excel file
    name_of_excel_file = name + '.xlsx'
    path_to_excel_file = output_folder + name_of_excel_file

    # Write table to individual sheets
    with pd.ExcelWriter(path_to_excel_file) as writer:

        # Go over the tables one by one
        for count, table in enumerate(tables, start=1):
            
            # Convert the Camelot tables to a pandas DataFrame to write
            # to excel file
            table.df.to_excel(writer, sheet_name="Table_{}".format(count))

    print("Saved tables to: " + name_of_excel_file)

    return None

def single_pdf_to_excel(
    pdf_path: str,
    tables: list,
    output_folder: str
    ) -> None:
    """
    This function will get tables from an individual PDF file and then save
    them to an excel file.

    Parameters:
        pdf_path:
            This is the path to the PDF file from which to extract the tables.

        tables:
            This is a list that should contain information about the pages on which
            the tables are to be found and the coordinates of where the tables are
            to be found.

        output_folder:
            This is the path to the output folder where the excel file will be saved.
    """

    # Get the name of the PDF file without extension
    file = basename(pdf_path)
    name = file.rsplit('.')[0]

    print()
    print("---")
    print("Evaluating PDF: " + file)

    # Check if the file exists and if it does then
    # extract tables from it to save to excel
    if (isfile(path=pdf_path)):

        # Extract all the tables fromt the PDFs
        extracted_tables = single_pdf_to_tables(
            pdf_path=pdf_path,
            tables=tables
        )

        # Save the tables to an excel file
        save_tables_to_excel(
            tables=extracted_tables,
            name=name,
            output_folder=output_folder
        )

    else:
        print("Skipping: File does not exist")

    print("---")


    return None

def list_of_pdfs_to_excel(
    pdf_and_tables: list,
    output_folder: str
    ) -> None:
    """
    This function will take in the resolved information about the PDFs and
    the table regions and save them to individual excel files for each
    PDF file.

    Parameters:
        pdf_and_tables:
            This is a list that contains the PDF file path and the information
            about the table regions on relevant pages.

        output_folder:
            This is the path to the folder where the excel files will be saved.
    """

    # Go through PDFs one by one and save to excel files.
    for pdf in pdf_and_tables:
        path = pdf[0]
        tables = pdf[1]

        single_pdf_to_excel(
            pdf_path=path,
            tables=tables,
            output_folder=output_folder
        )



    return None

def excel_action(
    path_to_csv: str,
    pdf_folder: str,
    output_folder: str
    ) -> None:
    """
    This is the front-facing function for extracting the table from the
    the PDF files that are provided as part of the CSV file and saving them
    to the excel files.

    Parameters:
        path_to_csv:
            This is the path to the CSV file that contains information
            about where the bounding boxes are located in the PDF file.
            The CSV file should contain the information in the following
            order:
                filename,pageno,x1,y1,x2,y2
            The CSV file should not include the header mentioned above.

        pdf_folder:
            This is the path to the folder that contains the PDFs which are
            mentioned in the CSV file.

        output_folder:
            This is the path to the folder where the generated files should
            be placed.
    """

    # Resolve the table information from the CSV file and get the paths
    info_list = resolve_pdf_info(
        path_to_csv=path_to_csv,
        pdf_folder=pdf_folder
    )

    list_of_pdfs_to_excel(
        pdf_and_tables=info_list,
        output_folder=output_folder
    )

    pass

# #################################################################
# Command-line Functions
# #################################################################

def parse_args():
    """
    This function will setup everything related to the
    parameters for utility to function as a command line
    app.
    """

    # Add the main description
    parser = argparse.ArgumentParser(
        description=u"""Given a CSV file with name of PDF file and the table
                        coordinates on the current page in the PDF coordinate
                        space, this utility will extract the tables from
                        the PDF file and save it in a structured format."""
    )

    # Add all of the required parameters
    required_named = parser.add_argument_group('required named arguments')

    required_named.add_argument(
        "-f", "--folder",
        required=True,
        metavar="PATH",
        dest="folder_path",
        help=u"""This is the path to the folder that contains all of the PDF
                 files."""
    )

    required_named.add_argument(
        "-c", "--csv",
        required=True,
        metavar="PATH",
        dest="csv_path",
        help=u"""This is the path to the CSV file that contains the PDF names
                 and the table coordinates. The CSV file should contain the
                 the information in the following order: filename,pageno,x1,y1,x2,y2.
                 The file should not include this header itself."""
    )

    required_named.add_argument(
        "-t", "--type",
        choices=['excel'],
        required=True,
        metavar="PATH",
        dest="type",
        help=u"""This is the path to the folder where the generate output files
                 will be placed. One file will be generated for each PDF file.
                 Currently only excel file generation is implemented so the only
                 choice for this parameter is 'excel'."""
    )

    required_named.add_argument(
        "-o", "--output",
        required=True,
        metavar="PATH",
        dest="output_path",
        help=u"""This is the path to the folder where the generate output files
                 will be placed. One file will be generated for each PDF file."""
    )

    # Parse the arguments and result the namespace object
    return parser.parse_args()

def main():
    """
    This is the main function that will execute all of the relevant
    scripts after reading the parsed arguments.
    """

    # Get the parsed arguments
    args = parse_args()

    # Extract all parameters from the command line
    folder_path = args.folder_path
    csv_path = args.csv_path
    type = args.type
    output_path = args.output_path


    print()
    print("Table Extraction Utility")

    print("PDF Folder Path: " + folder_path)
    print("CSV Path: " + csv_path)
    print("Output Folder: " + output_path)

    if type == 'excel':
        print("Output Type: Excel")
        
        excel_action(
            path_to_csv=csv_path,
            pdf_folder=folder_path,
            output_folder=output_path
        )



if __name__=="__main__":

    # Execute the main function
    main()