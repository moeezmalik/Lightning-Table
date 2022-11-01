"""
Name:
    pascalvoc-to-csv.py

Description:
    This is a utility script that was developed in order to
    convert the annotations from the PascalVOC format to the
    CSV type format.

    More information about the utility can be found in the
    documentation.

    Limitations:
        Only works for the cases where the annotations include
        only one class. This script was purpose written for
        reformating the labeling data for detecting tables in
        an image.

        This utility also assumes that the data-house-keeper.py
        was already run before running this one. This is because
        that this utility expects to find no extra files in
        the folder specified.

Author:
    Name:
        M. Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

import argparse
import xml.etree.cElementTree as et
import pandas as pd

from os import listdir
from os.path import isfile, join

from typing import Sequence

def xml_to_list(
    path_to_folder: str = None,
    xml_filename_without_extension: str = None
    ) -> list:
    """
    Purpose:
        This function will read the xml file, the path to which is
        is provided as the argument, and extract all the extensions
        from the file.

    Arguments:
        path_to_folder (str):
            This is the folder that contains the requested
            XML file

        xml_filename_without_extension (str):
            This is the name of the XML file without the extension (.xml)

    Returns:
        annotation_list (list):
            A list that contains the annotations that were in that
            particular file. The format will be (without any header):
            
            image_name, x1, y1, x2, y2, class_name

            Please note that the class_name will always be "Table" no
            matter what the XML file says. This is done to simplify
            the task at hand which is annotations of just one object
            i.e. Table in the images.
    """
    
    xml_filename_with_extension = xml_filename_without_extension + ".xml"
    path_to_xml = join(path_to_folder, xml_filename_with_extension)

    # This is the root of the XML tree
    tree = et.parse(path_to_xml)
    root = tree.getroot()

    # Here we extract the file name
    filename = xml_filename_without_extension

    # Get all objects in the current file
    objects_root = root.findall("object")

    # Initialise the list to fill all the annotations in
    read_annotations = []

    # Iterate over all found objects
    for object in objects_root:
        
        # This is the class name
        class_name = object.find("name").text

        # Set the class name to Table no matter what it is
        # in the XML file. This is done to simplify the 
        # implementation of this utility.
        class_name = "Table"


        # Now we go further into the tree and get the bounding box
        bnd_box_root = object.find("bndbox")

        x1 = bnd_box_root.find("xmin").text
        y1 = bnd_box_root.find("ymin").text
        x2 = bnd_box_root.find("xmax").text
        y2 = bnd_box_root.find("ymax").text

        read_annotations.append([filename, x1, y1, x2, y2, class_name])
    
    return read_annotations

def all_xmls_to_df(path_to_all_xml: str = None) -> pd.DataFrame:
    """
    Purpose:
        Given the folder where the annotations in XML file are placed,
        this function will go through all of the XML files and extract
        the annotations and place it in a pandas DataFrame.

    Arguments:
        path_to_all_xml (str):
            This is the path to the folder where all of the xml files
            are placed that will be read.
    
    Returns:
        annotations_df (pd.DataFrame):
            The function will return a pandas DataFrame with all the
            annotations.
    """

    total_xml_files_found = 0
    total_annotations_read = 0

    list_of_annotations = []

    for file in listdir(path_to_all_xml):

        if file.endswith('.xml'):

            total_xml_files_found += 1

            # Remove the extension and get the filename
            # Split the filename from the right until the
            # first period and get the first part
            filename = file.rsplit(".", 1)[0]

            read_annotations = xml_to_list(
                path_to_folder=path_to_all_xml,
                xml_filename_without_extension=filename
                )

            for annotation in read_annotations:
                if(len(annotation) == 6):
                    total_annotations_read += 1
                    list_of_annotations.append(annotation)

    print()
    print("Total Files Found: " + str(total_xml_files_found))
    print("Total Annotations Read: " + str(total_annotations_read))

    image_labels = pd.DataFrame(list_of_annotations, columns=["image_name", "x1", "y1", "x2", "y2", "class_name"])

    return image_labels

def main(args: Sequence = None) -> None:

    # Setup the argument parser to get the required parameters

    parser = argparse.ArgumentParser(description='Utility script for conversion of PascalVOC labelling to CSV format')

    parser.add_argument('--pascalvoc_path', help='Path to the folder with PascalVOC annotations', type=str, default=None)

    parser = parser.parse_args(args)

    pascalvoc_path = parser.pascalvoc_path

    print()
    print("PascalVOC to CSV Generator")

    print()
    print("Folder Specified: " + pascalvoc_path)


    df = all_xmls_to_df(pascalvoc_path)
    print(df.dtypes)


    return None

if __name__ == '__main__':
    main()