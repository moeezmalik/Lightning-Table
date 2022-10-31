"""
Name:
    data-house-keeper.py

Description:
    This file is a utility script that was written in order
    to do some house cleaning operations on the dataset.

    More information about the utility can be found in the
    documentation.

Author:
    Name:
        M. Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

from os import listdir
from os.path import isfile, join
import sys

import argparse

from typing import Sequence, Tuple

def query_yes_no(question: str, default="yes") -> bool:
    """
    Purpose:
        Ask a yes/no question via raw_input() and return their answer.
    
    Arguments:
        question (str):
            This is a question string that is presented to the user.
        default (str):
            This is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).
    
    Returns:
        answer (bool):
            The "answer" return value is True for "yes" or False for "no".
    """
    
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

def gather_info(path_to_folder: str = None) -> Tuple:
    """
    Purpose
        This function will gather information about the files
        present in the specified folder. It is implemented
        to work with the PascalVOC annotation style. It will
        look for images in .jpg, .jpeg or .png format and 
        annotations in the .xml file format.

    Arugments:
        path_to_folder (str):
            This is the string that provides the path to the
            folder that contains the labeling in the PascalVOC
            format

    Returns:
        A tuple (Tuple) of:
            list_of_xmls  (list):
                This is a list that contains the names of all
                the XML files found in the folder
            list_of_imgs (list):
                This is a list that contains the names of all
                the image files found in the folder
            list_of_imgs_with_xml (list):
                This is a list that contains the names of image
                files that have their annotations in the folder
    """

    # Error check the path to folder
    if (path_to_folder is None):
        print("Error: Cannot gather information because no folder path specified")
        return None

    # Print some initial stats
    print()
    print("Gathering Information")
    print()
    print("Folder Specified: " + path_to_folder)
    print()

    list_of_xmls = []
    list_of_imgs = []

    for file in listdir(path_to_folder):
        if file.endswith('.xml'):
            
            # Remove the extension and get the filename
            # Split the filename from the right until the
            # first period and get the first part
            filename = file.rsplit(".", 1)[0]
            
            list_of_xmls.append(filename)

        if file.endswith(('.jpg', '.jpeg', '.png')):

            # Remove the extension and get the filename
            # Split the filename from the right until the
            # first period and get the first part
            filename = file.rsplit(".", 1)[0]

            list_of_imgs.append(filename)

    print('No. of XML files: ' + str(len(list_of_xmls)))
    print('No. of JPG files: ' + str(len(list_of_imgs)))

    # Find images that have their labels in XML
    list_of_imgs_with_xml = list(set(list_of_imgs).intersection(set(list_of_xmls)))

    print('No. of JPG files that has XML counterparts: ' + str(len(list_of_imgs_with_xml)))

    return list_of_xmls, list_of_imgs, list_of_imgs_with_xml

def delete_extra_files(
    path_to_folder: str = None
    ) -> None:
    """
    Purpose:
        The purpose of this function is to delete extra files (XML and images)
        that in the provided labelled dataset folder.
    """

    return None

def main(args: Sequence = None) -> None:

    # Setup the argument parser to get the required parameters

    parser = argparse.ArgumentParser(description='Utility script for house keeping of dataset labelling')

    parser.add_argument('--pascalvoc_path', help='Path to the folder with PascalVOC labels', type=str, default=None)

    parser = parser.parse_args(args)

    pascalvoc_path = parser.pascalvoc_path

    # Error check the non-optional parameters
    if(pascalvoc_path is None):
        print("Error: No PascalVOC path specified, cannot continue")

    # Gather information from the folder provided
    xml_list, jpg_list, xml_jpg_intersect = gather_info(pascalvoc_path)

    if (len(xml_list) != len(xml_jpg_intersect) or len(jpg_list) != len(xml_jpg_intersect)):
        print()
        print("There seem to be extra XML or JPG file in the folder")

        response = query_yes_no("Do you want to delete extra files?")

        if response:
            print("Deleting extra files")
        else:
            print("Exiting without deleting extra files")
            return None

    else:
        print()
        print("There seem to be no extra files")

    return None

if __name__ == '__main__':
    main()

