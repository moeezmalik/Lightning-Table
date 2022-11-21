"""
Name:
    data-house-keeper.py

Description:
    This file is a utility script that was written in order
    to do some house cleaning operations on the dataset.

    More information about the utility can be found in the
    documentation.

    Limitations:
        Only work where annotations are in XML file format
        and image are in JPG file format.

Author:
    Name:
        M. Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

from os import listdir, remove, rename, makedirs
from os.path import join, exists
import sys
from shutil import copy

import argparse

from typing import Sequence, Tuple

# Smaller utility functions

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

# Main utility functions

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
    list_of_jpgs = []

    for file in listdir(path_to_folder):
        if file.endswith('.xml'):
            
            # Remove the extension and get the filename
            # Split the filename from the right until the
            # first period and get the first part
            filename = file.rsplit(".", 1)[0]
            
            list_of_xmls.append(filename)

        if file.endswith('.jpg'):

            # Remove the extension and get the filename
            # Split the filename from the right until the
            # first period and get the first part
            filename = file.rsplit(".", 1)[0]

            list_of_jpgs.append(filename)

    print('No. of XML files: ' + str(len(list_of_xmls)))
    print('No. of Image files: ' + str(len(list_of_jpgs)))

    # Find images that have their labels in XML
    list_of_jpgs_with_xml = list(set(list_of_jpgs).intersection(set(list_of_xmls)))

    print('No. of JPG files that has XML counterparts: ' + str(len(list_of_jpgs_with_xml)))

    return list_of_xmls, list_of_jpgs, list_of_jpgs_with_xml

def delete_extra_files(
    path_to_folder: str,
    xml_list: list,
    jpg_list: list,
    xml_jpg_intersect: list
    ) -> None:
    """
    Purpose:
        The purpose of this function is to delete extra files (XML and images)
        that in the provided labelled dataset folder.

    Arguments:
        path_to_folder (str):
            This is the path to the folder that contains all the image and
            annotation xml files
        xml_list (list):
            This is the list of file names of xml annotations found in the
            folder
        img_list (list):
            This is the list of images that are in the folder
        xml_jpg_intersect (list):
            This is the intersection between the list of image names and
            the list of xml file names. This essentially tells us which of
            the images have their labels done
    """

    extra_xml_list = list(set(xml_list).difference(set(xml_jpg_intersect)))
    extra_jpg_list = list(set(jpg_list).difference(set(xml_jpg_intersect)))

    print()
    print("Number of Extra XML files: " + str(len(extra_xml_list)))
    print("Number of extra images: " + str(len(extra_jpg_list)))

    # Delete extra XML files
    for file in extra_xml_list:

        filename = file + ".xml"
        full_path = join(path_to_folder, filename)

        # Try to delete the file
        try:
            remove(full_path)

        # Catch exceptions and errors
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))

    # Delete extra JPG files
    for file in extra_jpg_list:

        filename = file + ".jpg"
        full_path = join(path_to_folder, filename)

        # Try to delete the file
        try:
            remove(full_path)

        # Catch exceptions and errors
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))

    return None

def rename(
    path_to_folder: str,
    list_of_common_files: list,
    start_number: int = 1,
    folder_name: str = "renamed") -> None:
    """
    Purpose:
        This function will take in the list of files that are part of annotations
        and rename them with a sequence of images. This is done because for the
        case of assembling a training dataset, we do not need to worry about the
        names of training images and where they are coming from. It is much better
        in this case to have simple file names. The resulting name would be in
        this format for each pair of image and annotation:
            - image-{number}.jpg
            - image-{number}.xml
    
    Arguments:
        path_to_folder (str):
            This is the folder where the images and the annotations are located
        list_of_common_files (list):
            This should be the list of file names that need to be renamed. This
            function will assume that renaming is being done on files that both
            the jpg files and their respective annotation XML counterparts. 
        start_number (int):
            This parameter dicates which number the renaming will begin e.g. if
            10 is specified then files will be renamed starting from 10 and onwards.
        folder_name (str):
            This is the name of the folder into which the renamed files will be
            copied. This is done as a safety mechanism so that files do not
            get lost in the renaming process
    """

    count = start_number

    rename_folder_path = join(path_to_folder, folder_name)
    
    if not exists(rename_folder_path):
        makedirs(rename_folder_path)
    else:
        print("Exiting: Cannot rename, folder already exists")
        return None

    for name in list_of_common_files:

        old_name_xml = name + ".xml"
        old_path_xml = join(path_to_folder, old_name_xml)

        old_name_jpg = name + ".jpg"
        old_path_jpg = join(path_to_folder, old_name_jpg)

        new_name_xml = "image_" + str(count) + ".xml"
        new_path_xml = join(rename_folder_path, new_name_xml)

        new_name_jpg = "image_" + str(count) + ".jpg"
        new_path_jpg = join(rename_folder_path, new_name_jpg)

        copy(old_path_xml, new_path_xml)
        copy(old_path_jpg, new_path_jpg)

        count = count + 1

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
        return None

    # Gather information from the folder provided
    xml_list, jpg_list, xml_jpg_intersect = gather_info(pascalvoc_path)

    # Check if there are extra files in the folder
    if (len(xml_list) != len(xml_jpg_intersect) or len(jpg_list) != len(xml_jpg_intersect)):
        print()
        print("There seem to be extra XML or JPG file in the folder")

        # Ask if the extra files should be deleled. It is crucial for the
        # next step that the extra files are deleted at this stage. If the
        # reponse is to not delete extra files then the utility quits
        response = query_yes_no("Do you want to delete extra files?")

        if response:

            # Delete extra files
            print()
            print("Deleting extra files")
            delete_extra_files(
                path_to_folder=pascalvoc_path,
                xml_list=xml_list,
                jpg_list=jpg_list,
                xml_jpg_intersect=xml_jpg_intersect
            )
        else:
            print("Exiting without deleting extra files")
            return None

    else:
        print()
        print("There seem to be no extra files")

    # Now ask if renaming is required
    print()
    response = query_yes_no("Do you want rename the files?")

    if response:

        start_number = input("Enter the number from which the renaming will start (should be a positive integer): ")

        try:
            start_number = int(start_number)

        except ValueError:
            print("Valid number not entered")
            return None

        if start_number < 1:
            print("Valid number not entered")
            return None
        
        folder_name = input("Enter the name of folder to copy renamed files into: ")

        print()
        print("Renaming Files")
        rename(
            path_to_folder=pascalvoc_path,
            list_of_common_files=xml_jpg_intersect,
            start_number=start_number,
            folder_name=folder_name
        )


    else:
        print()
        print("Exiting: Files were not renamed")
        return None



    return None

if __name__ == '__main__':
    main()

