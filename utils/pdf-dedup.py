"""
This small utility was created as a preprocessing step before
running the inference of models on the PDF files. It will go
through the list of PDF files in the folder provided and will
detect the duplicates.

From the analysis of the folder that contains the files indicated
that the duplicate files have identical size and their time of
modification is very close i.e. within less than a minute. This
file exploits these features to detect the duplicates in the
folder. 

Note:
    This utility will only de-duplicate PDF files. It will ignore
    all other files in the folder that is provided.

Author:
    Name:
        Muhammad Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

# #################################################################
# All of the imports
# #################################################################

# Installed packages

import argparse
import pandas as pd
from datetime import datetime as dt

# From other self-written scripts

from files import get_list_of_files_with_ext, get_filesize, get_file_mod_time, create_dir, move_file, remove_file, join, get_filename_from_filepath


# #################################################################
# Other functions to run this utility
# #################################################################

def get_metadata_df(
    list_of_file_paths: list
    ) -> pd.DataFrame:
    """
    This function will take in a list of file paths, go through them
    one by one and then append the metadata alongwith the path to the
    file that was provided.

    Parameters:
        list_of_file_paths:
            This is the list that contains the paths relative or full
            in the form of strings, that will iterated upon to get the
            metadata.

    Returns:
        file_paths_metadata:
            This will be a pandas dataframe with metadata information
            about the files. Each element in the list will contain the
            path to the file in the first position, the size of the
            file in the second position and the date modified of the
            file at the last position.
    """

    # Initialise the list of files with metadata
    file_paths_metadata = []

    # Go through the files one by one
    for file in list_of_file_paths:

        # Get the information about the file
        file_size = get_filesize(file)
        mod_time = get_file_mod_time(file)

        # The modification time that is received from the
        # function above is in seconds elapsed since epoch,
        # that is not very readable, so we will convert it
        # into a datetime object
        mod_dt = dt.fromtimestamp(mod_time)

        # Assemble the information into a list
        info = [file, file_size, mod_dt]

        # Append the information into the main list
        file_paths_metadata.append(info)

    df = pd.DataFrame(file_paths_metadata, columns=['filepath', 'size', 'mod_time'])
    
    df = df.sort_values(
        by=['size', 'mod_time'],
        ascending=True
    )

    return df

def detect_duplicates(
    metadata_df: pd.DataFrame,
    verbose: bool = True
    ) -> pd.DataFrame:
    """
    The purpose of this function is to detect duplicates files using the
    metadata dataframe provided as an input. From the analysis of the
    folder that contains the files indicated that the duplicate files have
    identical size and their time of modification is very close i.e. within
    less than a minute. This function exploits these features to detect the
    duplicates in the folder.

    Parameters:
        metadata_df:
            This is the input dataframe that contains the crucial metadata
            information about the files. It expects the dataframe with the
            following columns (except the index):
            
            filepath, size, mod_time

            Where filepath is the path to the file, size is the file size and
            mod_time is the time when the file was last modified.

        verbose:
            If this flag is set, then this function will produce information
            about the files in the specifed folder on the command-line.
    """

    # Add an additional column that rounds the time stamps to the nearest
    # 5 minute mark
    metadata_df['mod_time_rnd'] = metadata_df['mod_time'].dt.round('5min')

    # Detect duplicates on the basis of file sizes and the time rounded off
    # to the closest 5 min mark.

    # Duplicate files will have the exact same file size. This is a necessary
    # but no sufficient condition. PDF files coming from the same manufacturer
    # can have the same file sizes even if the content is different. To counter
    # for that, we can use the last modified time of the file. It is noted that
    # the duplicate files have the modification time very close to each other.
    # So by rounding off to the closest 5 minute mark, we can identify the
    # duplicates because they will have the exact same file size and the exact
    # same rounded off time.
    nodups_df = metadata_df.drop_duplicates(
        subset=['size', 'mod_time_rnd'],
        keep='last'
    )

    num_duplicates = len(metadata_df.index) - len(nodups_df.index)

    if verbose:
        print("Number of duplicate files: " + str(num_duplicates))

    return nodups_df

def perform_action(
    command: str,
    dups_set: set,
    path_to_folder: str
    ) -> None:
    """
    This function will perform an action specified in the command-line
    arguments on the files that are deemed to be duplicates.

    Parameters:
        command:
            This is the command string that is taken from the command-line
            arguments. The possible options are:

            'noaction':
                No action will be performed on the files
            'move':
                A new folder called Duplicates will be created and duplicate
                files will be moved there.
            'delete':
                The duplicate files will be deleted.

        dups_set:
            This the set of fullpaths to the files that are deemed to be
            duplicates.

        path_to_folder:
            This the path to the folder that contains the pdf files. This is the
            original path that is taken from the command-line arguments. This is
            needed in order to move around the files.
    """

    if command == 'noaction':

        # Do nothing if no action is required.
        
        print()
        print("Taking no action on the duplicate files.")
        
        return None

    elif command == 'delete':
        
        # Delete the files here.

        # Go through the items one by one and delete
        for item in dups_set:
            remove_file(item)

        print()
        print("Completed the deletion action.")

        return None

    elif command == 'move':
        
        # Move the files to a folder named Duplicates here.

        # Create a string that contains the path to the folder
        folder_name = 'Duplicates'
        duplicates_folder_path = join(path_to_folder, folder_name)

        # Try creating a 'Duplicates' folder
        if create_dir(duplicates_folder_path):
            
            # If the directory is created successfully then
            # move all the files in that directory.
            
            # Move items one by one
            for item in dups_set:
                
                item_name = get_filename_from_filepath(item)
                path_of_item_with_duplicates_folder = join(duplicates_folder_path, item_name)

                move_file(source_path=item, dest_path=path_of_item_with_duplicates_folder)
                

        else:
            print("Error: Unable to create directory 'Duplicates'")
        
        print()
        print("Moved all files to folder named 'Duplicates'.")

        return None

    else:
        
        # Generate an error in case of programmatic mistake
        "perform_action: Invalid command specified"
        return None

# ##########################################################
# Functions to implement the command line utility
# ##########################################################


def parse_args():
    """
    This function will setup everything related to the
    parameters for utility to function as a command line
    app.
    """

    # Add the main description
    parser = argparse.ArgumentParser(
        description=u"""Detects duplicate PDF files in the folder that
        is provided and either deletes them or puts them in a separate
        folder depending upon the parameters that are provided."""
    )

    # Add all of the required parameters
    required_named = parser.add_argument_group('required named arguments')

    required_named.add_argument(
        "-p", "--path",
        required=True,
        metavar="PATH",
        dest="path",
        help=u"""This is the path to the folder that contains PDF files."""
    )

    required_named.add_argument(
        "-a", "--action",
        choices=['delete', 'move', 'noaction'],
        required=True,
        metavar="ACTION",
        dest="action",
        help=u"""This parameter specifies the action that will be taken on
        the duplicate PDF files in the folder that is provided. If 'delete'
        is specified then the duplicates will be deleted otherwise if 'move'
        is specified then the files will be moved to a newly created folder
        called 'Duplicates'. Please make sure that such a folder does not
        exist already before running this script. If 'noaction' is provided
        as an argument then no action will be taken against the duplicate
        files. This can be useful to just see how many duplicate files there
        are since the information will be shown regardless."""
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
    path = args.path
    action = args.action

    print()
    print("De-Duplication Utility")

    list_of_file_paths = get_list_of_files_with_ext(
        path_to_folder=path,
        ext='.pdf',
        verbose=True
    )

    metadata_df = get_metadata_df(
        list_of_file_paths=list_of_file_paths
    )

    no_dups_df = detect_duplicates(
        metadata_df=metadata_df,
        verbose=True
    )

    # We have the set of all the files and the
    # set of files that are unique i.e. they
    # do not contain any duplicates. Now we can
    # use set difference to get the set of files
    # that contain just the files that are deemed
    # duplicates so that the specified action can
    # be performed on them.
    all_files_set = set(metadata_df['filepath'])
    no_dups_set = set(no_dups_df['filepath'])

    dups_set = all_files_set - no_dups_set

    if len(dups_set) == 0:
        print()
        print("No duplicates found.")

    else:
        # Take an action on the duplicate files.
        perform_action(
            command=action,
            dups_set=dups_set,
            path_to_folder=path
        )





if __name__=="__main__":

    # Execute the main function
    main()