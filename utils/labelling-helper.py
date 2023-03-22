"""
This script was written to aid the process of labelling for
table classification. This is supposed to be temporary.

Author:
    Name:
        Muhammad Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

# #################################################################
# All of the imports
# #################################################################

# Installed Packages

import pandas as pd
from os import system

# Other libraries

from files import get_list_of_files_with_ext, join
from cellstructure import Datasheet


# #################################################################
# Classes and Functions
# #################################################################

def main():
    """
    This is the main function that will carry out the labelling process
    """

    print("Labelling Helper")

    path_to_folder = "/Volumes/T7/thesis-data/test/selected_excel/"
    path_to_csv = "/Volumes/T7/thesis-data/test/selected_excel/labels.csv"

    file_list = get_list_of_files_with_ext(
        path_to_folder=path_to_folder, ext="xlsx",
        randomise=True,
        verbose=False
    )

    total_files = len(file_list)

    labels = []
    ds = Datasheet()

    for count, file in enumerate(file_list, start=1):

        # Load the tables from the current file
        path_to_file = join(path_to_folder, file)
        ds.load_tables_from_excel(path_to_excel=path_to_file)

        # Show the name of the file
        print("---")
        print("File {}/{} - {}".format(count, total_files, ds.name))
        print("---")
        
        # Go through the tables and get the label from user
        for table in ds.tables:
            
            print(table.name)
            print(table.raw_df)

            label = input("Enter Label for Table: ")

            # Assemble one record
            row = [ds.name, table.name, label]
            labels.append(row)

        # Clearing screen on Linux and macOS
        system('clear')

    df = pd.DataFrame(labels, columns=['filename', 'tablename', 'class'])
    df.to_csv(path_to_csv, index=False, header=False)
    
    print("---")
    print("Labels saved in CSV file")
    print("---")
    
    print(df)

if __name__=="__main__":
    main()