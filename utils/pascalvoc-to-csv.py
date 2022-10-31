"""
Name:
    pascalvoc-to-csv.py

Description:
    This is a utility script that was developed in order to
    convert the annotations from the PascalVOC format to the
    CSV type format.

    More information about the utility can be found in the
    documentation.

Author:
    Name:
        M. Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

import argparse

from typing import Sequence

def main(args: Sequence = None) -> None:

    # Setup the argument parser to get the required parameters

    parser = argparse.ArgumentParser(description='Utility script for conversion of PascalVOC labelling to CSV format')

    parser.add_argument('--pascalvoc_path', help='Path to the folder with PascalVOC annotations', type=str, default=None)

    parser = parser.parse_args(args)

    pascalvoc_path = parser.pascalvoc_path



    return None

if __name__ == '__main__':
    main()