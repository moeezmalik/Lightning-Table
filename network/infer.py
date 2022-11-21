"""
This is the file that implements the command-line
utility for performing inference using the models
in this repository.

Author:
    Name:
        Muhammad Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

# ##########################################################
# Imports
# ##########################################################

# From installed packages
import argparse

# From custom scripts
from inferencers import pdf_to_csv, folder_of_pdf_to_csv


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
        description=u"""Shows the output of the model using the available models
        and the files provided as input. This utility does not save anything."""
    )

    # Add all of the required parameters
    required_named = parser.add_argument_group('required named arguments')

    required_named.add_argument(
        "-t", "--type",
        choices=['pdf', 'pdfs_folder'],
        required=True,
        metavar="TYPE",
        dest="type",
        help=u"""The type of file on which to perform the inference. Valid options
        are 'pdf', 'pdfs_folder'. Please look at the documentation for more
        information about this parameter."""
    )

    required_named.add_argument(
        "-p", "--path",
        required=True,
        metavar="PATH",
        dest="path",
        help=u"""This is the path to the folder or file that needs to be visualised."""
    )

    required_named.add_argument(
        "-m", "--model",
        required=True,
        metavar="MODEL_NAME",
        dest="model_name",
        help=u"""This is the name of model that needs to be utilised to perform
        inference. This should match the model for which the checkpoints are
        to be provided."""
    )

    required_named.add_argument(
        "-w", "--weights",
        required=True,
        metavar="CHECKPOINT_PATH",
        dest="ckpt_path",
        help=u"""This is the path to the checkpoint that needs to be used.
        The checkpoint should belong to the model that is provided."""
    )

    required_named.add_argument(
        "-o", "--output",
        required=True,
        metavar="OUTPUT_PATH",
        dest="output_path",
        help=u"""This should be the path to the where the output file will be
        saved. It must include the file name as well."""
    )

    # All of the optional parameters
    parser.add_argument(
        "-c", "--confidence",
        required=False,
        type=float,
        default=0.75,
        metavar="CONFIDENCE_THRESHOLD",
        dest="conf_thresh",
        help=u"""This parameter specifies the confidence threshold.
        It is a floating point number between 0 and 1.
        When the model makes predictions, it specfies how confident
        it is on those predicitons as well. This parameter will
        specify the cutoff, the predictions below this cutoff
        will not be considered. This is an optional parameter,
        if no value is specified then 0.75 will be taken."""
    )

    parser.add_argument(
        "-d", "--dpi",
        required=False,
        type=int,
        default=600,
        metavar="DPI",
        dest="dpi",
        help=u"""In order to detect tables in the PDF file. Each
        individual page is first converted to an image that will
        then be passed through the model. This parameter specifies
        the DPI for that rendered image. The higher this number, the
        higher the resolution of the rendered image will be but the
        process of inference will also will be slower. This parameter
        also needs to be selected in accordance with the utility that
        will extract the textual data from the table e.g. Camelot. This
        is because if the resolution is different, the coordinates of
        the detected table will also be different."""
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
    type = args.type
    path = args.path
    model_name = args.model_name
    cpkt_path = args.ckpt_path
    conf_thresh = args.conf_thresh
    output_path = args.output_path
    dpi = args.dpi

    print()
    print("Inferring Utility")

    if type == "pdf":
        
        print("Type: Single PDF")

        pdf_to_csv(
            path_to_pdf=path,
            model_name=model_name,
            ckpt_path=cpkt_path,
            conf_thresh=conf_thresh,
            output_path=output_path,
            dpi=dpi,
            verbose=True
        )


    elif type == "pdfs_folder":

        print("Type: Folder of PDFs")

        folder_of_pdf_to_csv(
            path_to_folder=path,
            model_name=model_name,
            ckpt_path=cpkt_path,
            conf_thresh=conf_thresh,
            output_path=output_path,
            dpi=dpi,
            verbose=True
        )



if __name__=="__main__":

    # Execute the main function
    main()