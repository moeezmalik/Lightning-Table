"""
This file will provide the functionality to perform
and generate the results for the evaluation of different
experiments in the thesis.

Author:
    Name:
        Muhammad Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

from infer import folder_of_pdf_to_csv
from tableextractor import save_to_excel as raw_tables_to_excel
from files import get_list_of_files_with_ext
from finalstep import Datasheet

import argparse
import yaml

class FullPipelineEvaluation():

    """
    This class will perform the evaluation for the full pipeline.
    It expects to be provided the path to the PDF folder from which
    first the tables will be detected. The raw values from these detected
    tables will then be extracted. Finally, the tables will be classified
    from the raw values and the structured values will be extracted. These
    structured values will then be compared against the ground truth values
    that also need to be specified.

    Args:
        path_to_pdf_folder:
            This is the path to the folder that contains the raw PDF files
        path_to_excel_folder:
            In this folder, the excel files will be generated and stored
            during the evaluation.
        path_to_gt_folder:
            This folder contains the files that contain the ground truth
            values. The files are expected to be in the JSON format.
    """

    def __init__(
            self,
            path_to_folder: str,
        ) -> None:

        # Assign the class variables
        self.path_to_pdf_folder = path_to_folder + "pdfs/"
        self.path_to_excel_folder = path_to_folder + "excels/"
        self.path_to_gt_folder = path_to_folder + "gt/"
        self.path_to_models_folder = path_to_folder + "models/"
    
    def evaluate(self):
        """
        This function will call all the necessary functions required
        for the evaluation procedure.
        """
        
        # First detect all the tables and save to a CSV file
        #self.detect_tables()

        # Then extract the raw tables and save to an excel file
        #self.recognise_tables()

        # Structure the values and extract the relevant ones
        self.perform_final_step()

    def detect_tables(self):
        """
        This function will perform the the process of table detection
        using the deep learning network. The settings that are used
        here for the model and the checkpoint are hard-coded depending
        on the results from training. In future this may be made
        modular.
        """

        print()
        print("Detecting Tables")
        print("----------------")

        folder_of_pdf_to_csv(
            path_to_folder=self.path_to_pdf_folder,
            model_name="VanillaFasterRCNN",
            ckpt_path=self.path_to_models_folder + "best-fasterrcnn.ckpt",
            output_path=self.path_to_excel_folder + "detectedtables.csv",
            conf_thresh=0.75,
            verbose=True
        )

    def recognise_tables(self):
        """
        This function will perform the table recognition part and extract
        the raw values from the detected table locations. These raw values
        will be saved as excel files in the folder corresponding to each
        table recognition algorithm used i.e. baseline, tabula or camelot.
        """

        print()
        print("Recognising Tables")
        print("----------------")

        # Use camelot to extract the raw tables and save them as files in
        # the camelot folder
        raw_tables_to_excel(
            path_to_csv=self.path_to_excel_folder + "detectedtables.csv",
            pdf_folder=self.path_to_pdf_folder,
            output_folder=self.path_to_excel_folder + "camelot/",
            reader="camelot"
        )

    def perform_final_step(self) -> dict:
        """
        This function will perform the final step as mentioned in the
        documentation. It will extract and structure the values from
        the excel files using regex patterns and return them as a 
        dictionary. This will be the major function that will perform
        the final step for all the excel files in all the folders i.e.
        baseline, camelot and tabula.
        """

        print()
        print("Final Step")
        print("----------------")

        path_to_baseline = self.path_to_excel_folder + "baseline/"
        path_to_camelot = self.path_to_excel_folder + "camelot/"
        path_to_tabula = self.path_to_excel_folder + "tabula/"

        self._fs_folder(path_to_folder=path_to_camelot)

    def _fs_folder(
            self,
            path_to_folder: str
        ) -> dict:
        """
        This is an internal function that will perform the final step
        for all the excel files in the folder specified.
        """

        list_of_files = get_list_of_files_with_ext(
            path_to_folder=path_to_folder,
            ext=".xlsx",
            verbose=True
        )

        # Go through all the files one by one and get the
        # values that were extracted
        for file in list_of_files:

            # Just run the function for now
            self._fs_file(
                path_to_file=file
            )

    def _fs_file(
            self,
            path_to_file: str
        ) -> dict:
        """
        This is an internal function that will perform the final step
        for the excel file specified and return a dictionary of items
        that were extracted. It will combine the electrical and thermal
        properties together into a single dictionary and just keep the
        values that were extracted and not the rows where they were
        found on.
        """

        # Load the yaml file that contains all the patterns for
        # detecting the correct columns and the values
        with open(self.path_to_models_folder + "patterns.yaml", "r") as stream:
            try:
                patterns = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)
        
        # Get type specific patterns
        elec_patterns = patterns.get("electrical")
        therm_patterns = patterns.get("temperature")

        curr_ds = Datasheet(
            path_to_excel=path_to_file,
            path_to_clf=self.path_to_models_folder + "nb_classifier.pickle",
            path_to_vec=self.path_to_models_folder + "vectoriser.pickle"
        )

        curr_ds.extract_electrical_props(patterns=elec_patterns)
        elec_extracted = curr_ds.extracted_elec

        curr_ds.extract_temp_props(patterns=therm_patterns)
        therm_extracted = curr_ds.extracted_temp

        for key, item in elec_extracted.items():
            elec_extracted[key] = item.get("vals")

        for key, item in therm_extracted.items():
            therm_extracted[key] = item.get("vals")

        return {
            "electrical" : elec_extracted,
            "thermal" : therm_extracted
        }


        


def parse_args():
    """
    This function will setup everything related to the
    parameters for utility to function as a command line
    app.
    """

    # Add the main description
    parser = argparse.ArgumentParser(
        description=u"""
        Provides the functionality to perform evaluation for
        different experiments in the thesis.
        """
    )

    # Add all of the required parameters
    required_named = parser.add_argument_group('required named arguments')

    required_named.add_argument(
        "-t", "--type",
        choices=['full', 'table_detection', 'table_classification'],
        required=True,
        metavar="TYPE",
        dest="type",
        help=u"""
        This specifies the type of evaluation that needs to be done. The
        choices are 'full', 'table_detection' and 'table_classification' 
        relating to different experiments performed in the thesis.
        """
    )

    required_named.add_argument(
        "-p", "--path",
        required=True,
        metavar="PATH",
        dest="path",
        help=u"""
        This is the path to the folder that contains further files and
        information for the evaluation procedure.
        """
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

    print()
    print("------------------")
    print("Evaluation Utility")
    print("------------------")

    if type == "full":
        print()
        print("Evaluating Full Pipeline")

        fpe = FullPipelineEvaluation(
            path_to_folder=path
        )

        fpe.evaluate()

if __name__=="__main__":

    # Execute the main function
    main()

