"""
This file will provide the functionality to extract information
from the PDF files using the pipeline developed while doing the
master thesis.

Author:
    Name:
        Muhammad Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

from infer import folder_of_pdf_to_csv
from tableextractor import save_to_excel as raw_tables_to_excel
from files import get_list_of_files_with_ext, basename
from finalstep import Datasheet

import pprint
import argparse
import yaml
import pandas as pd
from os import mkdir
from os.path import isdir

from statistics import mode

class FolderExtraction():
    """
    This class will extract the data from a folder full of PDF data sheets.
    The class also provides functions to save the data to an excel file.

    Args:
        path_to_folder:
            This is the path to the folder that contains the PDf files.

        path_to_models:
            This is the path to the folder that contains all the trained
            machine learning models.

        path_to_excels:
            This is the path where the temporary excel files will be saved.
    """

    # EXTERNAL FUNCTIONS
    # Functions that may be called using the class object

    def __init__(
            self,
            path_to_folder: str,
            path_to_models: str,
            path_to_excels: str = None
        ) -> None:

        # Assign the class variables
        self.path_to_pdf_folder = path_to_folder
        self.path_to_models_folder = path_to_models
        self.path_to_excel_folder = path_to_excels

        self.extracted_values = None
        self.confusion_matrix = None
        self.metrics = None

        # If no folder for temporary output is specified
        if self.path_to_excel_folder is None:

            self.path_to_excel_folder = self.path_to_pdf_folder + "temp/"
            
            # Create temporary folder if it doesnt already exist
            if not isdir(self.path_to_excel_folder):
                mkdir(path=self.path_to_excel_folder)
    
    def extract(self):
        """
        This function will call all the necessary functions required
        for the evaluation procedure.
        """
        
        # First detect all the tables and save to a CSV file
        self._detect_tables()

        # Then extract the raw tables and save to an excel file
        self._recognise_tables()

        # Structure the values and extract the relevant ones
        self._perform_final_step()

    def save_to_excel(
            self,
            path: str = "extracted.xlsx"
            ) -> None:
        """
        This is the path where the excel file containing the extracted
        values will be saved.

        Args:
            path:
                This is the path to the excel file where the extracted
                values will be saved.
        """

        print()
        
        print("Saving to excel")
        print("---------------")

        final_list = []

        for name, prop_type in self.extracted_values.items():
            
            # Get the values
            thermal = prop_type.get("thermal")
            electrical = prop_type.get("electrical")

            year = prop_type.get("misc").get("year")

            # Assuming equal lengths of extracted arrays of values
            value_count = []

            for value_type, value_list in electrical.items():
                if value_list is not None:
                    value_count.append(len(value_list))

            most_freq_count = mode(value_count)

            curr_file_list = [[name] * most_freq_count]
            curr_file_list.append([year] * most_freq_count)

            # Adding electrical properties
            elec_prop_types = ["eff", "pmpp", "vmpp", "impp", "voc", "isc", "ff"]

            for prop in elec_prop_types:
                vals = electrical.get(prop)
                if vals is None:
                    curr_file_list.append([""] * most_freq_count)
                else:
                    curr_file_list.append(vals)

            # Adding thermal properties
            thermal_prop_types = ["isc", "pmpp", "voc"]

            for prop in thermal_prop_types:
                vals = thermal.get(prop)
                if vals is None:
                    curr_file_list.append([""] * most_freq_count)
                else:
                    curr_file_list.append(vals * most_freq_count)

            # Transpose the list
            curr_file_list = list(map(list, zip(*curr_file_list)))

            final_list.extend(curr_file_list)

        # Create a dataframe
        final_df = pd.DataFrame(final_list,
                                columns=[
                                    "name",
                                    "year",
                                    "eff",
                                    "pmpp",
                                    "vmpp",
                                    "impp",
                                    "voc",
                                    "isc",
                                    "ff",
                                    "isc",
                                    "pmpp",
                                    "voc"
                                    ]
                                )
            
        #print(final_df)
            
        # Write to the excel file
        with pd.ExcelWriter(
                path=path, mode='w'
            ) as writer:
            
            final_df.to_excel(
                writer,
                index=False
                )


    # INTERNAL FUNCTIONS
    # Functions that are meant to be used inside the class
    
    def _detect_tables(self):
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
            model_name="VanillaFasterRCNNV2",
            ckpt_path=self.path_to_models_folder + "best-fasterrcnn-v2.ckpt",
            output_path=self.path_to_excel_folder + "detectedtables.csv",
            conf_thresh=0.75,
            verbose=True
        )

    def _recognise_tables(self):
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
            output_folder=self.path_to_excel_folder,
            reader="camelot"
        )

    def _perform_final_step(self):
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

        all_files = self._fs_folder(
            path_to_excel_folder=self.path_to_excel_folder,
            path_to_pdf_folder=self.path_to_pdf_folder
            )

        self.extracted_values = all_files


    def _fs_folder(
            self,
            path_to_excel_folder: str,
            path_to_pdf_folder: str
        ) -> dict:
        """
        This is an internal function that will perform the final step
        for all the excel files in the folder specified.
        """

        list_of_files = get_list_of_files_with_ext(
            path_to_folder=path_to_excel_folder,
            ext=".xlsx",
            verbose=True
        )

        all_files_extracted = {}

        # Go through all the files one by one and get the
        # values that were extracted
        for file in list_of_files:
            
            filename = str(basename(file)).rsplit(sep=".")[0]

            # Just run the function for now
            extracted_vals = self._fs_file(
                path_to_excel_file=file,
                path_to_pdf_file=path_to_pdf_folder + filename + ".pdf"
            )

            all_files_extracted[filename] = extracted_vals

        return all_files_extracted

    def _fs_file(
            self,
            path_to_excel_file: str,
            path_to_pdf_file: str
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
        with open(self.path_to_models_folder + "patterns.yaml", "r", encoding='utf-8') as stream:
            try:
                patterns = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)
        
        # Get type specific patterns
        elec_patterns = patterns.get("electrical")
        therm_patterns = patterns.get("temperature")

        curr_ds = Datasheet(
            path_to_pdf=path_to_pdf_file,
            path_to_excel=path_to_excel_file,
            path_to_clf=self.path_to_models_folder + "nb_classifier.pickle",
            path_to_vec=self.path_to_models_folder + "vectoriser.pickle"
        )

        curr_ds.extract_electrical_props(patterns=elec_patterns)
        elec_extracted = curr_ds.extracted_elec

        curr_ds.extract_temp_props(patterns=therm_patterns)
        therm_extracted = curr_ds.extracted_temp

        curr_ds.extract_misc_props()
        misc_extracted = curr_ds.extracted_misc

        curr_ds.extract_mech_props()

        if elec_extracted is not None:
            for key, item in elec_extracted.items():
                vals = item.get("vals")

                if vals is not None:
                    vals = [ str(x) for x in vals ]

                elec_extracted[key] = vals
        
        if therm_extracted is not None:
            for key, item in therm_extracted.items():
                vals = item.get("vals")

                if vals is not None:
                    vals = [ str(x) for x in vals ]

                therm_extracted[key] = vals

        return {
            "electrical" : elec_extracted,
            "thermal" : therm_extracted,
            "misc": misc_extracted
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
        Provides the functionality to extract data from
        solar cell datasheets and save them in an excel
        file.
        """
    )

    # Add all of the required parameters
    required_named = parser.add_argument_group('required named arguments')

    required_named.add_argument(
        "-p", "--path",
        required=True,
        metavar="PATH",
        dest="path",
        help=u"""
        This is the path to the folder that contains the PDF files to be
        processed.
        """
    )

    required_named.add_argument(
        "-m", "--models",
        required=True,
        metavar="MODELS",
        dest="models",
        help=u"""
        This is the path to the folder that contains the trained machine
        learning models.
        """
    )

    required_named.add_argument(
        "-e", "--excel",
        required=True,
        metavar="EXCEL",
        dest="excel",
        help=u"""
        This is the path where the excel file should be saved.
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
    pdf_folder_path = args.path
    models_folder_path = args.models
    excel_output_path = args.excel

    print()
    print("------------------")
    print("Extraction Utility")
    print("------------------")

    print()
    print("Extracting data from folder of PDFs")

    fe = FolderExtraction(
        path_to_folder=pdf_folder_path,
        path_to_models=models_folder_path
    )

    fe.extract()

    fe.save_to_excel(path=excel_output_path)

    
if __name__=="__main__":

    # Execute the main function
    main()

