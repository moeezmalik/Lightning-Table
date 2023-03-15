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

# Complete Pipeline Imports
from infer import folder_of_pdf_to_csv
from tableextractor import save_to_excel as raw_tables_to_excel
from files import get_list_of_files_with_ext, basename
from finalstep import Datasheet

import argparse
import yaml
import pandas as pd

# Table Classification Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB

class TableClassificationEvaluation():
    """
    This class will perform the evaluation for the task
    Table Classification. The goal of this evaluation is
    not to train all the models from scratch but to recreate
    the results from the best performing model that was
    incorporated into the pipeline.

    Args:
        path_to_folder:
            This is the path to the folder that contains the
            data required for the evaluation of the task of
            table classification. The folder structure must be:
                root:
                    data:
                        This folder should contain the csv files
                        that contains labelled keywords that are
                        extracted from the tables. The name of the
                        CSV file should be 'table-data.csv'.
    """

    def __init__(
            self,
            path_to_folder: str
            ) -> None:
        
        self.path_to_data_file = path_to_folder + "data/table-data.csv"

        # This will store the raw data frame that contains the keywords
        # from all the tables and their labels
        self.data_df = None

        # These are the vectorised keywords taken from the raw df above
        # along with their labels
        self.X_tfidf = None
        self.y = None

        self.avg_reports = None

    # EXTERNAL FUNCTIONS
    # Functions that are can called using the class object

    def evaluate(self) -> None:
        """
        This is the main function that will perform all of the evaluations
        for this task. It will call other internal functions for the purpose.
        """
        
        self.load_data()
        self._vectoriser_keywords()
        self._perform_kfold()
        self.show_results()

    # INTERNAL FUNCTIONS
    # Functions that are meant to be used inside the class
    
    def load_data(self) -> None:
        """
        This function will load the data file into class variable.
        """

        data_df = pd.read_csv(self.path_to_data_file, names=['keywords', 'class'])
        data_df.fillna(value="", inplace=True)

        self.data_df = data_df

    def _vectoriser_keywords(self) -> None:
        """
        This function will vectorise the keywords from the table
        using the TF-IDF vectoriser.
        """

        tf = TfidfVectorizer(max_features=5000)

        X = self.data_df['keywords']

        self.X_tfidf = tf.fit_transform(X)
        self.y = self.data_df['class']

    def _average_clf_reports(
            self,
            clf_reports: list
        ) -> dict:
        """
        This function will average out the list of classification
        reports that are provided to it.
        """

        d_precision = 0.0
        e_precision = 0.0
        t_precision = 0.0
        o_precision = 0.0

        d_recall = 0.0
        e_recall = 0.0
        t_recall = 0.0
        o_recall = 0.0

        d_f1 = 0.0
        e_f1 = 0.0
        t_f1 = 0.0
        o_f1 = 0.0

        accuracy = 0.0


        for report in clf_reports:

            d_precision = d_precision + report.get("d").get("precision")
            e_precision = e_precision + report.get("e").get("precision")
            t_precision = t_precision + report.get("t").get("precision")
            o_precision = o_precision + report.get("o").get("precision")

            d_recall = d_recall + report.get("d").get("recall")
            e_recall = e_recall + report.get("e").get("recall")
            t_recall = t_recall + report.get("t").get("recall")
            o_recall = o_recall + report.get("o").get("recall")

            d_f1 = d_f1 + report.get("d").get("f1-score")
            e_f1 = e_f1 + report.get("e").get("f1-score")
            t_f1 = t_f1 + report.get("t").get("f1-score")
            o_f1 = o_f1 + report.get("o").get("f1-score")

            accuracy = accuracy + report.get("accuracy")

        total_reports = len(clf_reports)

        d_precision = d_precision / total_reports
        e_precision = e_precision / total_reports
        t_precision = t_precision / total_reports
        o_precision = o_precision / total_reports
        d_recall = d_recall / total_reports
        e_recall = e_recall / total_reports
        t_recall = t_recall / total_reports
        o_recall = o_recall / total_reports
        d_f1 = d_f1 / total_reports
        e_f1 = e_f1 / total_reports
        t_f1 = t_f1 / total_reports
        o_f1 = o_f1 / total_reports
        accuracy = accuracy / total_reports

        avgd_output = {
            "d" : {
                "precision" : d_precision,
                "recall" : d_recall,
                "f1" : d_f1
            },
            "e" : {
                "precision" : e_precision,
                "recall" : e_recall,
                "f1" : e_f1
            },
            "t" : {
                "precision" : t_precision,
                "recall" : t_recall,
                "f1" : t_f1
            },
            "o" : {
                "precision" : o_precision,
                "recall" : o_recall,
                "f1" : o_f1
            },
            "accuracy" : accuracy
        }

        return avgd_output
    
    def _perform_kfold(self):
        """
        This function will perform the 5-Fold Cross Validation with the
        words vectorised using TF-IDF and Naive Bayes as the classifier.
        The results from the 5 folds will be averaged out and saved
        into the class variable.
        """

        kf = KFold(n_splits=5, random_state=None, shuffle=False)
        nb = MultinomialNB()

        nb_tfidf_folds = []

        for i, (train_index, test_index) in enumerate(kf.split(self.X_tfidf)):
            
            X_train = self.X_tfidf[train_index]
            X_test = self.X_tfidf[test_index]
            y_train = self.y[train_index]
            y_test = self.y[test_index]

            # Naive Bayes with TF-IDF
            nb.fit(X_train, y_train)
            nb_pred = nb.predict(X_test)
            nb_tfidf_currfold = classification_report(y_test, nb_pred, output_dict=True)
            nb_tfidf_folds.append(nb_tfidf_currfold)

        self.avg_reports = self._average_clf_reports(nb_tfidf_folds)

    def show_results(self):
        """
        This function will prettify the results and show them on the
        command line in a nice way.
        """

        results_list = []

        # For Electrical Class
        label = "Electrical Characteristics"
        precision = self.avg_reports.get("e").get("precision")
        recall = self.avg_reports.get("e").get("recall")
        f1 = self.avg_reports.get("e").get("f1")

        results_list.append([label, precision, recall, f1])

        # For Thermal Class
        label = "Thermal Characteristics"
        precision = self.avg_reports.get("t").get("precision")
        recall = self.avg_reports.get("t").get("recall")
        f1 = self.avg_reports.get("t").get("f1")

        results_list.append([label, precision, recall, f1])

        # For Mechanical Class
        label = "Mechanical Characteristics"
        precision = self.avg_reports.get("d").get("precision")
        recall = self.avg_reports.get("d").get("recall")
        f1 = self.avg_reports.get("d").get("f1")

        results_list.append([label, precision, recall, f1])

        # For Other Class
        label = "Other"
        precision = self.avg_reports.get("o").get("precision")
        recall = self.avg_reports.get("o").get("recall")
        f1 = self.avg_reports.get("o").get("f1")

        results_list.append([label, precision, recall, f1])

        # Add the gap for clarity
        results_list.append(["--------", "--------", "--------", "--------"])

        # For overall accuracy
        label = "Accuracy"
        accuracy = self.avg_reports.get("accuracy")
        results_list.append([label, "", "", accuracy])

        results_df = pd.DataFrame(
            results_list,
            columns=["Class", "Precision", "Recall", "F1"]
            )
        
        results_df = results_df.set_index(["Class"])
        
        print()
        print("Results")
        print("----------------")
        print(results_df)





class CompletePipelineEvaluation():
    """
    This class will perform the evaluation for the full pipeline.
    It expects to be provided the path to the PDF folder from which
    first the tables will be detected. The raw values from these detected
    tables will then be extracted. Finally, the tables will be classified
    from the raw values and the structured values will be extracted. These
    structured values will then be compared against the ground truth values
    that also need to be specified.

    Args:
        path_to_folder:
            This is the path to the folder that contains the models and the
            data for performing the evaluations. The folder structure
            should as following:
                root:
                    models:
                        This folder must contain the deep learning model
                        weights for FasterRCNN, the Naive Bayes classifier
                        and the TF-IDF word vectoriser pickle file and the
                        yaml file that contains the patterns for the final
                        step.
                    pdfs:
                        This folder must contain the PDF files that will be
                        fed for evaluation to the pipeline.
                    gt:
                        This folder must contain the yaml files that contain
                        the ground-truth values the PDF files in the pdfs
                        folder. The names of the yaml files must match those
                        of the PDFs.
                    excels:
                        Ideally, this folder should be empty. This is where
                        the intermediate files will be stored by the pipeline.
    """

    # EXTERNAL FUNCTIONS
    # Functions that are can called using the class object

    def __init__(
            self,
            path_to_folder: str,
        ) -> None:

        # Assign the class variables
        self.path_to_pdf_folder = path_to_folder + "pdfs/"
        self.path_to_excel_folder = path_to_folder + "excels/"
        self.path_to_gt_folder = path_to_folder + "gt/"
        self.path_to_models_folder = path_to_folder + "models/"

        self.extracted_values = None
        self.confusion_matrix = None
        self.metrics = None
    
    def evaluate(self):
        """
        This function will call all the necessary functions required
        for the evaluation procedure.
        """
        
        # First detect all the tables and save to a CSV file
        self.detect_tables()

        # Then extract the raw tables and save to an excel file
        self.recognise_tables()

        # Structure the values and extract the relevant ones
        self.perform_final_step()

        # Compare the extracted values to the ground truth values
        self.compare_values()

        # Generate results from the comparisons performed
        self.generate_results()

        # Show the generated results
        self.show_results()

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
            output_folder=self.path_to_excel_folder + "baseline/",
            reader="baseline"
        )

        raw_tables_to_excel(
            path_to_csv=self.path_to_excel_folder + "detectedtables.csv",
            pdf_folder=self.path_to_pdf_folder,
            output_folder=self.path_to_excel_folder + "camelot/",
            reader="camelot"
        )

        raw_tables_to_excel(
            path_to_csv=self.path_to_excel_folder + "detectedtables.csv",
            pdf_folder=self.path_to_pdf_folder,
            output_folder=self.path_to_excel_folder + "tabula/",
            reader="tabula"
        )

    def perform_final_step(self):
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

        baseline_all_files = self._fs_folder(path_to_folder=path_to_baseline)
        camelot_all_files = self._fs_folder(path_to_folder=path_to_camelot)
        tabula_all_files = self._fs_folder(path_to_folder=path_to_tabula)

        #print(baseline_all_files)
        #print(camelot_all_files)
        #print(camelot_all_files)

        self.extracted_values = {
            "baseline" : baseline_all_files,
            "camelot" : camelot_all_files,
            "tabula" : tabula_all_files
        }
    
    def compare_values(self):
        """
        This function will compare the extracted values with the ground
        truth ones and generate the final results.
        """

        self.confusion_matrix = {}

        for folder, files in self.extracted_values.items():

            # Calculate the confusion matrix values for the
            # whole folder
            folder_tp, folder_fp, folder_fn = self._compare_folder(
                foldername=folder,
                files=files
            )

            # Put the confusion matrix values in the class
            # dictionary.
            folder_cm = {
                "tp" : folder_tp,
                "fp" : folder_fp,
                "fn" : folder_fn
            }

            self.confusion_matrix[folder] = folder_cm

    def generate_results(self):
        """
        This function will generate the results from the confusion
        matrix that was calcaulted by comparing the extracted values
        with the ground truth values.
        """

        self.metrics = {}
        
        for folder, cm in self.confusion_matrix.items():

            folder_metrics = self._compute_metrics(
                cm_dict=cm
            )

            self.metrics[folder] = folder_metrics

    def show_results(self):
        
        print()
        print("Results")
        print("----------------")
        
        results = []

        for folder, metrics in self.metrics.items():

            folder_result = [
                folder,
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1")
            ]

            results.append(folder_result)

        results_df = pd.DataFrame(
            data=results,
            columns=["", "Precision", "Recall", "F1-Score"]
        )

        results_df = results_df.set_index([""])

        print(results_df)
            


    # INTERNAL FUNCTIONS
    # Functions that are meant to be used inside the class

    def _compute_metrics(
            self,
            cm_dict: dict
        ) -> dict:
        """
        This function will compute the metrics i.e. precision, recall
        and the f1-score from the confusion matrix provided as a
        dictionary.

        Returns:
            metric:
                Dictionary type object that contains the three metrics
        """

        tp = cm_dict.get("tp")
        fp = cm_dict.get("fp")
        fn = cm_dict.get("fn")

        try:
            precision = (tp / (tp + fp))
        except ZeroDivisionError:
            precision = None

        try:
            recall = (tp / (tp + fn))
        except ZeroDivisionError:
            recall = None

        try:
            f1 = (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = None
        except TypeError:
            f1 = None

        return {
            "precision" : precision,
            "recall" : recall,
            "f1" : f1
        }


    def _compare_folder(
            self,
            foldername: str,
            files: dict
        ):
        """
        This function will compare the values extracted from files
        of a particular folder i.e. baseline, camelot and tabula
        with the ground truth values.
        """

        folder_tp = 0
        folder_fp = 0
        folder_fn = 0

        for file, properties in files.items():

            file_tp, file_fp, file_fn = self._compare_file(
                filename=file,
                vals=properties
            )

            folder_tp += file_tp
            folder_fp += file_fp
            folder_fn += file_fn

        return folder_tp, folder_fp, folder_fn

    def _compare_file(
            self,
            filename: str,
            vals: dict
        ):
        """
        This function will compare the values extracted from a single
        file with its counterpart in ground truth.
        """

        # Load the ground    truth values for the current file
        with open(self.path_to_gt_folder + filename + ".yml", "r") as stream:
            try:
                gt_vals = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)

        # Calcaulate the confusion matrix for the whole file

        file_tp = 0
        file_fp = 0
        file_fn = 0

        for prop_type, prop in gt_vals.items():

            for prop, gt_vals in prop.items():
                
                # If the current property was not detected by
                # the pipeline then replace with an empty list
                try:
                    predicted_vals = vals.get(prop_type)
                    predicted_vals = predicted_vals.get(prop)

                except AttributeError:
                    predicted_vals = []

                prop_tp, prop_fp, prop_fn = self._two_list_confusion_matrix(
                    true=gt_vals,
                    preds=predicted_vals
                )

                file_tp += prop_tp
                file_fp += prop_fp
                file_fn += prop_fn

        return file_tp, file_fp, file_fn

    def _two_list_confusion_matrix(
            self,
            true: list,
            preds: list
        ) -> tuple:

        #print(true)
        #print(preds)

        if true is None:
            true = []

        if preds is None:
            preds = []

        # OLD METHODS
        #intersection_count = len(set(preds) & set(true))

        #tp = intersection_count
        #fp = len(preds) - intersection_count
        #fn = len(true) - intersection_count

        # NEW METHOD
        tp = 0
        fp = 0
        fn = 0

        for true_item in true:

            fn += 1

            for i, pred_item in enumerate(preds):

                if true_item == pred_item:

                    tp += 1
                    fn -= 1
                    del preds[i]

                    break

        fp = len(preds)


        #print(str((tp, fp, fn)))
        #print()
        #input("Press enter to continue")

        return tp, fp, fn


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

        all_files_extracted = {}

        # Go through all the files one by one and get the
        # values that were extracted
        for file in list_of_files:

            # Just run the function for now
            extracted_vals = self._fs_file(
                path_to_file=file
            )

            filename = str(basename(file)).rsplit(sep=".")[0]

            all_files_extracted[filename] = extracted_vals

        return all_files_extracted

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
        choices=['complete', 'detection', 'classification'],
        required=True,
        metavar="TYPE",
        dest="type",
        help=u"""
        This specifies the type of evaluation that needs to be done. The
        choices are 'complete', 'table_detection' and 'table_classification' 
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

    if type == "complete":
        print()
        print("Evaluating Complete Pipeline")

        cpe = CompletePipelineEvaluation(
            path_to_folder=path
        )

        cpe.evaluate()

    if type == "classification":
        print()
        print("Evaluating Table Classification")

        tpe = TableClassificationEvaluation(
            path_to_folder=path
        )

        tpe.evaluate()

if __name__=="__main__":

    # Execute the main function
    main()

