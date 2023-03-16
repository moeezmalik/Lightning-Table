"""
This file will contain a collection of functions and classes that
will be used to bring structure to the raw data extracted from the
PDF files.

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
import numpy as np
import re
import pickle

from statistics import mode, StatisticsError

# Other libraries

from files import basename

# #################################################################
# Classes and Functions
# #################################################################

class Table():
    """
    This class structure will hold values and information for the
    individual tables that were extracted from the datasheets.

    Parameters:
        name:
            This is the name of the table as found in the saved file.

        raw_df:
            This is the table that is read from the excel file in the
            pandas dataframe format.

        path_to_clf:
            This is the path to the pickle file of the scikit learn
            classifier that will be used to perform the classification
        
        path_to_vec:
            This is the path to the pickle file of the sckikit learn
            word vectoriser that will be used to vectorise the keywords
            of the table

        verbose:
            This flag is used for enabling or disabling the printing
            of useful debug information, if any.
    """
    def __init__(
        self,
        name: str,
        raw_df: pd.DataFrame = None,
        path_to_clf = None,
        path_to_vec = None,
        verbose: bool = False
        ) -> None:
        
        # Class variables
        self.name = None
        self.raw_df = None
        self.low = None
        self.pred_class = None

        self.verbose = verbose
        
        # Set the raw dataframe if available
        self.set_raw_df(
            name=name,
            raw_df=raw_df
            )

        # Create list of words
        if self.raw_df is not None:
            self.prepare_low()

        # Classify the table
        if path_to_clf is not None and path_to_vec is not None:
            self.classify(
                path_to_clf=path_to_clf,
                path_to_vec=path_to_vec
            )

    def set_raw_df(
        self,
        name: str,
        raw_df: pd.DataFrame
        ) -> None:
        """
        This function will load the specified dataframe as the class raw
        dataframe. It will also perform some preparatory operations.

        Parameters:
            name:
                This is the name of the table

            raw_df:
                This is the Dataframe that contains the table information
        """

        self.name = name
        self.raw_df = raw_df

        # Replace NaN values with nothing
        self.raw_df.fillna(
            value='',
            inplace=True
            )

    def prepare_low(
        self
        ) -> None:
        """
        This function will convert the values found in the raw dataframe to
        list of words. It will also perform clean up operations that will remove
        unnecessary characters.
        """

        # Convert the dataframe to a list
        raw_list = self.raw_df.values.tolist()

        # Flatten the list
        flat_list = [item for sublist in raw_list for item in sublist]

        def cleanup(keywords_list):
            
            individual = []

            # Evaluate each element in the flat list
            for keyword in keywords_list:
                
                # Cast the key
                string = str(keyword)

                # Remove brackets
                no_bracs = string.replace('(', ' ')
                no_bracs = no_bracs.replace(')', ' ')
                no_bracs = no_bracs.replace('[', ' ')
                no_bracs = no_bracs.replace(']', ' ')

                # Remove Non-ASCII characters
                noascii = ''.join(char for char in no_bracs if ord(char) < 128)

                # Strip the leading and trailing spaces
                stripped = noascii.strip()

                # Split the string on spaces
                split = stripped.split(" ")

                # Remove digits
                for subsplit in split:
                    # result = ''.join(i for i in subsplit if not i.isdigit())
                    result = re.sub('\W+','', subsplit)
                    result = re.sub('\d+', '', result)

                    if result:
                        individual.append(result)

            return individual

        self.low = cleanup(keywords_list=flat_list)

    def classify(
        self,
        path_to_clf: str,
        path_to_vec: str
        ) -> None:
        """
        This function will classify the table based on the keywords that
        it contains. The predicted class will be saved in the class variable
        "pred_class"

        Parameters:
            path_to_clf:
                This is the path to the pickle file of the scikit learn
                classifier that will be used to perform the classification
            
            path_to_vec:
                This is the path to the pickle file of the sckikit learn
                word vectoriser that will be used to vectorise the keywords
                of the table
        """

        # Create a string from the keywords
        sow = [' '.join(self.low)]

        clf = pickle.load(open(path_to_clf, "rb"))
        vec = pickle.load(open(path_to_vec, "rb"))

        tow = vec.transform(sow)
        self.pred_class = clf.predict(tow)[0]

class Datasheet():
    """
    This class structure will hold the raw data and the extracted
    information from the datasheets.

    Parameters:
        path_to_excel_file:
            This is the path to the excel file that contains the raw
            tables that are extracted by for example Camelot.
        
        path_to_clf:
            This is the path to the pickle file of the scikit learn
            classifier that will be used to perform the classification.
        
        path_to_vec:
            This is the path to the pickle file of the sckikit learn
            word vectoriser that will be used to vectorise the keywords
            of the table.
    """

    def __init__(
        self,
        path_to_excel = None,
        path_to_clf = None,
        path_to_vec = None
        ) -> None:

        # Class Variables

        # This is the name of the datasheet which is the same as
        # the filename
        self.name = None

        # This list will hold all the tables found in the raw format
        self.tables = []

        # Dictionary type for extracted electrical properties
        self.extracted_elec = None

        # Dictionary type for extracted temperature coefficients
        self.extracted_temp = None

        # Calling the constructor functions

        if path_to_excel is not None:
            self.load_tables_from_excel(
                path_to_excel=path_to_excel,
                path_to_clf=path_to_clf,
                path_to_vec=path_to_vec
            )

    def load_tables_from_excel(
        self,
        path_to_excel: str,
        path_to_clf: str = None,
        path_to_vec: str = None
        ) -> None:
        """
        This function will load the raw tables from the excel files

        Parameters:
            path_to_excel:
                This is the path to the excel file that contains the
                raw tables. There should be only one table per sheet.
            
            path_to_clf:
                This is the path to the pickle file of the scikit learn
                classifier that will be used to perform the classification
            
            path_to_vec:
                This is the path to the pickle file of the sckikit learn
                word vectoriser that will be used to vectorise the keywords
                of the table
        """

        # Clear the tables
        self.tables = []

        # Set the name for the datasheet
        self.name = basename(path_to_excel)

        # Read all the sheets in the excel file and get the dictionary
        # of tables
        df = pd.read_excel(
            io=path_to_excel,
            sheet_name=None,
            header=None
            )
            
        # Save the raw tables in the Table class format
        for key, value in df.items():
            self.tables.append(
                Table(
                    name=key,
                    raw_df=value,
                    path_to_clf=path_to_clf,
                    path_to_vec=path_to_vec
                    )
            )

    def print_tables(self) -> None:
        """
        This function is used to show all the tables that were read
        from the saved files.
        """

        print("Datasheet: " + self.name)
        print()

        for table in self.tables:
            print("Table: " + table.name)
            print(table.raw_df)
            print()

    def extract_electrical_props(
        self,
        patterns: dict
        ) -> None:
        """
        This function will attempt to extract the electrical properties
        from the raw tables that are the present in this datasheet. This is
        the master function that will make use of other internal class
        functions in order to keep things organised.

        Parameters:
            patterns:
                Theses are regex patterns that will be used to detect the
                right columns or rows that contain the required electrical
                properties.
        """

        # Get the table that contains the electrical properties in the
        # datasheet

        elec_table = []

        for table in self.tables:
            if table.pred_class == 'e':
                elec_table.append(table)

        if len(elec_table) == 0:
            elec_table = None
        elif len(elec_table) == 1:
            elec_table = elec_table[0]
        else:
            max_len = 0
            selected_table = None

            for table in elec_table:

                curr_len = len(table.raw_df.stack().tolist())
                
                if(curr_len) > max_len:
                    max_len = curr_len
                    selected_table = table

            elec_table = selected_table

        # If no electrical table is found, go no further
        if elec_table is None:
            print("Error: No Electrical Characteristics Table Found")
            return None

        # Try determining the table axis and do not continue if it couldnt
        # be determined for any reason
        axis = None

        try:
            axis = self._determine_table_axis(
                table=elec_table
            )
        except:
            print("Error: Exception occured during table axis detection")

        if axis is None:
            print("Error: Cannot Determine Table Axis, Cannot Continue")
            return None

        print("Table Axis: " + axis)

        if axis == "vertical":

            # Try to find the columns in the table that contain the properties
            # specified in the patterns dictionary
            found_columns = self._find_columns(
                table=elec_table,
                patterns=patterns
            )

            # Get the actual values from the classified columns
            found_vals = self._extract_values_from_columns(
                table=elec_table,
                columns=found_columns,
                pattern=patterns
            )

            self.extracted_elec = found_vals

        elif axis == "horizontal":

            found_rows = self._find_rows(
                table=elec_table,
                patterns=patterns
            )

            found_vals = self._extract_values_from_rows(
                table=elec_table,
                rows=found_rows,
                pattern=patterns
            )

            self.extracted_elec = found_vals
        
    def extract_temp_props(
        self,
        patterns: dict
        ) -> None:
        """
        This function will attempt to extract the temperature coefficients
        from the raw tables that are the present in this datasheet. This is
        the master function that will make use of other internal class
        functions in order to keep things organised.

        Parameters:
            patterns:
                Theses are regex patterns that will be used to detect the
                right columns or rows that contain the required electrical
                properties.
        """

        # Get the table that contains the electrical properties in the
        # datasheet
        temp_table = None

        for table in self.tables:
            if table.pred_class == 't':
                temp_table = table

        # If no electrical table is found, go no further
        if temp_table is None:
            print("Error: No Temperature Coefficient Table Found")
            return None

        # print(temp_table.raw_df)

        # Try determining the table axis and do not continue if it couldnt
        # be determined for any reason
        axis = None

        try:
            axis = self._determine_table_axis(
                table=temp_table
            )
        except:
            print("Error: Exception occured during table axis detection")

        if axis is None:
            print("Error: Cannot Determine Table Axis, Cannot Continue")
            return None

        if axis == 'vertical':
            print("Cannot process vertical tables yet")
            return None

        # Find the rows that represent required values according to the
        # patterns dictionary
        found_rows = self._find_rows(
            table=temp_table,
            patterns=patterns
        )

        # Get the actual values from the classified rows
        found_vals = self._extract_values_from_rows(
            table=temp_table,
            rows=found_rows,
            pattern=patterns
        )

        self.extracted_temp =  found_vals

        return None

        
    
    ##################################
    # Internal functions for the class
    ##################################

    def _determine_table_axis(
        self,
        table: Table
        ) -> str:
        """
        This function will attempt to determine the dominant axis of the table.
        What this essentially means is that, along the determined axis, values
        of the same category will be found.

        Parameters:
            table:
                This is the table of which the dominant axis needs to be found

        Returns:
            result:
                'vertical', 'horizontal' or None.
                None will be returned when there was an error in determining
                the axis.
        """

        # Convert the DataFrame to a 2D list so make the iterations more
        # Pythonic

        table_as_list = table.raw_df.values.tolist()

        # Create an empty list that will contain the raw values on which
        # the decision of either vertical or horizontal will be made.
        table_of_extracted_values = []

        # Iterate over the rows in the table
        for row in table_as_list:
            
            # This empty list will hold all the values that are extracted
            # from the current row
            extracted_row = []

            # Iterate over all columns in the current row
            for col in row:
                
                match = re.search(
                    pattern="\d+.\d+|\d+",
                    string=str(col)
                )

                if match is not None:

                    col = match.group(0)

                    # The following try-except clauses are used to determine if
                    # the current element is a number or not.
                    try:
                        curr_val = float(col)
                        extracted_row.append(curr_val)

                    except ValueError as e:
                        pass
                
            # If something is extracted then add it to the main matrix
            if len(extracted_row) != 0:
                table_of_extracted_values.append(extracted_row)

        # If no values were extracted then return nothing
        if not table_of_extracted_values:
            return None

        # Calculate the mode of lengths which will be the most common length
        # of values found. This can then be used to remove misidentified values
        extracted_values_lens = [len(x) for x in table_of_extracted_values]

        try:
            m = mode(extracted_values_lens)

        except StatisticsError as e:
            return None

        mode_table_of_extracted_values = []

        for item in table_of_extracted_values:

            if len(item) == m:
                mode_table_of_extracted_values.append(item)

        table_of_extracted_values = mode_table_of_extracted_values

        # Determine if the extracted list of lists is a valid matrix, to do
        # that the lengths of all the rows are computed. If all the lengths
        # of the rows are equal then we have valid matrix otherwise not.
        is_valid_matrix = all([len(x)==len(table_of_extracted_values[0]) for x in table_of_extracted_values])

        # Exit with None if the matrix is not valid.
        if not is_valid_matrix:
            return None

        table_of_extracted_values = np.asarray(table_of_extracted_values)

        # print(table_of_extracted_values)

        # Calculate the variance for the extracted matrix in both the horizontal
        # and the vertical directions. The direction of least variation will be
        # the direction of the table
        vertical_var = np.sum(np.var(table_of_extracted_values, axis=1))
        horizontal_var = np.sum(np.var(table_of_extracted_values, axis=0))

        if horizontal_var > vertical_var:
            return 'horizontal'
        else:
            return 'vertical'

    # Functions for columns

    def _extract_values_from_columns(
        self,
        table: Table,
        columns: dict,
        pattern: dict
        ) -> None:
        """
        This function will iterate over the provided column in the provided table
        and extract the values using the provided pattern and put them in a list.

        Parameters:
            table:
                This is the table which contains the column from which to extract
                the values from.

            pattern:
                This is the regex pattern that will be used to match the values in
                order to extract them.
        """

        found_vals = {}

        for name, info in columns.items():
            
            vals = []

            cols = info.get("cols")
            
            if cols is None:
                found_vals.update({name:{'cols': cols, 'vals' : None}})

            else:

                for col in cols:
                    
                    raw = table.raw_df.iloc[:, col].values.tolist()

                    for element in raw:
                        result = re.search(
                            pattern=pattern.get(name).get("vals"),
                            string=str(element)
                        )

                        if result is not None:
                            vals.append(element)

                found_vals.update({name:{'cols': cols, 'vals' : vals}})

        return found_vals
 
    def _check_col_for_pattern(
        self,
        table: Table,
        col: int,
        pattern: str
        ) -> bool:
        """
        This function checks the provided column in the provided table
        for the presence of the regex pattern.

        Parameters:
            pattern:
                This is the regex pattern to check

            col:
                This is the column in the table to check

            table:
                This is the table in which the column to check exists

        Returns:
            result:
                A boolean type result that is True if the column contains the requireed
                pattern otherwise False.
        """


        # Check each row of the column if the pattern exists
        row_check = table.raw_df.iloc[:, col].str.contains(
            pattern,
            flags=re.IGNORECASE,
            regex=True,
            na=False
            )

        # Return True if the pattern exists in any of the rows or False
        # if it doesnt.
        return any(row_check)

    def _find_columns(
        self,
        table: Table,
        patterns: dict
        ) -> dict:
        """
        This function will attempt to find the required columns in the table.

        Parameters:
            table:
                This is the table that contains the required columns.

            patterns:
                These are regex patterns that will be used to find the columns in
                the table. These patterns will also govern the number of columns
                of information that is extracted.

        Returns:
            cols:
                This is a dictionary type variable that contains the list of columns
                that have been found for each item in the provided pattern list.
        """

        found_cols = {}

        # Get the number of columns of the table
        n_cols = table.raw_df.shape[1]

        for name, pattern in patterns.items():
            
            col_list = []

            for i in range(0, n_cols):
                if self._check_col_for_pattern(
                    table=table,
                    col=i,
                    pattern=pattern.get("series")
                ):
                    col_list.append(i)

            if col_list:
                found_cols.update({name : {'cols' : col_list}})
            else:
                found_cols.update({name : {'cols' : None}})

        return found_cols

    # Functions for rows

    def _extract_values_from_rows(
        self,
        table: Table,
        rows: dict,
        pattern: dict
        ) -> dict:
        """
        This function will iterate over the provided column in the provided table
        and extract the values using the provided pattern and put them in a list.

        Parameters:
            table:
                This is the table which contains the column from which to extract
                the values from.

            rows:
                This is a dictionary that contains the names of the rows and where
                the rows are located from which the information is to be extracted.

            pattern:
                This is the regex pattern that will be used to match the values in
                order to extract them.

        Returns:
            values:
                This is a dictionary that will contain the values that are extracted
                from each row.
        """

        values = {}

        for name, info in rows.items():
            
            vals = []

            rows = info.get("rows")
            
            if rows is None:
                values.update({name:{'rows': rows, 'vals' : None}})

            else:

                for row in rows:
                    
                    raw = table.raw_df.iloc[row, :].values.tolist()

                    for element in raw:
                        result = re.search(
                            pattern=pattern.get(name).get("vals"),
                            string=str(element)
                        )

                        if result is not None:
                            
                            # Only keep the matched entity
                            vals.append(result.group(0))

                values.update({name:{'rows': rows, 'vals' : vals}})


        return values

    def _check_row_for_pattern(
        self,
        table: Table,
        row: int,
        pattern: str
        ) -> bool:
        """
        This function checks the provided row in the provided table
        for the presence of the regex pattern.

        Parameters:
            pattern:
                This is the regex pattern to check

            row:
                This is the row in the table to check

            table:
                This is the table in which the row to check exists

        Returns:
            result:
                A boolean type result that is True if the row contains the required
                pattern otherwise False.
        """

        # Check each row of the row if the pattern exists
        row_check = table.raw_df.iloc[row, :].str.contains(
            pattern,
            flags=re.IGNORECASE,
            regex=True,
            na=False
            )

        # Return True if the pattern exists in any of the rows or False
        # if it doesnt.
        return any(row_check)

    def _find_rows(
        self,
        table: Table,
        patterns: dict
        ) -> dict:
        """
        This function will attempt to find the required rows in the table.

        Parameters:
            table:
                This is the table that contains the required rows.

            patterns:
                These are regex patterns that will be used to find the rows in
                the table. These patterns will also govern the number of rows
                of information that is extracted.

        Returns:
            rows:
                This would be a dictionary that will indicate the found rows according
                to the input patterns
        """
        
        found_rows = {}

        # Get the number of rows of the table
        n_rows = table.raw_df.shape[0]

        for name, pattern in patterns.items():
            
            row_list = []

            for i in range(0, n_rows):
                if self._check_row_for_pattern(
                    table=table,
                    row=i,
                    pattern=pattern.get("series")
                ):
                    row_list.append(i)

            if row_list:
                found_rows.update({name : {'rows' : row_list}})
            else:
                found_rows.update({name : {'rows' : None}})

        return found_rows



def main():
    """
    This is the main function. This will call all other functions in
    this utility.
    """

    path_to_excel = "/Volumes/T7/thesis-data/cells/excel/1-1.xlsx"

    datasheet = Datasheet()

    datasheet.load_tables_from_excel(path_to_excel)
    datasheet.print_tables()

if __name__=="__main__":

    main()