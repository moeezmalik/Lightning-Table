"""
This file implements the functions for testing the code
and the models to ensure correctness.

Author:
    Name:
        Muhammad Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

from extract import FolderExtraction
import pandas as pd


def test_all():
    """
    This function will perform the extraction operations and then
    compare the outputs.
    """

    # The paths are hard-coded right now
    fe = FolderExtraction(
        path_to_folder="../test-data/pdfs/",
        path_to_models="../test-data/models/"
    )

    # Perform the extraction
    fe.extract()

    fe.save_to_excel(
        path="../test-data/comparisons/extd.xlsx"
    )

    extd_df = pd.read_excel("../test-data/comparisons/extd.xlsx")
    gt_extd_df = pd.read_excel("../test-data/comparisons/gt_extd.xlsx")


    pd.testing.assert_frame_equal(extd_df, gt_extd_df)