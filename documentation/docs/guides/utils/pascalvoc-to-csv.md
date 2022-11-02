# PascalVOC to CSV

This script was written in order to aid the process of labelling data. At the time of writing this, the labelling tool of choice was LabelImg. It is a python utility that provides a nice GUI for labelling the images.

One of the limitations of labelimg is that it does not support the labeling data in the CSV format which is required by the training scripts that were developed.

In order to deal with the inconsistencies between the two formats, some scripts had to be written to keep everything clean and tidy. This is one of those scripts.

## Purpose

This script converts the image annotations/labellings from the PascalVOC (XML) format to CSV and writes them as a CSV file.

!!! warning
    The script only reads the XML annotation files and converts them to a single CSV file for saving. It is essential that before running this script, that the folder has been cleaned up using the [Data House Keeper](data-house-keeper.md) script. 

## How it works

- It will go through each XML file in the folder and get the respective labelled objects. 

!!! warning
    Since this script was purpose built for the table detection algorithm, it will set the labels of all the annotations to the same thing i.e. "Table". This script is not made for the case of multiple objects.

- It will assemble one big Pandas DataFrame that will contain the annotations from all of the XML files present.

- Finally, it will ask if you want to save the CSV file, if you say yes then it will ask for the name of the file. Please enter a valid name i.e. not empty to save the file. Do not enter the extension i.e. ".csv" in the filename, that is understood.

## Required Parameters

Following parameters are required to run the script:

- `pascalvoc_path`: This is the path to folder that contains the PascalVOC format annotations.

!!! note
    The images and their corresponding XML annotation files should be in the same folder, the path to which is provided above.

## Example

An example on how to run the script is following:

```
python utils/data-house-keeper.py --pascalvoc_path "path/to/folder"
```