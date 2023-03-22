# Extract Tables

A utility to get the textual data of tables from the PDF files.

## Purpose

After the tables have been detected by a machine learning model or the regions are indicated by a human. This utility can be used to extract the table content from the PDF files and put them into an excel file.

## How it works

For this utility to work, two major things are required. One is a folder that contains all of the PDF files that need to be evaluated, the other one is the CSV file that contains the regions in the PDF file where the tables are located. 

The CSV file such ideally be generated by the trained Deep Learning model which can be accessed from the [Infer Tool](../network/infer.md). It should generate the CSV in the following format.

```
filename,pageno,x1,y1,x2,y2
601fa1f389734.pdf,1,258,2891,2155,5165
601fa1f389734.pdf,1,244,5426,2107,6218
601fa1f389733.pdf,2,210,3268,3162,4738
601fa1f389733.pdf,2,2079,5861,3125,6346
601fa1f389733.pdf,2,177,5774,2042,6376
601fa1f389733.pdf,2,1899,4921,3177,5538
601fa1f389733.pdf,2,216,4897,1865,5516
601fa1f389733.pdf,2,220,1165,3135,2009
601fa1f389733.pdf,2,246,2317,3114,2998
```

Here, the `filename` is the name of the PDF file that needs to evaluated. The filename should contain the extension i.e. '.pdf'. The `pageno` field specifies the page on which the table exists. The coordinates `x1`, `y1`, `x2` and `y2` are that of the table found. `x1` and `y1` specify the top left corner of the table and `x2` and `y2` specify the bottom corner of the table.

!!! important
    The x1, y1, x2, y2 coordinates mentioned in the file should be in the PDF coordinate space. This means that the (0, 0) position of the x and y-axis respectively is at the bottom-left of the page. This is different than what normally is the case where the (0, 0) position is the top-left of the page. This is done because the table extraction utilities e.g. Camelot expect the table regions in PDF coordinate space.

!!! note
    The header filename,pageno,x1,y1,x2,y2 are just added in the CSV representation here for information. The infer tool will not generate this header. This utility also expects the CSV file without this header, however the order of information should be respected.

The utility will go through the PDF files one by one. Take in all the tables found in that file, extract the textual data in a tabular format and save them as a file. In case an excel file is required, it will generate an excel file for that PDF that will contain all of the tables. Each table will constitute one sheet in the excel document.

## Parameters

Following parameters are required to run the script:

- `-f` or `--folder`:
This is the path to the folder that contains all of the PDF files.

- `-c` or `--csv`:
This is the path to the CSV file that contains information about the table areas in the PDF files. See the section above for more information about this file.

- `-r` or `--reader`:
This parameter specifies the type of reader that will be used to read the tables from the PDF files provided their coordinate locations. Three options are available: 'baseline', 'camelot' and 'tabula'. Baseline is very simple custom rule-based table extractor that is implemented just to provide a comparison with the other utilities. Camelot and Tabula, on the other hand, are more sophisticated Python utilities for extracting the tables.

- `-o` or `--output`:
This is the path to the folder where the generated files will be stored.



## Example

An example on how to run the script is following:

```
python extract-tables.py -f ../misc/pdf-test/ -c ../misc/pdf-test/output6.csv -o ../misc/pdf-test/ -t excel
```