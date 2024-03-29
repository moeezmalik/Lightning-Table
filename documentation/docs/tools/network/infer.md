# Infer Tool

A command-line tool for generating an output file i.e. CSV for all
the bounding boxes found in all of the pdf files that are provided
as input to this utility.

## Purpose

This tool was developed in order to get the inference results from the
model and save them as a CSV file so that the data can be extracted
from the tables that were detected.

The tool can either take in a single pdf as an input or it can take
folder that contains multiple pdf files as an input.

## How it works

The utility when provided with an appropriate model, the weights for the
model and the an appropriate input pdf file, can detect the tables in the
input pdf file and save the output as a CSV file.

It takes the input file, converts it into an image in case of a pdf file,
then passes that image through the specified model, gets the bounding boxes
with confidence score higher than the threshold provided. It then collects
all the coordinates of the detected tables into a list and saves it in a
CSV file.

An example of such a CSV file is shown below:

!!! note
    The header "filename,pageno,x1,y1,x2,y2" shown below is just for reference and it will not appear in the actual file generated by this tool.

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

The first column is the name of the pdf was that was provided, the second column contains the page number on which the table was found. The column 3 through 6 mention the coordinates of the table that was detected. 'x1' and 'y1' are the coorindates for the top-left corner and 'x2' and 'y2' shows the bottom-right coordinates of the bounding box for the table that was detected.

!!! important
    The x1, y1, x2, y2 coordinates generated will be in the PDF coordinate space. This means that the (0, 0) position of the x and y-axis respectively is at the bottom-left of the page. This is different than what normally is the case where the (0, 0) position is the top-left of the page. This is done because the table extraction utilities e.g. Camelot expect the table regions in PDF coordinate space.

If a page in PDF contains no table, according to the model, then it is not
mentioned at all in the CSV file.

!!! limitation
    At the moment, due to limitations of the table text extraction utility, the PDF file that do not contain any text layer e.g. if they are scanned, will be skipped. Bounding boxes of tables
    on such files cannot be extracted at the moment.

## Parameters

The following parameters can be specified to run the command line utility.

### Required Parameters

- `-t` or `--type`:
The type of file on which to perform the inference. Valid options are 'pdf',
'pdfs_folder', 'image' or 'images_folder'. To visualised a single pdf file
'pdf' should be specified. To visualise a folder of pdf file, 'pdfs_folder'
should be specified. To visualise a single image, 'image' should be specified
and lastly in order to visualise a folder of images, 'images_folder' should
be specified. Please make sure to include the full path to the files, in
case of individual image or pdf, when specifying the path parameter below.

- `-p` or `--path`:
This is the path to the type of file that was specified above. 

- `-m` or `--model`:
This is the name of the model that should be used to generate the bounding
boxes on the images provided. The name of the model should exactly match
the name of the class that implements the model in the models.py file.

- `-w` or `--weights`:
In this parameter, the path to the Pytorch Lightning checkpoint should be
provided that contains the trained weights for the model that was specified
above. Please make sure that the checkpoint provided belongs to the model
that was specified above. You can find the checkpoints in the
[Downloads](../../downloads/index.md) section.

- `-o` or `--output`:
This parameter will specify the path, where to save the CSV file. Please
specify the name of the CSV file alongwith the extension '.csv' when specifying
the folder in which to save the CSV file.

### Optional Parameters

- `-h` or `--help`:
This will show instructions on how to use the utility.

- `-c` or `--confidence`:
This parameter specifies the confidence threshold. It is a floating point
number between 0 and 1. When the model makes predictions, it specfies how
confident it is on those predicitons as well. This parameter will specify
the cutoff, the predictions below this cutoff will not be considered.
This is an optional parameter, if no value is specified then 0.75 will be
taken.

- `-d` or `--dpi`:
In order to detect tables in the PDF file. Each individual page is first
converted to an image that will then be passed through the model. This
parameter specifies the DPI for that rendered image. The higher this number,
the higher the resolution of the rendered image will be but the process of
inference will also will be slower. This parameter also needs to be
selected in accordance with the utility that will extract the textual data
from the table e.g. Camelot. This is because if the resolution is different,
the coordinates of the detected table will also be different.


## Example

An example of how to run the utility is provided below:

```
python infer.py -t pdf -m VanillaRetinaNet -w ../misc/best-chkpnt-epoch=35.ckpt -c 0.80 -p ../misc/601fa1f389733.pdf -o ../misc/output.csv
```