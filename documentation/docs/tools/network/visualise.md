# Visualise Tool

A command-line tool for visualing the outputs of the model given
the trained checkpoint and an input image or a pdf file.

## Purpose

This tool was developed in order to get a lens into what the network
is doing before running the model in production and saving all of the
results in a file.

The purpose of this utility is to generate bounding boxes on a variety
of input file types. This can be very useful for debugging purposes to
see how well the network is detecting the tables.

The goal was to create tool that can take a diverse range of input files,
to that effect, this utility can visualise images, all of the images in a
folder, pdf files and all of the pdf files in any given folder.

## How it works

The utility when provided with an appropriate model, the weights for the
model and the an appropriate input file, can show the user what the
model is detecting by drawing bounding boxes on the detected tables.

It takes the input file, converts it into an image in case of a pdf file,
then passes that image through the specified model, gets the bounding boxes
with confidence score higher than the threshold provided. It then draws
the selected bounding boxes on the image and shows it on the screen.

Each image will be shown one-by-one. Enter key needs to be pressed to go
to the next image.

!!! limitation
    At the moment, due to limitations of the table text extraction utility, the PDF file that do not contain any text layer e.g. if they are scanned, will be skipped. Such files cannot be
    visualised with this tool.

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
that was specified above.You can find the checkpoints in the
[Downloads](../../downloads/index.md) section.

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


## Example

An example of how to run the utility is provided below:

```
python visualise.py -t pdf -m VanillaRetinaNet -w ../misc/best-chkpnt-epoch=35.ckpt -c 0.80 -p ../misc/601fa1f389733.pdf
```