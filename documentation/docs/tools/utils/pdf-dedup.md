# Data House Keeper

This script was created as part of one of the pre-processing steps before the PDFs are passed through the network for inference and detection of tables.

## Purpose

When the PDF files were collected during the data scraping process, a lot of duplicates were created because same files were saved under conflicting namees provided by the website. This utility helps in detecting those duplicates. 

## How it works

When the folder of PDF was analysed, it was found that the duplicate files had the identical file size. Initially the goal was to detect the duplicates only on the basis of the file sizes.

This turns out was not enough to detect the duplicates, since the files from the same manufacturers had the same file sizes even if the content of the PDF files was different. To counter this, the last-modified time of the PDF file was also considered. PDFs that were duplicates had the last-modified time very close to each other i.e. within 5 minutes.

To take that into account, the metadata of the files is read. The last-modified time is rounded off to the closest 5 minute mark. Then duplicates are detected based on the file sizes and the rounded-off last-modified time.

## Parameters

The following parameters can be specified to run the script:

### Required Parameters

- `-p` or `--path`:
This is the path to the folder that contains the PDF files.

- `-a` or `--action`:
Once the PDF files have been detected as duplicates, this parameter will specify the action that will be taken against them. The valid actions are 'noaction', 'delete' and 'move'. If 'noaction' is specified the no action will be taken against the duplicated files. If 'delete' is specified then the duplicated files will be deleted. In case 'move' is specified then a new folder named 'Duplicates' will be created and the files will be moved over to that folder. Please make sure that a folder named 'Duplicates' does not already exist.

### Optional Parameters

- `-h` or `--help`:
This will show instructions on how to use the utility.


## Example

An example on how to run the script is following:

```
python pdf-dedup.py -p ../pdf-files/ -a noaction
```