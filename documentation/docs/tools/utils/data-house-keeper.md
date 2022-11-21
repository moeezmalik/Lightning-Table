# Data House Keeper

This script was written in order to aid the process of labelling data. At the time of writing this, the labelling tool of choice was LabelImg. It is a python utility that provides a nice GUI for labelling the images.

One of the limitations of labelimg is that it does not support the labeling data in the CSV format which is required by the training scripts that were developed.

In order to deal with the inconsistencies between the two formats, some scripts had to be written to keep everything clean and tidy. This is one of those scripts.

## Purpose

Data House Keeper helps in deleting extra files from a PascalVOC folder i.e. Extra XML and JPG files. It also can rename the files from the number specified so that all the XML and JPG files can be organised.

!!! limitation
    The image files can only be in the JPG format.

!!! limitation
    The name of the both counterparts (the image and its annotation XML) should have the same name for this script to work. This is the case for the files produced by labelimg, so if you're using that tool, it should be fine.

## How it works

- Once the script is started, it will go through the folder and find all the JPGs that have their XML files already present, it will match the names for doing so.

- If there are extra files present i.e. JPG files that have no matching XML files and vice versa, it will then ask if you want to delete the extra files, if the extra files are deleted only then it will proceed to the next step otherwise quit.

- If the extra files have been cleaned up (or there are no extra files) the script will ask if you want to rename the files, if you say yes then it will ask a number from which to start the numbering, this can be useful if you already have images in another folder and you just want to add more labelled data.

## Parameters

Following parameters are required to run the script:

- `pascalvoc_path`: This is the path to folder that contains the PascalVOC format annotations.

!!! note
    The images and their corresponding XML annotation files should be in the same folder, the path to which is provided above.

## Example

An example on how to run the script is following:

```
python utils/data-house-keeper.py --pascalvoc_path "path/to/folder"
```