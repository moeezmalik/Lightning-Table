# About the Dataset
The way the code has been setup, it requires the dataset folder to be in a very specific format. The format is mentioned below.

## Folder Structure of Dataset
The folder structure should be the following.

```
- dataset
	- images
	- annotations.csv
	- classes.csv
```

!!! important
    The name of the files and the folder should be exactly like mentioned here. This is because the names are hard coded so that only the path to root of the dataset folder needs to supplied when training. The headers shown in description below is just for information and should not be included in the CSV files themselves. 


- `images`: A folder that contains all the images.
- `annotations.csv`: Annotations CSV file that contains the annotations in the following format: 
  `image_id,x1,y1,x2,y2,class_name`
  Note that the `image_id` should just be name of the file in the images folder and not a path to the image. The dataset implementation in the code takes care of the path automatically. `x1,y1` are the coordinates for the top-left of the box. `x2,y2` are the coordinates for the bottom-right of the box. 
- `classes.csv`: Class list CSV file that contains the list of class in the following format: 
  `class_name,class_id`
  The object classes should start from 1 and not 0, 0 is reserved for background classes, if the dataset contains examples for the background class then 0 class_id can be used.

## Some Notes
There are some particularities about the current implementation of dataset that need to be noted:

- There is no separate annotation file for the validation and training set. All the annotations should be mentioned in the `annotations.csv` file. Instead the Evaluation and Training split should be specified as an argument for the `trainer.py` file (See the 'How to Train the Networks' section).

- Negative examples are not allowed. These are images where there is no object to be detected. If such images are present, the loss is drived to infinity. This is a known issue currently for the code.