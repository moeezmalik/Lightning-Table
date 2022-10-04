"""
Name:
    data.py

Description:
    This file was created to hold the implementations
    of the core data libraries that are required for
    the training of PyTorch modules.

Author: 
    Name:
        M. Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

### - IMPORTS - ###

# PyTorch and Lightning
from typing import Tuple
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

# Image Related Imports
from PIL import Image

# OS related Imports
from os.path import join

### - CLASSES - ###

# CSV Dataset Implementation
class CSVDataset(Dataset):
    """
    This class implements the functionality to read the
    dataset in the CSV format.

    Args:
        path (str): 
            Path to the directory that contains the following.
            Note: The folder and filenames should be exactly as
            mentioned.

            images:
                A folder that contains all the images

            annotations.csv:
                Annotations CSV file that contains in the following
                format:
                image_id,x1,y1,x2,y2,class_name

            classes.csv:
                Class list CSV file that contains the list of class
                in the following format:
                class_name,class_id
                
                The object classes should start from 1 and not 0, 0
                is reserved for background classes, if the dataset
                contains examples for the background class then 0
                class_id can be used.
    """
    
    def __init__(self, path, tranforms = None) -> None:
        """
        This is the contructor function of the class CSVDataset. This
        will be called one time when the class is created.

        Args:
            path (str):
                This is the path to the directory that contains
                the dataset. See class description for more details
                on the folder structure.

            transforms (torchvision.transforms):
                This area the transforms that will be applied to
                all the images when the __get_item__ function is
                called. Default value is None.
        """
        
        # Get the full relative path to the CSV files
        path_to_annotations = join(path, "annotations.csv")
        path_to_classes = join(path, "classes.csv")
        path_to_images_folder = join(path, "images/")

        # Read the annotations CSV file into a pandas DataFrames
        annotations_df = pd.read_csv(path_to_annotations,
                                     names=['image_id', 'x1', 'y1', 'x2', 'y2', 'class_name']
                                    )
        
        # Use the correct datatypes for the integer columns in order to account
        # for missing values. Missing values occur in case of negative examples
        # when no bounding boxes are specified.
        annotations_df.x1 = annotations_df.x1.astype('Int64')
        annotations_df.y1 = annotations_df.y1.astype('Int64')
        annotations_df.x2 = annotations_df.x2.astype('Int64')
        annotations_df.y2 = annotations_df.y2.astype('Int64')
        annotations_df.class_name = annotations_df.class_name.astype('object')

        # Read the classes CSV file into a pandas DataFrame
        classes_df = pd.read_csv(path_to_classes,
                                 names=['class_name', 'class_id']
                                )
        
        # Use the correct datatypes for the integer columns in order to account
        # for missing values. Missing values occur in case of negative examples
        # when no bounding boxes are specified.
        classes_df.class_id = classes_df.class_id.astype('Int64')

        # Join the two dataframes to get the class_ids and remove
        # the class_name column because we do not need it
        annotations_with_class_id = pd.merge(left=annotations_df,
                                             right=classes_df,
                                             how='left',
                                             on='class_name'
                                            ).drop(columns=['class_name'])

        # Now that we have the list of all the individual annotations
        # with the box coordinates and the labels for the boxes. We
        # want to group these together by an individual image. We
        # are doing this in order to be compatible with the PyTorch
        # object detection model inputs.

        # The following query first groups the list of annotations
        # according to the image ids and then converts the box annotation
        # coordinates and the labels to a list per image. The return
        # type of the query is a Pandas Series object
        self.labels = annotations_with_class_id.groupby('image_id') \
                                               .apply(lambda x: x[['x1', 'y1', 'x2', 'y2', 'class_id']].values.tolist())
        
        # Extract the image ids from the result of the above query
        self.image_ids = self.labels.index

        # Save the path to images folder as global variable
        self.path_to_images_folder = path_to_images_folder

        # Save the transforms that were provided as global variables
        self.transforms = tranforms

        return None

    def __getitem__(self, idx) -> Tuple:
        """
        This function will be called whenever the dataloader requires a 
        new file or image in our case to be read from the file system
        alongwith all the annotations.

        Args:
            idx (int):
                This is the index argument and the image will be
                loaded according to this index argument. 

        Returns:
            image, target (tuple):
                The function should return a tuple of:
                    image:
                        A PIL Image of size (H, W)
                    target:
                        A dict containing the following fields
                            boxes (FloatTensor[N, 4]):
                                The coordinates of the N bounding boxes in [x0, y0, x1, y1]
                                format, ranging from 0 to W and 0 to H
                            labels (Int64Tensor[N]):
                                The label for each bounding box. 0 represents always the
                                background class.
                            image_id (Int64Tensor[1]):
                                An image identifier. It should be unique between all the
                                images in the dataset, and is used during evaluation
                            area (Tensor[N]):
                                The area of the bounding box. This is used during
                                evaluation with the COCO metric, to separate the metric
                                scores between small, medium and large boxes.
                            iscrowd (UInt8Tensor[N]):
                                Instances with iscrowd=True will be ignored during
                                evaluation.
                            (optionally) masks (UInt8Tensor[N, H, W]):
                                The segmentation masks for each one of the objects
                            (optionally) keypoints (FloatTensor[N, K, 3]):
                                For each one of the N objects, it contains the K keypoints
                                in [x, y, visibility] format, defining the object. 
                                visibility=0 means that the keypoint is not visible.
                                Note that for data augmentation, the notion of flipping
                                a keypoint is dependent on the data representation, and
                                you should probably adapt references/detection/transforms.py
                                for your new keypoint representation
        """

        # Get the image id for the specified index
        image_name = self.image_ids[idx]

        # Get the path to the image
        path_to_image = join(self.path_to_images_folder, image_name)

        # Read the image as a PIL Image object and convert to RGB format
        image = Image.open(path_to_image).convert('RGB')

        # Get the corresponding objects to the image loaded
        list_of_objects = self.labels.iat[idx]

        # Extract the box coordinates and the class labels
        # and put them into individual lists, then convert
        # the lists to tensors for pytorch
        num_objs = len(list_of_objects)

        boxes = []
        labels = []
        is_negative_example = False

        for object in list_of_objects:
            xmin = object[0]
            ymin = object[1]
            xmax = object[2]
            ymax = object[3]

            label = object[4]

            if label is pd.NA:
                is_negative_example = True
            else:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)


        # There is a possibility to specify negative examples in the dataseet
        # i.e. images in which are no objects of interest. In case of such examples
        # the target dictionary needs special tensors since the box and label lists
        # will be empty. This is done below. More information about this can be found
        # in this github issue: https://github.com/pytorch/vision/issues/2144

        if is_negative_example:

            boxes = torch.zeros((0, 4), dtype=torch.float32) 
            labels = torch.zeros(0, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            image_id = torch.tensor([idx])
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        else:

            boxes = torch.as_tensor(data=boxes, dtype=torch.float32)
            labels = torch.as_tensor(data=labels, dtype=torch.int64)

            # Calculate the area for the boxes
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            # Assign the index as the unique image id
            image_id = torch.tensor([idx])

            # Let all the instances of the object are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)






        # Now making a dictionary of the targets as specified
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        # Apply transforms if specified
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target

    def __len__(self):
        """
        This function will be called to get the overall size of the
        dataset.

        Returns:
            size (int):
                The size of the dataset i.e. how many images there are in this case
        """

        # Just returning the of the Pandas Series object of labels should
        # sufficient
        return self.labels.size