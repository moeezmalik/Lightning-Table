"""
Name:
    datamodules.py

Description:
    This file was created in order to establish data
    and organise different data modules that will be
    required by PyTorch and PyTorch Lightning for
    loading in the data required to train the models.

Author: 
    Name:
        M. Moeez Malik
    Email:
        muhammad.moeez.malik@ise.fraunhofer.de
"""

### - IMPORTS - ###

# PyTorch and Lightning
from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import Dataset,DataLoader, random_split, Subset

# Others
from transforms import Compose, PILToTensor, ConvertImageDtype
import pandas as pd
from utilities import collate_fn
import random
from os.path import join
from PIL import Image

# Misc
from typing import Optional, Tuple


### - CLASSES - ###

# CSV Dataset Implementation
class CSVDataset(Dataset):
    """
    This class implements the functionality to read the
    dataset in the CSV format. This class inherits from the
    built-in Dataset class from PyTorch. A custom dataset
    class such as this one needs to written when dealing with
    custom datasets. There was no default dataset class that
    could read CSV data format. That is why this was class
    was built from scratch.

    Args:
        path (str): 
            Path to the directory that contains the following.
            Note: The folder and filenames should be exactly as
            mentioned.

            images:
                A folder that contains all the images

            annotations.csv:
                Annotations CSV file that contains the annotations
                in the following format:
                image_id,x1,y1,x2,y2,class_name

            classes.csv:
                Class list CSV file that contains the list of class
                in the following format:
                class_name,class_id
                
                The object classes should start from 1 and not 0, 0
                is reserved for background classes, if the dataset
                contains examples for the background class then 0
                class_id can be used.
        
        transforms (Custom Transform Classes from transforms.py):
                This area the transforms that will be applied to
                all the images when the __get_item__ function is
                called. Default value is None.
    """
    
    def __init__(self, path, tranforms = None) -> None:
        
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
        Image.MAX_IMAGE_PIXELS = None
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
        
        # Even with the considerations, negative examples are not working yet
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

# Full Table Dataset
class TableDatasetModule(LightningDataModule):
    """
    This class loads the full Table Detection dataset
    that can be used for the training of the PyTorch
    detection networks.

    Args:
        path (str): 
            Path to the directory that contains the following.
            Note: The folder and filenames should be exactly as
            mentioned.

            images:
                A folder that contains all the images.

            annotations.csv:
                Annotations CSV file that contains the annotations in the following
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

        batch_size (int):
            This is the batch size that will be used when sampling the
            images from the dataset.

        train_eval_split (float):
            This is the argument that will denote the split of the full
            dataset between training and evaluation. It should be a value
            between 0 and 1. For example 0.8 denotes 80% split whereby
            80% of the images will go to the training set and 20% will
            go the evaluation set.

        subset (int):
            If specified, this parameter will take out a subset of data
            from the provided dataset.

        num_workers (int):
            This sets the number of workers to use for the dataloading
            purposes.

        
    """
    def __init__(
        self,
        path: str='./',
        batch_size: int=2,
        train_eval_split: float=0.8,
        subset: int=None,
        num_workers: int=2
    ) -> None:
        
        super().__init__()

        self.path = path
        self.batch_size = batch_size
        self.transforms = Compose([
            PILToTensor(),
            ConvertImageDtype(torch.float)
        ])
        self.train_eval_split = train_eval_split
        self.subset = subset
        self.num_workers = num_workers

        return None

    def prepare_data(self) -> None:
        """
        This function is one of the hooks for the PyTorch Lightning Data Modules.
        The purpose of this function is initially prepare the dataset that will
        eventually be fed into the data loaders. As it can be observed in the
        implementations, it makes us of the custom CSVDataset that was built
        above.
        """

        original_dataset = CSVDataset(self.path, tranforms=self.transforms)
        len_original_data = original_dataset.__len__()

        if self.subset is None:
            # If we dont want a subset of our data

            self.full_dataset = original_dataset
        
        else:
            # If we want a subset of our data then we can generate the required number of
            # random indices and then select the files at those indices

            # Generate Random Indicees in range
            random_indices = random.sample(
                population=range(0, len_original_data),
                k=self.subset
            )

            # Select those random indices
            self.full_dataset = Subset(
                dataset=original_dataset,
                indices=random_indices
            )

    def setup(
        self,
        stage: Optional[str] = None
    ) -> None:
        """
        This function is one of the hooks for the PyTorch Lightning Data Modules.
        The purpose of this function is to return appropriate datasets depending
        upon the current stage of the model i.e. either training or testing. It
        also randomly splits the dataset into training and evaluation set depending
        upon the percentage provided.
        """

        full_set = self.full_dataset
        full_set_length = full_set.__len__()

        train_set_length = round(self.train_eval_split * full_set_length)
        eval_set_length = full_set_length - train_set_length

        # Printout the dataset information.
        print()
        print("Dataset Information")
        print("Total Images in Dataset: " + str(full_set_length))
        print("Number of Training Images: " + str(train_set_length))
        print("Number of Evaluation Images: " + str(eval_set_length))
        print()

        if stage == 'fit' or stage is None:
            self.train_set, self.eval_set = random_split(
                dataset=self.full_dataset,
                lengths=[train_set_length, eval_set_length],
                generator=torch.Generator().manual_seed(42)
                )

        if stage == 'test' or stage is None:

            print()
            print("TESTING")
            print("TESTING")
            print("TESTING")
            print()
            self.test_set = full_set

    def train_dataloader(self) -> DataLoader:
        """
        This function is one of the hooks for the PyTorch Lightning Data Modules.
        It returns the Dataloader object for the training steps.
        """

        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """
        This function is one of the hooks for the PyTorch Lightning Data Modules.
        It returns the Dataloader object for the validation steps.
        """
        return DataLoader(
            dataset=self.eval_set,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers
        )





# Toy Table Dataset
