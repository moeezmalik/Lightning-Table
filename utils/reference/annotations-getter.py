from os import listdir
from os.path import isfile, join
import pandas as pd
import xml.etree.cElementTree as et

# Path to the assigned IDs CSV file
pathToImageNamesCSV = "dataset/table-dataset/id-assignments.csv"
pathToXMLAnnotationsFolder = "dataset/table-dataset/annotations/"

pathToSaveAllAnnotations = "dataset/table-dataset/all-annotations.csv"
pathToSaveAllAnnotationsWithoutOriginalNames = "dataset/table-dataset/all-annotations-no-original.csv"


def xmlAnnotationsToList(pathToXML):

    # This is the root of the XML tree
    tree = et.parse(pathToXML)
    root = tree.getroot()

    # Here we extract the file name
    filename = root.find("filename").text

    # Get all objects in the current file
    objectsRoot = root.findall("object")

    # Initialise the list to fill all the annotations in
    readAnnotations = []

    # Iterate over all found objects
    for object in objectsRoot:
        
        # This is the class name
        className = object.find("name").text


        # Now we go further into the tree and get the bounding box
        bndBoxRoot = object.find("bndbox")

        x1 = bndBoxRoot.find("xmin").text
        y1 = bndBoxRoot.find("ymin").text
        x2 = bndBoxRoot.find("xmax").text
        y2 = bndBoxRoot.find("ymax").text

        readAnnotations.append([filename, x1, y1, x2, y2, className])


    return readAnnotations

def annotationsXMLToDataFrame(pathToAnnotationsFolder):

    totalElementsFound = 0
    totalFilesFound = 0
    totalAnnotationsRead = 0

    listOfAnnotations = []

    for f in listdir(pathToAnnotationsFolder):

        totalElementsFound += 1

        completePathToFile = join(pathToAnnotationsFolder, f)

        if isfile(completePathToFile):
            totalFilesFound += 1
            readAnnotations = xmlAnnotationsToList(completePathToFile)

            for annotation in readAnnotations:
                if(len(annotation) == 6):
                    totalAnnotationsRead += 1
                    listOfAnnotations.append(annotation)

    print("Total Elements Found: " + str(totalElementsFound))
    print("Total Files Found: " + str(totalFilesFound))
    print("Total Annotations Read: " + str(totalAnnotationsRead))

    imageLabels = pd.DataFrame(listOfAnnotations, columns=["original-name", "x1", "y1", "x2", "y2", "class"])
    # print(imageLabels)
    return imageLabels

def imageNamesCSVToDataFrame(pathToCSV):
    return pd.read_csv(pathToCSV)

def main():

    # Read the files and get image labels and annotations
    imageLabels = annotationsXMLToDataFrame(pathToXMLAnnotationsFolder)
    imageNames = imageNamesCSVToDataFrame(pathToImageNamesCSV)

    print("")
    print("Unique in Names: ")
    print(imageNames.nunique())
    print("")
    print("Unique in Labels: ")
    print(imageLabels.nunique())
    print("")

    # Dataframe and CSV preparation for all Annotations
    allAnnotations = pd.merge(left=imageNames, right=imageLabels, how='left', on="original-name")
    print(allAnnotations)
    allAnnotations.to_csv(path_or_buf=pathToSaveAllAnnotations, header=True, index=False)

    # Dataframe and CSV preparation for all Annotations without Original Names
    allAnnotationsNoOriginal = allAnnotations.drop(columns=['original-name'])
    print(allAnnotationsNoOriginal)
    allAnnotationsNoOriginal.to_csv(path_or_buf=pathToSaveAllAnnotationsWithoutOriginalNames, header=True, index=False)


if __name__ == "__main__":
    main()