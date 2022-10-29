from os import listdir, rename
from os.path import isfile, join
import pandas as pd

pathToIDAssignments = "dataset/table-dataset/id-assignments.csv"
pathToImagesFolder = "dataset/table-dataset/images/"

def csvToDataFrame(pathToCSV):
    return pd.read_csv(pathToCSV)

def main():
    print("Starting Renamer")

    # Get the ID Assignments
    idAssignments = csvToDataFrame(pathToIDAssignments);

    foundFiles = 0

    for f in listdir(pathToImagesFolder):

        # Match the filename in the folder to the assignments
        find = idAssignments.loc[idAssignments['original-name'] == f]
        
        # Convert the found files to a list
        filesFoundInAssignments = find.values.tolist()

        # Only proceed if a file was found
        if(len(filesFoundInAssignments) > 0):
            foundFiles += 1
            oldNamePath = join(pathToImagesFolder, f)
            newNamePath = join(pathToImagesFolder, filesFoundInAssignments[0][0])

            rename(oldNamePath, newNamePath)

    
    print("Files Renamed: " + str(foundFiles))



if __name__ == "__main__":
    main()

