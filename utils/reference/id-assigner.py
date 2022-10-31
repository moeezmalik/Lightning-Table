from os import listdir
from os.path import isfile, join
from sys import path
import pandas as pd

# Path to the folder that contains the images
pathToImages = "dataset/table-dataset/images/"
pathToCSV = "dataset/table-dataset/id-assignments.csv"

# Get the name of all the files in the folder, skip the nested folders
onlyFiles = [f for f in listdir(pathToImages) if isfile(join(pathToImages, f))]

# Create a Pandas DataFrame and create a unique ID column
imageNames = pd.DataFrame(onlyFiles, columns=['original-name'])
imageNames.insert(0, 'id', range(1, 1 + len(imageNames)))
imageNames['id'] = "image_" + imageNames['id'].astype(str) + ".jpg"

# Print to console just for validation purposes
print(imageNames)

# Write the assigned ID and original name to a CSV file
imageNames.to_csv(path_or_buf=pathToCSV, header=True, index=False)