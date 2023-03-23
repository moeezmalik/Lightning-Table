# Detecting Tables using Pytorch and Lightning

> **WARNING:** This repository is still a work in progress and things might change drastically.

This repository contains the code for detecting tables in PDF documents by using the PyTorch and the Lightning framework. The following image is just an example of passing a PDF through one of the networks in this repository that is trained on detecting the tables. The red bounding-boxes show the areas in the image that the model has predicted as a table.

![Image title](/documentation/docs/assets/main-table-photo.png)

## More Information

Please find more information about this repository in the [documentation](https://moeezmalik.github.io/Lightning-Table/).

All of the documentation is also available locally. If the link does not work, the documentation website can be served locally using mkdocs. To do so, please run the following commands.

Install the material for mkdocs package
```
pip install mkdocs-material
```

Switch to the documentation folder
```
cd documentation
```

Run the website
```
mkdocs serve
```

Running the last command will show a link on the command line of the website running on local machine. That link can be used to acceess the complete documentation.