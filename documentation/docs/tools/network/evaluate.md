# Evaluate

This is a simple utility that can be used to evaluate different approaches from the pipeline of table content extraction. The [Evaluation Guide](../../guides/evaluation/index.md) makes use of the functionality of this utility, the only difference is that it provides a much cleaner interface using the makefile. Same results can be achieved using this utility as well. 

## Purpose

This utility provides an ability to reproduce the results generated during the work done as part of the master thesis. In essence, these results can be used as a baseline for future work to be done on the project.

## How it works

The utility provides the ability to replicate the experiments done during the master thesis for the evaluation of the complete pipeline. For that, some known data is required on which different parts of the pipeline can be executed in order to make predictions and compare them. This data is available in the [Downloads](../../downloads/index.md) section.

## Parameters

Following parameters are required to run the script:

- `-t` or `--type`:
This parameter specifies the type of evaluation to perform. The options are 'detection' which denotes Table Detection Evaluation, 'classification' which denotes Table Classification Evaluation, 'complete' which denotes Complete Pipeline Evaluation and finally 'all' which will perform previous three evaluations together and show the results collectively.

- `-p` or `--path`:
This parameter specifies the path to the folder that contains the appropriate data according to the evaluation type specified.

## Example

An example on how to run the script is following:

```
python network/evaluate.py -t complete -p evaluation-data/complete/
```