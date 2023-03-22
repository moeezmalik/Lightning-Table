---
hide:
    - navigation
---

# Downloads

This section will provide links to download files and folders necessary to run the scripts in this utility.

## Evaluation Data

[Download Link](https://www.dropbox.com/s/lal6pdm77zlnuok/evaluation-data.zip?dl=1)

The evaluation data contains all the models and the data required for reproducing the evaluation results for the experiments performed as part of the thesis. The checkpoints for deep learning models are also included, there is no need to download the checkpoits below separately. The [Reproduce Evaluation](../guides/evaluation/index.md) guide can be studied to use this evaluation data correctly.

## Checkpoints

Checkpoints are files that contains the trained weights of the networks (alongwith other information). Each model type has its own type checkpoints that will be generated during the training. Please find the relevant checkpoints below:

#### VanillaFasterRCNNV2
[Download Link](https://www.dropbox.com/s/a08mfi9xjh88bwd/best-fasterrcnn-v2.ckpt?dl=1)

This PyTorch Lightning Checkpoint contains the weights for the best trained FasterRCNN v2 model. This models was selected as being the best at detecting tables during the evaluations performed.

#### VanillaFasterRCNN
[Download Link](https://www.dropbox.com/s/oag1osnax4wu9tw/best-fasterrcnn.ckpt?dl=1)

This PyTorch Lightning Checkpoint contains the weights for the best trained FasterRCNN model. 

#### VanillaRetinaNet
[Download Link](https://www.dropbox.com/s/lqdkk2holgu4urk/best-retinanet.ckpt?dl=1)

This PyTorch Lightning Checkpoint contains the weights for the best trained RetinaNet model. 