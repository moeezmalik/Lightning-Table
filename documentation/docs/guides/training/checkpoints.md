# About the Checkpoints
The checkpoint is a terminology taken from the PyTorch Lightning framework. It is a file that contains the state of the model. Checkpoints are saved successively throughout the training process. The benefit of keeping checkpoints is that we can keep track and save the model state that performed the best since training of the network is an iterative process. Following checkpoints are saved through the training process:

- Checkpoint for the last epoch
- Checkpoint for the best Evaluation Average IoU
- Checkpoint for the best Evalutaion Precision (at 75% Confidence Score and 90% IoU)
- Checkpoint for the best Evaluation Recall (at 75% Confidence Score and 90% IoU)