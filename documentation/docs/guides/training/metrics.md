# About Logging and Evaluation Metrics
Evaluation metrics are critical in order to evaluate the model performance. It is necessary to keep track if these metrics throughout the training process as well in order make decisions about hyperparameters for example.

## Weights and Biases Integration
In order to keep track of the model performance throughout the training process, there is [Weights and Biases](https://wandb.ai/site) integration built into the code. Weights and biases is an MLOps platform that allows us to keep track of Machine Learning experiements.

In order to log values to the Weights and Biases dashboard, a project name needs to be specified when running the training script, see the 'How to Train the Networks' section for more information. In addition to specifying the name of the projects, an account at Weights and Biases is needed. After logging in, the secret key will be available at [https://wandb.ai/authorize](https://wandb.ai/authorize). Enter this key when prompted while running the training script. 

## Logged Metrics
Different metrics are logged during the training and the validation phases. These can be visualised on the Weights and Biases dashboard. 

### During training
During the training process, following quantities are logged.

- **Classification Loss:** This is loss that the network calculates when classifying the region of objects into classes. For our case there is only one class of interest i.e. Table hence this loss goes down fairly quickly during the training phase.
- **Regression Loss:** This represents the loss that is calculated when the network tries to draw the bounding boxes around the tables. For us this quantity is more representative of the network training success.
- **Total Loss:** This is just the sum of both the classification and regression loss and provides the overview of the network training performance.
- **Mean Total Loss:** This is just the total loss averaged over the entire epoch.
  
### During Evaluation
During the evaluation of the dataset, following quantities are logged.

- **Averaged IoU:** During the evaluation phase, the network is asked to make bounding box predictions over an image. Those box predictions are taken and compared with the ground truth bounding boxes. An overlap is calculated using a measure called IoU (Intersection of Union). 
- **Precision:** During the evaluation phase, the network tries its best to make predictions of where the tables are. The predictions are the bounding boxes. Each of these bounding boxes come with their confidence scores. We select some of these boxes with a certain confidence score threshold. In addition to the confidence score, we also compute the overlap of these boxes with the ground truths and only select the boxes with a certain overlap threshold. So after both of these thresholds applied, we will have final set of detected tables. Precision value will tell us how many of these detected tables were actually tables. 
- **Recall:** The process of calculating the recall metric is very similar to precision as described above. After applying both the confidence score and IoU thresholds, we get a set of predicted detections for the table. Recall tells us how many of the tables were actually detected by the network. 