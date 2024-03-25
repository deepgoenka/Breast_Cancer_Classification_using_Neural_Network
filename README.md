# Breast Cancer Classification using Neural Network
This repository contains code for building a neural network model to classify breast cancer using the Breast Cancer Wisconsin (Diagnostic) Dataset from the UCI Machine Learning Repository.

# Dataset
The Breast Cancer Wisconsin (Diagnostic) Dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image. The dataset has 569 instances with 30 features each. More information about the dataset can be found here.

# Libraries Used
 * pandas
 * numpy
 * matplotlib
 * seaborn
 * scikit-learn
 * TensorFlow
 * Keras

# Model Architecture
The neural network model consists of:

 * Input layer with 30 neurons (input shape corresponding to the number of features)
 * One hidden layer with 32 neurons and ReLU activation function
 * Output layer with 2 neurons (representing the two classes) and softmax activation function

# Training and Evaluation
The model is trained using the Adam optimizer and categorical crossentropy loss function. It is trained for 20 epochs with a batch size of 64. Evaluation metrics such as accuracy, loss, confusion matrix, and classification report are generated to assess the model's performance.

# Results
Accuracy: 96%
