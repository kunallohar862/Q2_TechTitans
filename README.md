# Q2_TechTitans
***Overview***

This repository contains code for a Convolutional Neural Network (CNN) implemented with TensorFlow and Keras to classify medical images into two classes: "NoEdema" and "Edema". The model is trained on a dataset of images organized in the specified directory structure.

**Requirements**

1. Python 3.x
2. TensorFlow
3. OpenCV
4. NumPy
5. Matplotlib
6. Scikit-learn
7. Seaborn (for confusion matrix visualization)

**Model Training**

1. Our model contain Convolutional Neural Network of 15 layers. Which consist of 5 2D concolutional layers, 5 max pooling layers, 5 dense layer.
2. Training is performed for 10 epochs and with a batch size of 64.
4. Model checkpoints are saved for the best validation accuracy.
5. And reduce factor of 0.9
6. For trainig dataset is split into 80-20, 80% for training and 20% for testing

**Evaluation**

1. The training history, accuracy, and loss plots are generated for model evaluation.
2. we get the accuracy of 81.5% 
3. Confusion matrix is plotted to assess classification performance on the test set.

**Saved Model**

1. The trained model's architecture is saved in model_architecture.json.
2. Model weights are saved in model_weights.h5.

**Single Image Prediction**

1. Load the saved model architecture and weights.
2. Predict the class of a single image in our case path of image to be predicted is (/content/drive/MyDrive/Colab Notebooks/Dev-o-Thon/archive/img_data/Edema/00000032_034.png in this example).
3. Visualize the prediction on the image.

**Acknowledgments**

This project was developed for the Dev-o-Thon competition, by team TechTitans.





