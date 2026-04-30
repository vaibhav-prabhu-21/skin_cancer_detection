# Skin Cancer Detection using CNN with Flask Deployment

## Overview

This project implements a Convolutional Neural Network (CNN) for automated skin lesion classification using dermoscopic images from the ISIC Skin Lesion Dataset. The trained model predicts whether a lesion is **benign** or **malignant** and is deployed through a Flask web application for real-time browser-based inference.

The application allows users to upload a skin lesion image and receive an instant prediction from the trained deep learning model.

---

## Dataset

Dataset used:

ISIC Skin Lesion Analysis Dataset  
Source: International Skin Imaging Collaboration (ISIC)

Dataset components:

- Dermoscopic images
- Segmentation masks
- GroundTruth.csv label file

Classification mapping used:

melanoma → malignant  
nevus / seborrheic keratosis → benign

---

## Features

- Binary classification: benign vs malignant
- CNN-based deep learning architecture
- Class imbalance handling using class weights
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Automatic best-model checkpoint saving
- Flask web deployment for real-time prediction
- Image upload interface for inference
- TensorFlow native model saving format (.keras)



---

## Model Architecture

The CNN architecture includes:

- Input preprocessing layer
- Convolutional layers
- ReLU activation
- MaxPooling layers
- Fully connected dense layers
- Softmax output layer

Loss function:

categorical_crossentropy

Optimizer:

Adam optimizer

Evaluation metric:

Validation accuracy

---

## Training Strategy

Training includes the following optimization techniques:

- ModelCheckpoint callback
- EarlyStopping callback
- ReduceLROnPlateau callback
- Class weight balancing
- Validation monitoring during training

These improve convergence and reduce overfitting.

---


---

## Model Architecture

The CNN architecture includes:

- Input preprocessing layer
- Convolutional layers
- ReLU activation
- MaxPooling layers
- Fully connected dense layers
- Softmax output layer

Loss function:

categorical_crossentropy

Optimizer:

Adam optimizer

Evaluation metric:

Validation accuracy

---

## Training Strategy

Training includes the following optimization techniques:

- ModelCheckpoint callback
- EarlyStopping callback
- ReduceLROnPlateau callback
- Class weight balancing
- Validation monitoring during training

These improve convergence and reduce overfitting.

---

## Installation

Clone the repository:
git clone:https://github.com/vaibhav-prabhu-21/skin-cancer-detection-cnn-flask.git

Move into project directory:


cd skin-cancer-detection-cnn-flask


Install dependencies:


pip install -r requirements.txt


---

## Run Flask Application

Start the Flask server:


python app.py


Open browser:http://127.0.0.1:5000

Upload a dermoscopic image to receive prediction results.

---

## Model File

The trained model is saved in TensorFlow native format:


skin_cancer_model.keras


This format preserves:

- model architecture
- trained weights
- optimizer configuration

and enables easy deployment in Flask applications.

---

## Technologies Used

Python  
TensorFlow  
Keras  
OpenCV  
NumPy  
Pandas  
Matplotlib  
Scikit-learn  
Flask  

---

## Results

The CNN model successfully classifies dermoscopic skin lesion images into benign and malignant categories using the ISIC dataset. Performance improvements were achieved using class weighting, learning rate scheduling, and early stopping strategies.

---

## Future Improvements

Possible future extensions:

- Multi-class lesion classification
- Transfer learning using EfficientNet or ResNet
- Lesion segmentation using U-Net
- Docker-based deployment
- Cloud deployment (AWS / Azure / GCP)
- Mobile application integration

---

## Disclaimer

This project is intended for educational and research purposes only.

It is not a medical diagnostic tool. Clinical decisions must always be made by qualified healthcare professionals.

---

## Author

Vaibhav Prabhu

