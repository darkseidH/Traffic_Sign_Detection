# Traffic Sign Recognition using Machine Learning

## Project Overview
This project focuses on developing an intelligent system capable of detecting and classifying traffic signs in images using machine learning techniques. The system aims to enhance road safety by providing accurate recognition of traffic signs, which is crucial for both autonomous vehicles and human drivers.

## Table of Contents
- [Introduction](#introduction)
- [Project Description and Objectives](#project-description-and-objectives)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Desktop Application](#desktop-application)
- [Challenges Encountered](#challenges-encountered)
- [Conclusion](#conclusion)
- [Usage](#usage)

## Introduction
Traffic sign recognition is vital for road safety, playing a significant role in guiding and ensuring the safety of drivers. This project, developed as part of the Master’s program in Data Science and Intelligent Systems, leverages advanced machine learning techniques to create a system that can accurately detect and classify traffic signs from images.

## Project Description and Objectives
### Description
The goal of this project is to develop a machine learning-based system that can identify and classify traffic signs in images. This system is essential for improving road safety by helping autonomous vehicles and drivers recognize and understand traffic signs, thus reducing accidents.

### Objectives
1. **Data Collection and Preprocessing**: Gather a comprehensive dataset of traffic sign images, annotate them, and preprocess the images for model training.
2. **Model Development**: Design, train, and evaluate various machine learning models, including Convolutional Neural Networks (CNNs) and Support Vector Machines (SVMs), for traffic sign classification.
3. **Real-Time System Integration**: Implement the trained model into a real-time system capable of detecting and classifying traffic signs in live videos or captured images.
4. **Desktop Application Development**: Develop a desktop application that utilizes the trained model to recognize traffic signs from images captured by users.

## Dataset
The dataset used in this project is sourced from the "traffic-sign-dataset-classification" available on Kaggle. It consists of images categorized into different classes of traffic signs, each labeled accordingly. The data is divided into training and testing sets to evaluate the model's performance.

## Model Training
The machine learning model used for traffic sign recognition is based on the RidgeClassifierCV. This model is chosen for its effectiveness in handling multicollinearities in the data through L2 regularization and cross-validation techniques. The training process includes:
1. **Data Preprocessing**: Resizing images, normalizing pixel values, and augmenting data to improve model robustness.
2. **Model Training**: Training the RidgeClassifierCV model on the preprocessed data and optimizing hyperparameters using cross-validation.
3. **Model Evaluation**: Assessing the model's performance on the test set to ensure its accuracy and reliability.

## Desktop Application
A desktop application is developed to demonstrate the real-time capabilities of the trained model. This application allows users to upload images and receive instant recognition results for the traffic signs present in the images.

### Application Features
- Upload images for traffic sign recognition.
- Display recognition results with confidence scores.
- User-friendly interface for seamless interaction.

## Challenges Encountered
1. **Image Quality**: Variations in image quality and lighting conditions posed challenges for consistent recognition.
2. **RAM Limitations**: Managing memory constraints during model training on large datasets required careful optimization.

## Conclusion
This project successfully demonstrates the development of a machine learning system for traffic sign recognition. The integration of the model into a desktop application showcases its practical utility in real-time scenarios, contributing to road safety advancements.

## Usage
### Prerequisites
- Python 3.7 or higher
- Required libraries: numpy, pandas, scikit-learn, tensorflow, tkinter

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd traffic-sign-recognition


## Install the required libraries:
```bash
pip install -r requirements.txt
```
## Running the Application
Launch the desktop application:

```bash
python Application.py
```
Use the interface to upload an image and get the recognition result.


## Files Description
- **Notebook**: The Jupyter notebook `ML_Traffic_Sign_Classification_Notebook.ipynb` contains the entire workflow for training the machine learning model. It includes steps such as data loading, preprocessing, model training, evaluation, and saving the trained model.
- **PDF Report**: The PDF report `Le rapport Projet ML.pdf` provides a detailed overview of the project, including the motivation, methodology, experiments, results, and conclusions. This document is essential for understanding the project's context and findings.
- **Desktop Application Code**: The Python script `Application.py` contains the code for the desktop application. This application utilizes the trained model to classify traffic signs from user-uploaded images. It includes functionalities for image upload, model prediction, and displaying results.

## Key Functions in Application.py
- `upload_image()`: Allows the user to select an image file for classification.
- `classify_image()`: Uses the trained model to predict the class of the uploaded image and displays the result.
- `main()`: Initializes and runs the desktop application.
