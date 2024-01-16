# Celebrity Image Classifier

![Celebrity Image Classifier](https://your-image-url.com)

## Overview

This repository presents a Celebrity Image Classifier that utilizes both traditional Machine Learning and Deep Learning techniques. The project involves preprocessing celebrity images using OpenCV, employing Haar Cascade for the Machine Learning part, and implementing an Artificial Neural Network (ANN) for Deep Learning. The dataset comprises labeled images of five prominent celebrities: Lionel Messi, Maria Sharapova, Serena Williams, Roger Federer, and Virat Kohli.

## Project Stages

### 1. Data Collection

The image dataset is gathered for each celebrity, with a focus on detecting faces having at least two eyes. Haar Cascade Classifier is used to extract valid faces from raw images. The extracted faces are then manually curated to ensure a more accurate result and optimize the model.

### 2. Data Organization

Cropped images are stored in a dedicated folder using Python code and the `os` library.

### 3. Feature Extraction

Wavelet transformation is applied to the cropped celebrity images for extracting facial features.

### 4. Developing Training and Testing Data

The images undergo preprocessing, including resizing to a normalized size (32 X 32), and are stacked vertically to optimize the dataset. The dataset is divided into input data (X) and labels (Y), where celebrity names are encoded from [0 — 4].

### 5. Training the Model using GridSearchCV

The X and Y data are fed into different machine learning algorithms, including Random Forest Classifier, Logistic Regression, and Support Vector Classifier. GridSearchCV is employed for hyperparameter optimization, ensuring the best algorithm is selected for the project. Pipelining is used to automate dataset scaling and hyperparameter optimization.

## Project Structure

- **/data**: Contains the raw and cropped celebrity image datasets.
- **/models**: Stores the trained machine learning and deep learning models.
- **/utils**: Utility functions and scripts for image processing and feature extraction.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/celebrity-image-classifier.git
   cd celebrity-image-classifier
