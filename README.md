# Celebrity Image Classifier

<img src="https://github.com/John-Alex07/Portfolio/blob/master/static/img/portfolio/portfolio-4.jpg" alt="Project Image" width="600" height="400">

## Overview

This repository presents a Celebrity Image Classifier that utilizes both traditional Machine Learning and Deep Learning techniques. The project involves preprocessing celebrity images using OpenCV, employing Haar Cascade for the Machine Learning part, and implementing an Artificial Neural Network (ANN) for Deep Learning. The dataset comprises labeled images of nine prominent celebrities.
# Celebrity Labels

Here are the labels assigned to each celebrity for reference:

```python
{
 'ben_afflek': 0,
 'jerry_seinfeld': 1,
 'lionel_messi': 2,
 'madonna': 3,
 'maria_sharapova': 4,
 'mindy_kaling': 5,
 'roger_federer': 6,
 'serena_williams': 7,
 'virat_kohli': 8
}
```

## Project Stages

### 1. Data Collection

The image dataset is gathered for each celebrity, with a focus on detecting faces having at least two eyes. Haar Cascade Classifier is used to extract valid faces from raw images. The extracted faces are then manually curated to ensure a more accurate result and optimize the model.

### 2. Data Organization

Cropped images are stored in a dedicated folder using Python code and the `os` library.

### 3. Feature Extraction

Wavelet transformation is applied to the cropped celebrity images for extracting facial features.

### 4. Developing Training and Testing Data

The images undergo preprocessing, including resizing to a normalized size (32 X 32), and are stacked vertically to optimize the dataset. The dataset is divided into input data (X) and labels (Y), where celebrity names are encoded from [0 — 8].

### 5. Training the Model using GridSearchCV

The X and Y data are fed into different machine learning algorithms, including Random Forest Classifier, Logistic Regression, and Support Vector Classifier. GridSearchCV is employed for hyperparameter optimization, ensuring the best algorithm is selected for the project. Pipelining is used to automate dataset scaling and hyperparameter optimization.

```python
model_cnn = keras.Sequential([
    keras.layers.Conv2D(input_shape=(64,64,1) ,filters=12, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D((2,2)),
    
    keras.layers.Conv2D(filters= 12, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D((2,2)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='tanh'),
    keras.layers.Dense(256, activation='tanh'),
    keras.layers.Dense(8, activation='softmax')
])
opt = keras.optimizers.Adam(learning_rate=0.00001)
model_cnn.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

# CNN Model Evaluation

Here's the detailed classification report showcasing the model's performance:

| Celebrity         | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| ben_afflek        | 1.00      | 0.14   | 0.25     | 7       |
| jerry_seinfeld    | 1.00      | 0.94   | 0.97     | 18      |
| lionel_messi      | 0.85      | 0.83   | 0.84     | 35      |
| madonna           | 1.00      | 0.64   | 0.78     | 11      |
| maria_sharapova   | 0.82      | 1.00   | 0.90     | 32      |
| mindy_kaling      | 0.94      | 0.89   | 0.92     | 19      |
| roger_federer     | 0.87      | 0.93   | 0.90     | 29      |
| serena_williams   | 0.93      | 0.89   | 0.91     | 28      |
| virat_kohli       | 0.87      | 0.98   | 0.92     | 42      |

- **Accuracy**: 0.89
- **Macro Avg**: 0.92, 0.81, 0.82, 221
- **Weighted Avg**: 0.90, 0.89, 0.88, 221

# Machine Learning Model Evaluation

Here's a summary of the accuracy for different models:

| Model                | Accuracy |
|----------------------|----------|
| SVM                  | 0.805970 |
| Random Forest        | 0.686567 |
| Logistic Regression  | 0.809970 |
| AdaBoost             | 0.492537 |
| LightGBM             | 0.701493 |
| KNN                  | 0.582090 |

# Conclusion
This project portrays a comparative study between the machine learning models and the deep learning models. The customised CNN model outperforms the Machine Learning models, however the performance of the model can be further improved with the help of a larger dataset. The limitations of the dataset impacts the evaluation of both the machine learning models as well as the deep learning models.
Out of the machine learning models, the performance of SVM and Logistic Regression is the best.

