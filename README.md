
-----

#  Credit Card Fraud Detection using Deep Learning

This repository contains a deep learning project for detecting fraudulent transactions in a credit card dataset. The workflow utilizes techniques to address severe class imbalance and employs a simple **Feedforward Neural Network (FNN)** built with **TensorFlow/Keras** for classification.

##  Features

  * **Data Exploration (EDA):** Initial analysis including correlation heatmap.
  * **Imbalance Handling:** Uses the **SMOTE (Synthetic Minority Over-sampling Technique)** algorithm to balance the highly skewed 'Class' distribution.
  * **Feature Scaling:** Standardizes the feature set using **`StandardScaler`**.
  * **Deep Learning Model:** Implements a simple FNN with two hidden layers.
  * **Training & Evaluation:** Uses **Early Stopping** for regularization and evaluates performance using **Precision, Recall, and F1-Score**.



### Prerequisites

  * Python 3.x
  * The following libraries (already included in your code):
      * `pandas`
      * `numpy`
      * `matplotlib`
      * `seaborn`
      * `scikit-learn`
      * `imblearn` (for SMOTE)
      * `tensorflow`

### Data

The dataset used is the **`creditcard.csv`** file, which is a popular dataset for fraud detection.

> **Note:** The code assumes the file is stored in your Google Drive at `/content/drive/My Drive/creditcard.csv` as it uses `google.colab import drive` and mounts the drive.

### Execution

The entire workflow can be run sequentially, preferably within a **Google Colab** environment, due to the drive mounting commands.

1.  **Mount Google Drive** and load the data.
2.  Perform **EDA** and visualize the imbalanced class distribution.
3.  Apply **SMOTE** to generate synthetic minority class samples.
4.  Scale the features and **split** the data into training and testing sets.
5.  **Build, compile, and train** the Keras model.
6.  **Evaluate** the model and calculate performance metrics.

##  Performance and Results

### 1\. Class Balancing (SMOTE)

The original dataset was highly imbalanced. After applying SMOTE, the class distribution was perfectly balanced:

| Class | Count |
| :---: | :---: |
| **0** | **284,315** |
| **1** | **284,315** |

### 2\. Model Architecture

A sequential model was used:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 28)]              0         
_________________________________________________________________
dense (Dense)                (None, 64)                1856      
_________________________________________________________________
dense_1 (Dense)              (None, 64)                4160      
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 65        
=================================================================
Total params: 6,081
Trainable params: 6,081
Non-trainable params: 0
_________________________________________________________________
```

### 3\. Final Metrics

The model was evaluated on the test set (`x_test`, `y_test`).

| Metric | Result |
| :---: | :---: |
| **Loss** | **0.01** |
| **Accuracy (Acc)** | **0.99** |
| **Precision** | **99.682%** |
| **Recall** | **99.969%** |
| **F1-Score** | **99.825%** |

-----

