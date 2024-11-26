# EEG Brainwave Data Sentiment Classification

This project uses EEG brainwave data to classify emotional states (Positive, Neutral, and Negative) based on preprocessed statistical features. The data was collected using a Muse EEG headband and processed to derive frequency-domain features, enabling machine learning and deep learning models to identify emotional states effectively.

---

## Project Overview

This project investigates the feasibility of using EEG brainwave data to discern emotional states. The dataset contains processed brainwave features (via statistical extraction) recorded for three emotional states:
- **Positive**
- **Neutral**
- **Negative**

By applying various machine learning and deep learning models, this project aims to achieve robust sentiment classification from EEG-derived features.

---

## Dataset Description

The dataset was collected using a **Muse EEG headband** with electrodes placed at **TP9, AF7, AF8, TP10**. The emotional states were evoked using curated stimuli, with each state recorded for 3 minutes. Additionally, 6 minutes of resting (Neutral) data was collected.  

### Stimuli Used:
- **Negative**:
  - *Marley and Me (Death Scene)*
  - *Up (Opening Death Scene)*
  - *My Girl (Funeral Scene)*
- **Positive**:
  - *La La Land (Opening Musical Number)*
  - *Slow Life (Nature Timelapse)*
  - *Funny Dogs (Funny Clips)*

### Preprocessing:
- Data was processed with a **statistical extraction strategy** to represent temporal features mathematically.
- Features derived from the dataset include Fourier Transform (FFT) components (`fft_0_b` to `fft_749_b`), capturing the frequency characteristics of brainwaves.

### Data Link:
The dataset is available on Kaggle:
[EEG Brainwave Dataset: Feeling Emotions](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions/data)

---

## Methodology

1. **Data Preprocessing:**
   - Labels were mapped to numeric values:  
     - *Positive (2)*, *Neutral (1)*, *Negative (0)*.
   - Features were normalized using `StandardScaler`.

2. **Machine Learning and Deep Learning Models:**
   - **Machine Learning Models**:
     - Random Forest
     - Support Vector Machine (SVM)
     - Multi-layer Perceptron (MLP)
   - **Deep Learning Model**:
     - Fully connected neural network using TensorFlow/Keras.
   - The dataset was split into training and testing sets (70:30 ratio).

3. **Evaluation:**
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-Score)
   - Accuracy Comparison across models.

---

## Models Implemented

1. **Random Forest Classifier**
   - Ensemble-based model for robust feature importance and classification.

2. **Support Vector Machine (SVM)**
   - Kernel-based model for high-dimensional classification.

3. **Multi-layer Perceptron (MLP)**
   - Feed-forward neural network with 2 hidden layers.

4. **Deep Learning Neural Network**
   - TensorFlow/Keras model:
     - 3 fully connected layers with ReLU activation.
     - Dropout for regularization.
     - Output layer with Softmax activation.

---

## Requirements

### Python Packages
- Python 3.7 or higher
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`

### Installation
Install the required packages using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
