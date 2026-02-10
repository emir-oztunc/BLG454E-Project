# Autoencoder-Based Mask Anomaly Detection

This project implements a deep learning-based anomaly detection approach for face mask detection using a **Convolutional Autoencoder (CAE)**. The system is designed to identify improperly worn or absent masks by modeling the distribution of "normal" (correctly masked) faces and identifying unmasked faces as out-of-distribution instances.

## Project Overview

The core objective is to automate the monitoring of mask compliance in public environments using visual surveillance. Unlike traditional binary classifiers that require balanced datasets, this model leverages **One-Class Classification (OCC)**, training solely on masked samples and using **reconstruction error** as a proxy for mask presence.

## Dataset

The project utilizes the **MAFA (Masked Faces Dataset)**, which contains over 35,000 annotated face images.
* **Training Set:** Filtered to include only correctly masked face images.
* **Test Set:** A heterogeneous mix of correctly masked, incorrectly masked, and unmasked faces.
* **Preprocessing:** Images are resized to $256 \times 256$, normalized to the [0, 1] range, and processed using **torchvision** transforms.

## Methodology

### 1. Autoencoder Architecture
The model utilizes a lightweight **Encoder-Decoder** structure:
* **Encoder:** 3 convolutional layers with ReLU activations to downsample the input and increase feature depth.
* **Decoder:** 3 transposed convolutional layers to upsample the latent representation and reconstruct the original image with a Sigmoid activation.

### 2. Training Procedure
* **Loss Function:** Mean Squared Error (MSE) between input and reconstructed images.
* **Optimizer:** Adam optimizer with a learning rate of $10^{-3}$.
* **Optimization:** Mixed precision training and early stopping with a patience of 5 epochs were employed to optimize resource usage and prevent overfitting.

## Evaluation Protocol

Anomaly detection is performed by computing the reconstruction error for each test image:
* **Thresholding:** A threshold (default: 75th percentile) is applied to classify images.
* **Labels:** Below-threshold images are predicted as "masked," while above-threshold images are "unmasked".
* **Metric:** Performance is assessed using the **ROC-AUC** score to evaluate discriminative ability across all thresholds.

## Reproducibility

### 1. Environment Setup
Clone the repository and install the necessary dependencies:

```bash
git clone [https://github.com/emir-oztunc/BLG454E-Project.git](https://github.com/emir-oztunc/BLG454E-Project.git)
cd BLG454E-Project
pip install -r requirements.txt
```

### 2. Data Preparation
Ensure the MAFA dataset is structured and label files (converted from .mat to .csv ) are placed in the appropriate directory as required by the custom dataset loaders:

```bash
# Example structure
# data/MAFA/train/     # Training images (correctly masked faces only) 
# data/MAFA/test/      # Evaluation images (mixed masked/unmasked faces) 
# labels/test_labels.csv # Converted labels from MATLAB .mat files 
```

### 3. Training and Evaluation
To train the autoencoder and evaluate it on the test set using the optimized parameters from the research report:

```bash
# Run training script (15 epochs as specified in methodology)
python train.py --epochs 15 --batch_size 32

# Run evaluation and generate ROC-AUC results
python evaluate.py --model_path artifacts/best_model.pth
```
## Results

The system demonstrated high classification performance on the MAFA test set:

* **ROC-AUC Score:** Achieved a performance score of **0.8943**.
* **Analysis:** Masked samples yielded significantly lower reconstruction errors compared to unmasked ones, as validated by the error distribution analysis.
* **Validation Loss:** The best performing model reached a validation loss of **0.0014**.

---
*Developed by **Emir Zal Öztunç** and **Tevhidenur Serdar** at **Istanbul Technical University**.*
