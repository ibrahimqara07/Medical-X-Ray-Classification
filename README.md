# Medical-X-Ray-Classification
– Medical X-Ray Classification

Chest X-Ray Pneumonia Detection using ResNet18

This repository contains a complete Google Colab training script for binary classification of chest X-ray images into NORMAL and PNEUMONIA classes. The solution uses transfer learning with a pre-trained ResNet18 model and strictly follows the task requirements.

The main objective is to achieve a minimum test accuracy of 88% while training only the final fully connected layer of the network.

Project Overview

The project applies a feature extraction approach using a ResNet18 model pre-trained on ImageNet. All convolutional layers are frozen, and only the classifier head is trained. Two training configurations are compared to analyze how a single hyperparameter change affects performance.

Dataset

Name: Chest X-Ray Images (Pneumonia)

Source: Kaggle (paultimothymooney/chest-xray-pneumonia)

The dataset is divided into:

Training

Validation

Test

Each subset contains two classes:

NORMAL

PNEUMONIA

The dataset is downloaded and extracted automatically using the Kaggle API in Google Colab.

Reproducibility

A fixed random seed is used for Python, NumPy, and PyTorch (CPU and GPU).
Deterministic behavior is enforced to ensure reproducible results across multiple runs, as required by the task.

Data Preprocessing

All images undergo the same preprocessing steps:

Resize to 224×224

Convert grayscale images to 3 channels

Convert to tensor

Normalize using ImageNet mean and standard deviation

This preprocessing ensures compatibility with the ImageNet-trained ResNet18 model.

Model Architecture

Pre-trained ResNet18 (ImageNet weights)

All layers are frozen

Final fully connected layer replaced with Linear(512, 2)

Only the final layer is trained

This setup reduces training time and minimizes overfitting.

Training Strategy
Version 1 (Baseline)

Optimizer: Adam

Learning Rate: 0.0001

Batch Size: 32

Epochs: 10

Version 2

Only one change is applied:

Reduced learning rate or

Increased batch size or

Optimizer changed to SGD with momentum

All other parameters remain identical for fair comparison.

Evaluation and Results

The script provides the following outputs:

Class distribution for train/validation/test sets

Training and validation accuracy/loss curves

Comparison table with test accuracies

Confusion matrix (2×2) from the best model

Visualization of 10 sample predictions from the test set

The best-performing model is selected automatically based on test accuracy.

Analysis

The final analysis compares Version 1 and Version 2 against the target accuracy. Performance differences are explained based on optimization behavior and convergence stability. The dataset imbalance is noted, as Pneumonia images are more frequent than Normal images.
