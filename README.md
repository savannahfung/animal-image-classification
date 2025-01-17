# Animal Image Classification

## Overview

This repository contains the code for the Animal Image Classification task. The objective of the task was to categorize images of animals into 151 different categories using a convolutional neural network (CNN) and various model improvement techniques.

## Dataset

The dataset consists of:

- Total Images: 6270
- Classes: 151
- Image Format: RGB
- Image Size: 224x224 pixels
- File Format: JPEG

The images are organized into 151 folders, each corresponding to a different animal class.

## Baseline Model

The baseline model provided is a simple Convolutional Neural Network (CNN) designed for the task of classifying animal images into 151 different categories.

## Task

The task is to modify the provided baseline code to improve the model's performance in classifying animal images.

## Getting Started

### Prerequisites

- Python 3.7 or later
- PyTorch
- Torchvision
- Other dependencies specified in `requirements.txt`

### Installation

1. Clone this repository.

   ```bash
   git clone https://github.com/savannahfung/animal-image-classification.git
   cd animal-image-classification
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebooks

Navigate to the `notebooks` directory and open the provided Jupyter notebooks to view the performance of different models:

- Baseline model: `baseline_cnn.ipynb`
- Final baseline model: `final_baseline.ipynb`
- MobileNetV3 model: `mobilenet_v3.ipynb`
- Quantizable MobileNetV3 model (final model): `final_model.ipynb`

### Model Definition

Model definitions are provided in the `models` directory:

- Base class for image classification models: `base_class.py`
- Baseline CNN model definition: `baseline_cnn.py`
- Final baseline model definition: `final_baseline.py`
- MobileNetV3 model definition: `mobilenet_v3.py`
- Quantizable MobileNetV3 model definition: `mobilenet_v3_quant.py`

### Training and Evaluating the Model

Use the provided scripts in the `src` directory to train the model and evaluate the model performance:

- Train model: `train.py`
- Train model with learning rate scheduler: `train_lr_scheduler.py`
- Evaluate model: `evaluate.py`
- Plot results: `plot.py`
- Compute number of FLOPs: `FLOPs_counter.py`

## Project Structure

`animal-image-classification/`

- `dataset/`: Directory for the dataset
- `models/`
  - `base_class.py`: Base class for image classification models
  - `baseline_cnn.py`: Baseline CNN model definition
  - `final_baseline.py`: Final baseline model definition
  - `mobilenet_v3_quant.py`: Quantizable MobileNetV3 model definition
  - `mobilenet_v3.py`: MobileNetV3 model definition
- `notebooks/`
  - `baseline_cnn.ipynb`: Notebook for training and testing the baseline CNN model
  - `final_baseline.ipynb`: Notebook for training and testing the final baseline model
  - `final_model.ipynb`: Notebook for training and testing the final model
  - `mobilenet_v3.ipynb`: Notebook for training and testing the MobileNetV3 model
- `results/`: Directory for saving model results and training histories
- `src/`
  - `device_manager.py`: Script to manage device configuration
  - `evaluate.py`: Script to evaluate model performance
  - `FLOPs_counter.py`: Script to compute FLOPs for the model
  - `plot.py`: Script for plotting results
  - `train_lr_scheduler.py`: Script to train the model with learning rate scheduler
  - `train.py`: Script to train the model
- `.gitignore`
- `README.md`: This README file
- `requirements.txt`: Python dependencies
