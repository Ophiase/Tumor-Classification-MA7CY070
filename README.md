# Tumor MRI Classification Comparison: ViT vs. CNN

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9.13+](https://img.shields.io/badge/Python-3.9.13%2B-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg)](https://pytorch.org/)

This repository compares **Vision Transformers (ViT)** and **Convolutional Neural Networks (CNN)** for tumor classification using MRI data.
It aims at verifying the performances of each on a specific dataset for a course report at university.

## Summary

- Installation and Execution
- Key Architectural Differences
- Data
- Experiments

## Installation and Execution

### Installation

```bash
# Specify in the Makefile your python command
make install-cpu
make install-gpu # highly recommended 
```

### Download the datasets

You may need to configure your kaggle API key to download the dataset.
- [Kaggle API documentation](https://www.kaggle.com/docs/api)

```python
make download
# or
python3 -m download 
```

You can visualize the datasets in : ``visualize.py``

## Training

Each model use the function to train on the dataloader.\
I choose to call it from a notebook :
- Regular CNN : ``cnn_approach.ipynb``
- EfficientNet (*Advanced CNN*) : ``efficientnet_approach.ipynb``
- ViT : ``vit_approach.ipynb``

## Benchmarks

Models are compared in ``compare_trained_models.ipynb`` on their:
- Loss (train/test)
- Accurracy (train/test)
- Dataset from other sources

## Key Architectural Differences

Below are key differences between the architectures:

1. **Inductive Bias**  
   - **CNNs**: Strong spatial inductive bias (localized filters, hierarchical feature detection).  
   - **ViTs**: Weaker inductive bias, relying on self-attention to capture global dependencies.  

2. **Feature Detection**  
   - **CNNs**: Detect local patterns (edges, textures) via convolutional filters.  
   - **ViTs**: Model long-range relationships through attention mechanisms.  

3. **Data Requirements**  
   - **CNNs**: Perform well with limited data due to their hierarchical structure.  
   - **ViTs**: Require large datasets for pretraining to avoid overfitting.  

## Datasets

We will train a model to classify a tumor, and safe check its capacity of generalization on another dataset (eg. just for tumor/healthy detection). 

- $\checkmark$ [masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
   - Classes: ``Glioma``, ``Meningioma``, ``Pituitary``, ``NoTumor`` 
   - Used for training
- [sartajbhuvaji/brain-tumor-classification-mri](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
   - Classes: ``Glioma``, ``Meningioma``, ``Pituitary``, ``NoTumor`` 
   - Used for validation
- [preetviradiya/brian-tumor-dataset](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset?select=Brain+Tumor+Data+Set)
   - Classes: ``Brain Tumor``, ``Healthy``
   - Used for validation

## Experiments

- Best at learning the dataset
   - ...
- Best at generalization (outside of the original dataset)
   - ...
- Fastest to train
   - ...
