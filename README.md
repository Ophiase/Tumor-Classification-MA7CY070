# Tumor MRI Classification Comparison: ViT vs. CNN

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11.5+](https://img.shields.io/badge/Python-3.11.5%2B-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg)](https://pytorch.org/)

This repository compares **Vision Transformers (ViT)** and **Convolutional Neural Networks (CNN)** for tumor classification using MRI data. 
It aims at verifying the performances of each on a specific dataset for a course project at university.

## Summary

- Installation
- Requirements

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

---
