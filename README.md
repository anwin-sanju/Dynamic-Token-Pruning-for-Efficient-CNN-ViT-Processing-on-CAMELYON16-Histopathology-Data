# cnn-vit-camelyon16-token-pruning

**Dynamic Token Pruning for Efficient CNN-ViT Processing on CAMELYON16 Histopathology Data**

---

## Description

This repository presents a lightweight, efficient architecture for histopathology image classification on the CAMELYON16 dataset. We combine the spatial awareness of Convolutional Neural Networks (CNNs) with the global attention of Vision Transformers (ViTs), guided by a novel **dynamic token pruning strategy**.

Whole slide images (WSIs) in CAMELYON16 are extremely large, and processing every patch uniformly in a ViT is computationally expensive. To address this, a CNN generates heatmaps to identify regions-of-interest (ROIs). Only the most relevant image patches are tokenized and passed to the ViT, improving both speed and efficiency without sacrificing diagnostic accuracy.

### Key Features:
- Focused token selection using CNN-generated ROI heatmaps  
- ViT inference on a reduced set of high-importance patches  
- Faster and lighter computation suitable for clinical applications  
- Full pipeline with modular codebase for easy experimentation
