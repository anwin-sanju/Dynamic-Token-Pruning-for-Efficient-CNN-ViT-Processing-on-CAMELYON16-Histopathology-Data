# CNN-ViT Dynamic Token Pruning for Medical Image Analysis

A research implementation comparing CNN, Vision Transformer (ViT), and CNN-ViT hybrid architectures with dynamic token pruning for efficient medical image classification.

## Project Overview

This project implements and compares three model architectures for image classification:

- **ResNet18**: CNN baseline for performance comparison
- **ViT-Small**: Vision Transformer baseline 
- **CNN-ViT Hybrid**: Novel architecture with CNN-guided dynamic token pruning

### Research Goals

Demonstrate that CNN-ViT dynamic token pruning:
- Maintains ViT-level accuracy (~97-98% vs CNN ~95%)
- Reduces computational cost through intelligent token selection
- Achieves superior accuracy-efficiency trade-offs

## Quick Start

### Environment Setup

1. **Create conda virtual environment:**
conda create -n cnn-vit python=3.9 -y
conda activate cnn-vit

text

2. **Install dependencies:**
pip install -r requirements.txt

text

3. **Verify installation:**
python -c "import torch, timm; print('Setup complete!')"

text

### Training Models

1. **Train CNN baseline:**
python experiments/train_resnet18.py

text

2. **Train ViT baseline:**
python experiments/train_vit_small.py

text

3. **Train CNN-ViT hybrid:**
python experiments/train_hybrid.py

text

4. **Compare all models:**
python experiments/compare_models.py

text

## Development Workflow

### Prototype Phase (Current)
- **Dataset**: CIFAR-10 for rapid prototyping
- **Hardware**: M1 MacBook Air with MPS acceleration
- **Focus**: Algorithm validation and comparative analysis

### Production Phase (Future)
- **Dataset**: CAMELYON16/17 whole-slide images (WSI)
- **Preprocessing**: Patch extraction with tissue segmentation
- **Scaling**: GPU cluster deployment for large-scale evaluation

## Technical Implementation

### CNN-ViT Hybrid Architecture

1. **Stage 1: CNN ROI Scoring**
- ResNet18 predicts importance scores for image patches
- Dual-head architecture: classification + importance prediction

2. **Stage 2: Dynamic Token Selection**
- Top-k selection based on CNN importance scores
- Configurable pruning ratios (20%-60% token reduction)

3. **Stage 3: ViT Inference**
- ViT-Small processes selected tokens only
- Maintains spatial relationships with positional embeddings

### Training Strategy

- **Two-stage training**: CNN pre-training → End-to-end fine-tuning
- **Loss function**: Classification loss + token efficiency penalty
- **Optimization**: AdamW with cosine annealing schedule

## Expected Results

| Model | Accuracy | Inference Speed | Memory Usage | Token Reduction |
|-------|----------|-----------------|--------------|-----------------|
| ResNet18 | ~95% | High | Low | N/A |
| ViT-Small | ~98% | Medium | High | 0% |
| CNN-ViT Hybrid | ~97-98% | High | Medium | 40-60% |

## Development Notes

### Hardware Optimization
- **M1 MacBook Air**: Optimized for Apple Silicon with MPS backend
- **Memory Management**: Efficient handling of 8GB unified memory
- **Batch Processing**: Adaptive batch sizes for hardware constraints

### Dependencies
- **PyTorch**: Deep learning framework with MPS support
- **timm**: Vision Transformer implementations
- **OpenSlide**: WSI preprocessing capabilities (future scaling)
- **Albumentations**: Advanced image augmentations

## Research Context

This implementation supports research into:
- **Efficient Vision Transformers**: Reducing computational overhead
- **Medical Image Analysis**: Application to histopathology classification
- **Hybrid Architectures**: Combining CNN and ViT strengths
- **Dynamic Token Pruning**: Adaptive attention mechanisms

## Contributing

This is a research prototype. Key areas for contribution:
- Model architecture improvements
- Evaluation metric enhancements
- Visualization tools for attention analysis
- Documentation and code cleanup

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

@misc{cnn_vit_token_pruning,
title={CNN-ViT Dynamic Token Pruning for Efficient Medical Image Analysis},
author={Your Name},
year={2025},
url={https://github.com/yourusername/cnn-vit-token-pruning}
}

text

---

**Status**: Active Development | **Phase**: Prototype Implementation | **Target**: Research Publication
