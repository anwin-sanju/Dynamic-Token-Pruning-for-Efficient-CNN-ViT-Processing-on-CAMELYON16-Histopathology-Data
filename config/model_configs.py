"""
Model-specific configurations for different architectures.
"""
from dataclasses import dataclass
from .base_config import BaseConfig

@dataclass
class ResNet18Config(BaseConfig):
    """Configuration for ResNet18 CNN baseline"""
    model_name: str = "resnet18"
    pretrained: bool = True
    dropout: float = 0.1
    
    # ResNet-specific settings
    modify_for_cifar: bool = True  # Modify conv1 and remove maxpool for 32x32 images

@dataclass
class ViTConfig(BaseConfig):
    """Configuration for ViT-Small baseline"""
    model_name: str = "vit_small"
    pretrained: bool = True
    
    # ViT architecture settings
    patch_size: int = 4  # 4x4 patches for 32x32 CIFAR-10
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    dropout: float = 0.1
    
    # Training adjustments for small images
    pos_embed_type: str = "learned"

@dataclass
class HybridConfig(BaseConfig):
    """Configuration for CNN-ViT Hybrid with dynamic token pruning"""
    model_name: str = "cnn_vit_hybrid"
    
    # CNN ROI scorer settings
    cnn_pretrained: bool = True
    freeze_cnn_stage2: bool = True  # Freeze CNN during stage 2 training
    
    # Token pruning settings
    pruning_ratio: float = 0.5  # Prune 50% of tokens
    selection_method: str = "top_k"  # "top_k" or "threshold"
    min_tokens: int = 16  # Minimum tokens to keep
    
    # ViT backbone settings
    vit_pretrained: bool = True
    patch_size: int = 4
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    
    # Two-stage training strategy
    two_stage_training: bool = True
    stage1_epochs: int = 20  # CNN ROI scorer pre-training
    stage2_epochs: int = 30  # End-to-end fine-tuning
    
    # Loss weighting
    classification_weight: float = 1.0
    efficiency_weight: float = 0.1  # Weight for token reduction penalty

def get_config(model_name: str) -> BaseConfig:
    """Factory function to get configuration by model name"""
    configs = {
        "resnet18": ResNet18Config,
        "vit_small": ViTConfig,
        "cnn_vit_hybrid": HybridConfig
    }
    
    if model_name not in configs:
        raise ValueError(f"Model {model_name} not supported. Available: {list(configs.keys())}")
    
    return configs[model_name]()
