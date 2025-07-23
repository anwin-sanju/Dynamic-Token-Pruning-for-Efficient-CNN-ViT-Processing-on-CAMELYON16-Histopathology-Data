"""
Base configuration for CNN-ViT token pruning experiments.
"""
import torch
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class BaseConfig:
    # Dataset configuration
    dataset: str = "cifar10"
    data_dir: str = "data/raw/cifar10"
    num_classes: int = 10
    batch_size: int = 64
    num_workers: int = 4
    
    # Training configuration
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    
    # Device configuration (M1 MacBook optimized)
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    mixed_precision: bool = False  # MPS doesn't support AMP yet
    
    # Logging and checkpointing
    log_interval: int = 50
    save_interval: int = 10
    checkpoint_dir: str = "results"
    use_tensorboard: bool = True
    use_wandb: bool = False
    
    # Model-specific settings
    img_size: int = 32  # CIFAR-10 image size
    patch_size: int = 4  # For ViT patch creation
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.device == "mps":
            print("✅ Using MPS (Apple Silicon) acceleration")
        else:
            print("⚠️ Using CPU - consider GPU acceleration for training")