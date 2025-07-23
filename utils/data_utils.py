"""
Data utility functions for CNN-ViT token pruning experiments.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

def get_dataset_info(dataset_name: str = "cifar10") -> Dict:
    """Get dataset information and statistics"""
    if dataset_name == "cifar10":
        return {
            'num_classes': 10,
            'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck'],
            'input_size': (3, 32, 32),
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010],
            'num_train_samples': 50000,
            'num_test_samples': 10000
        }
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

def calculate_patches_per_image(img_size: int, patch_size: int) -> int:
    """Calculate number of patches per image for ViT processing"""
    if img_size % patch_size != 0:
        raise ValueError(f"Image size {img_size} must be divisible by patch size {patch_size}")
    return (img_size // patch_size) ** 2

def extract_patches(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Extract non-overlapping patches from batch of images
    
    Args:
        images: Tensor of shape [batch_size, channels, height, width]
        patch_size: Size of square patches to extract
        
    Returns:
        patches: Tensor of shape [batch_size, num_patches, channels, patch_size, patch_size]
    """
    batch_size, channels, height, width = images.shape
    
    if height != width:
        raise ValueError("Only square images are supported")
    
    if height % patch_size != 0:
        raise ValueError(f"Image size {height} must be divisible by patch size {patch_size}")
    
    num_patches_per_side = height // patch_size
    num_patches = num_patches_per_side ** 2
    
    # Reshape image into patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(batch_size, num_patches, channels, patch_size, patch_size)
    
    return patches

def simulate_importance_scores(batch_size: int, num_patches: int, 
                             device: str = "cpu", strategy: str = "random") -> torch.Tensor:
    """
    Simulate CNN importance scores for testing (will be replaced by actual CNN predictions)
    
    Args:
        batch_size: Number of images in batch
        num_patches: Number of patches per image
        device: Device to create tensor on
        strategy: Simulation strategy ("random", "center_bias", "corner_bias")
        
    Returns:
        importance_scores: Tensor of shape [batch_size, num_patches]
    """
    if strategy == "random":
        scores = torch.rand(batch_size, num_patches, device=device)
    elif strategy == "center_bias":
        # Simulate higher importance for center patches
        scores = torch.rand(batch_size, num_patches, device=device)
        # Assuming 8x8 patches for 32x32 image with patch_size=4
        center_patches = [27, 28, 35, 36]  # Center 2x2 patches in 8x8 grid
        for idx in center_patches:
            if idx < num_patches:
                scores[:, idx] += 0.3
    elif strategy == "corner_bias":
        # Simulate higher importance for corner patches
        scores = torch.rand(batch_size, num_patches, device=device)
        corner_patches = [0, 7, 56, 63]  # Corners in 8x8 grid
        for idx in corner_patches:
            if idx < num_patches:
                scores[:, idx] += 0.3
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return scores

def select_top_k_patches(patches: torch.Tensor, importance_scores: torch.Tensor, 
                        k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k patches based on importance scores
    
    Args:
        patches: Tensor of shape [batch_size, num_patches, channels, patch_size, patch_size]
        importance_scores: Tensor of shape [batch_size, num_patches]
        k: Number of top patches to select
        
    Returns:
        selected_patches: Tensor of shape [batch_size, k, channels, patch_size, patch_size]
        selected_indices: Tensor of shape [batch_size, k]
    """
    batch_size, num_patches = importance_scores.shape
    
    if k > num_patches:
        raise ValueError(f"k ({k}) cannot be larger than num_patches ({num_patches})")
    
    # Get top-k indices for each image in batch
    _, top_k_indices = torch.topk(importance_scores, k, dim=1)
    
    # Select corresponding patches
    batch_indices = torch.arange(batch_size, device=patches.device).unsqueeze(1).expand(-1, k)
    selected_patches = patches[batch_indices, top_k_indices]
    
    return selected_patches, top_k_indices

def select_threshold_patches(patches: torch.Tensor, importance_scores: torch.Tensor, 
                           threshold: float, min_patches: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select patches above importance threshold
    
    Args:
        patches: Tensor of shape [batch_size, num_patches, channels, patch_size, patch_size]
        importance_scores: Tensor of shape [batch_size, num_patches]
        threshold: Minimum importance score to keep patch
        min_patches: Minimum number of patches to keep per image
        
    Returns:
        selected_patches: List of tensors (variable length per batch item)
        selected_indices: List of tensors (variable length per batch item)
        num_selected: Tensor of shape [batch_size] with number of selected patches per image
    """
    batch_size = importance_scores.shape[0]
    selected_patches = []
    selected_indices = []
    num_selected = []
    
    for i in range(batch_size):
        # Find patches above threshold
        mask = importance_scores[i] >= threshold
        indices = torch.where(mask)[0]
        
        # Ensure minimum number of patches
        if len(indices) < min_patches:
            _, top_indices = torch.topk(importance_scores[i], min_patches)
            indices = top_indices
        
        selected_patches.append(patches[i, indices])
        selected_indices.append(indices)
        num_selected.append(len(indices))
    
    num_selected = torch.tensor(num_selected, device=patches.device)
    
    return selected_patches, selected_indices, num_selected

def calculate_token_reduction_stats(original_tokens: int, selected_tokens: torch.Tensor) -> Dict:
    """
    Calculate token reduction statistics
    
    Args:
        original_tokens: Original number of tokens per image
        selected_tokens: Tensor with number of selected tokens per image
        
    Returns:
        stats: Dictionary with reduction statistics
    """
    avg_selected = selected_tokens.float().mean().item()
    reduction_rate = (original_tokens - avg_selected) / original_tokens
    
    return {
        'original_tokens': original_tokens,
        'avg_selected_tokens': avg_selected,
        'reduction_rate': reduction_rate,
        'reduction_percentage': reduction_rate * 100,
        'min_selected': selected_tokens.min().item(),
        'max_selected': selected_tokens.max().item(),
        'std_selected': selected_tokens.float().std().item()
    }
