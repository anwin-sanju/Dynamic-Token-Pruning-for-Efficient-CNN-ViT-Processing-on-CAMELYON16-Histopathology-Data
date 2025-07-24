"""
Model factory for creating different architectures in the CNN-ViT token pruning project.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Union
from .resnet18 import ResNet18Baseline, create_resnet_model
from .vit_small import ViTSmallBaseline, ViTSmallCustom, create_vit_model
from .cnn_vit_hybrid import CNNViTHybrid, create_hybrid_model

def create_model(model_name: str, config) -> nn.Module:
    """
    Factory function to create models based on configuration
    
    Args:
        model_name: Name of model to create ('resnet18', 'vit_small', 'cnn_vit_hybrid')
        config: Configuration object (BaseConfig or subclass)
        
    Returns:
        Initialized model
    """
    model_creators = {
        'resnet18': lambda cfg: create_resnet_model('baseline', cfg),
        'vit_small': lambda cfg: create_vit_model('custom', cfg),
        'cnn_vit_hybrid': lambda cfg: create_hybrid_model(cfg)
    }
    
    if model_name not in model_creators:
        raise ValueError(f"Model {model_name} not supported. Available: {list(model_creators.keys())}")
    
    model = model_creators[model_name](config)
    
    print(f"✅ Created {model_name} model")
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print(f"   Parameters: {info.get('total_parameters', 'N/A'):,}")
        if 'efficiency_metrics' in info:
            metrics = info['efficiency_metrics']
            print(f"   Token reduction: {metrics.get('token_reduction_rate', 0)*100:.1f}%")
    
    return model

def get_model_comparison_info() -> Dict[str, Dict[str, Any]]:
    """Get comparison information for all three models"""
    return {
        'resnet18': {
            'type': 'CNN Baseline',
            'description': 'Standard ResNet18 for comparison',
            'expected_params': '~11.2M',
            'token_processing': 'Full image',
            'strengths': ['Fast inference', 'Low memory', 'Strong local features'],
            'use_case': 'Baseline comparison and CNN component'
        },
        'vit_small': {
            'type': 'ViT Baseline', 
            'description': 'Vision Transformer baseline',
            'expected_params': '~21.3M',
            'token_processing': 'All 64 patches',
            'strengths': ['Global context', 'Self-attention', 'High accuracy'],
            'use_case': 'Baseline comparison and ViT component'
        },
        'cnn_vit_hybrid': {
            'type': 'Novel Hybrid',
            'description': 'CNN-guided dynamic token pruning',
            'expected_params': '~32.5M (CNN + ViT)',
            'token_processing': 'Selected patches only',
            'strengths': ['Efficient inference', 'Adaptive processing', 'Best of both worlds'],
            'use_case': 'Main research contribution'
        }
    }

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def get_model_memory_usage(model: nn.Module, input_size: tuple = (1, 3, 32, 32), 
                          device: str = 'cpu') -> Dict[str, float]:
    """Estimate model memory usage"""
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_size, device=device)
    
    # Measure memory before forward pass
    if device == 'mps':
        torch.mps.empty_cache()
        memory_before = torch.mps.current_allocated_memory()
    elif device.startswith('cuda'):
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated()
    else:
        memory_before = 0
    
    # Forward pass
    with torch.no_grad():
        if hasattr(model, 'forward') and 'return_attention' in model.forward.__code__.co_varnames:
            # For hybrid model
            output = model(dummy_input)
        else:
            output = model(dummy_input)
    
    # Measure memory after forward pass
    if device == 'mps':
        memory_after = torch.mps.current_allocated_memory()
    elif device.startswith('cuda'):
        memory_after = torch.cuda.memory_allocated()
    else:
        memory_after = 0
    
    memory_used = (memory_after - memory_before) / 1024 / 1024  # Convert to MB
    
    return {
        'forward_pass_memory_mb': memory_used,
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024,
        'device': device
    }
