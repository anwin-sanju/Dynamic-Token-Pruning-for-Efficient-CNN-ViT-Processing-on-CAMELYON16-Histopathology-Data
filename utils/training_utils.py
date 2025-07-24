"""
Training utilities for CNN-ViT token pruning experiments.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

class TrainingUtils:
    """Collection of training utility functions"""
    
    @staticmethod
    def create_optimizer(model: nn.Module, config) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        optimizer_name = getattr(config, 'optimizer', 'adamw').lower()
        lr = getattr(config, 'learning_rate', 1e-3)
        weight_decay = getattr(config, 'weight_decay', 1e-4)
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizer
    
    @staticmethod
    def create_scheduler(optimizer: optim.Optimizer, config) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        scheduler_name = getattr(config, 'scheduler', 'cosine').lower()
        epochs = getattr(config, 'epochs', 50)
        
        if scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=epochs // 3,
                gamma=0.1
            )
        elif scheduler_name == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=10,
                factor=0.5,
                verbose=True
            )
        elif scheduler_name == 'none':
            scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        return scheduler
    
    @staticmethod
    def create_loss_function(config) -> nn.Module:
        """Create loss function based on configuration"""
        num_classes = getattr(config, 'num_classes', 10)
        
        # For multi-class classification
        if num_classes > 2:
            return nn.CrossEntropyLoss()
        else:
            # For binary classification (medical imaging)
            return nn.BCEWithLogitsLoss()
    
    @staticmethod
    def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler: Optional[optim.lr_scheduler._LRScheduler],
                       epoch: int, loss: float, metrics: Dict[str, float],
                       checkpoint_dir: str, model_name: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'metrics': metrics,
            'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"{model_name}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / f"{model_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model: {best_path}")
        
        return checkpoint_path
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str, model: nn.Module, 
                       optimizer: Optional[optim.Optimizer] = None,
                       scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                       device: str = 'cpu') -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"📂 Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    @staticmethod
    def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate classification accuracy"""
        if outputs.dim() > 1 and outputs.size(1) > 1:
            # Multi-class classification
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == targets).sum().item()
            total = targets.size(0)
        else:
            # Binary classification
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct = (predicted.squeeze() == targets.float()).sum().item()
            total = targets.size(0)
        
        return correct / total
    
    @staticmethod
    def log_metrics(writer: SummaryWriter, metrics: Dict[str, float], 
                   step: int, phase: str = 'train'):
        """Log metrics to tensorboard"""
        for name, value in metrics.items():
            writer.add_scalar(f'{phase}/{name}', value, step)
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time duration"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    @staticmethod
    def count_model_flops(model: nn.Module, input_size: Tuple[int, ...], device: str = 'cpu') -> int:
        """Estimate model FLOPs (simplified)"""
        model.eval()
        model = model.to(device)
        
        def flop_count_hook(module, input, output):
            if isinstance(module, nn.Conv2d):
                # Conv2D FLOPs: batch_size * output_height * output_width * kernel_ops
                batch_size = input[0].size(0)
                output_dims = output.shape[2:]  # H, W
                kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                output_elements = batch_size * np.prod(output_dims)
                flops = kernel_ops * output_elements
                module.__flops__ = flops
            elif isinstance(module, nn.Linear):
                # Linear FLOPs: batch_size * in_features * out_features
                flops = input[0].numel() * module.out_features
                module.__flops__ = flops
            elif isinstance(module, nn.MultiheadAttention):
                # Attention FLOPs (simplified): batch_size * seq_len^2 * embed_dim
                if len(input) > 0:
                    batch_size, seq_len, embed_dim = input[0].shape
                    flops = batch_size * seq_len * seq_len * embed_dim * 2  # Q*K + Attn*V
                    module.__flops__ = flops
        
        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.MultiheadAttention)):
                hook = module.register_forward_hook(flop_count_hook)
                hooks.append(hook)
        
        # Forward pass with dummy input
        dummy_input = torch.randn(1, *input_size, device=device)
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Sum FLOPs
        total_flops = 0
        for module in model.modules():
            if hasattr(module, '__flops__'):
                total_flops += module.__flops__
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return total_flops

class MetricsTracker:
    """Track and compute training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.losses = []
        self.accuracies = []
        self.predictions = []
        self.targets = []
        self.batch_times = []
        self.start_time = time.time()
    
    def update(self, loss: float, outputs: torch.Tensor, targets: torch.Tensor, 
               batch_time: float):
        """Update metrics with batch results"""
        self.losses.append(loss)
        self.batch_times.append(batch_time)
        
        # Calculate accuracy
        accuracy = TrainingUtils.calculate_accuracy(outputs, targets)
        self.accuracies.append(accuracy)
        
        # Store predictions and targets for detailed analysis
        if outputs.dim() > 1 and outputs.size(1) > 1:
            # Multi-class
            _, predicted = torch.max(outputs, 1)
            self.predictions.extend(predicted.cpu().numpy())
        else:
            # Binary
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            self.predictions.extend(predicted.cpu().numpy())
        
        self.targets.extend(targets.cpu().numpy())
    
    def get_metrics(self) -> Dict[str, float]:
        """Get computed metrics"""
        if not self.losses:
            return {}
        
        total_time = time.time() - self.start_time
        
        return {
            'loss': np.mean(self.losses),
            'accuracy': np.mean(self.accuracies),
            'total_time': total_time,
            'avg_batch_time': np.mean(self.batch_times),
            'samples_per_second': len(self.losses) / total_time if total_time > 0 else 0
        }
    
    def get_confusion_matrix(self) -> Optional[np.ndarray]:
        """Get confusion matrix"""
        if not self.predictions or not self.targets:
            return None
        
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(self.targets, self.predictions)

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """Check if should stop early"""
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                    print("🔄 Restored best weights")
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        
        return self.early_stop
    
    def save_checkpoint(self, model: nn.Module):
        """Save best model weights"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

def setup_training_environment(config, model_name: str) -> Dict[str, Any]:
    """Setup complete training environment"""
    # Create directories
    checkpoint_dir = Path(config.checkpoint_dir) / model_name / "checkpoints"
    log_dir = Path(config.checkpoint_dir) / model_name / "logs"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard writer
    writer = SummaryWriter(log_dir) if config.use_tensorboard else None
    
    # Setup wandb if configured
    wandb_run = None
    if config.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="cnn-vit-token-pruning",
                name=f"{model_name}-{int(time.time())}",
                config=config.__dict__
            )
        except ImportError:
            print("Warning: wandb not installed, skipping wandb logging")
    
    return {
        'checkpoint_dir': checkpoint_dir,
        'log_dir': log_dir,
        'writer': writer,
        'wandb_run': wandb_run
    }
