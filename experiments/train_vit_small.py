"""
Training script for ViT-Small baseline.
This establishes the Vision Transformer performance benchmark for comparison.
"""
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.model_configs import ViTConfig
from models.model_factory import create_model
from data.datasets import create_data_loaders
from utils.training_utils import (
    TrainingUtils, MetricsTracker, EarlyStopping, setup_training_environment
)
from evaluation.metrics import ComprehensiveEvaluator

class ViTSmallTrainer:
    """Trainer for ViT-Small baseline"""
    
    def __init__(self, config: ViTConfig):
        self.config = config
        self.device = config.device
        
        # Setup training environment
        self.env = setup_training_environment(config, "vit_small")
        
        # Create model
        self.model = create_model("vit_small", config)
        self.model.to(self.device)
        
        # Create data loaders with patch-based processing
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Setup training components
        self.optimizer = TrainingUtils.create_optimizer(self.model, config)
        self.scheduler = TrainingUtils.create_scheduler(self.optimizer, config)
        self.criterion = TrainingUtils.create_loss_function(config)
        
        # Setup evaluation
        self.evaluator = ComprehensiveEvaluator(
            num_classes=config.num_classes,
            device=self.device
        )
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=20,  # ViT may need more patience
            min_delta=0.001,
            restore_best_weights=True
        )
        
        # Training state
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        print(f"✅ ViT-Small Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Training samples: {len(self.train_loader.dataset)}")
        print(f"   Validation samples: {len(self.val_loader.dataset)}")
        print(f"   Patches per image: {config.patch_size}×{config.patch_size} → {(config.img_size//config.patch_size)**2} tokens")
    
    def _create_data_loaders(self):
        """Create data loaders with patch-based processing for ViT"""
        # Disable pin_memory for MPS devices
        if self.device == 'mps':
            original_pin_memory = getattr(self.config, 'pin_memory', True)
            self.config.pin_memory = False
        
        train_loader, val_loader = create_data_loaders(self.config, "patch")
        
        # Restore original setting
        if self.device == 'mps':
            self.config.pin_memory = original_pin_memory
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch"""
        self.model.train()
        train_metrics = MetricsTracker()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.epochs} [Train]",
            leave=False
        )
        
        for batch_idx, batch_data in enumerate(progress_bar):
            batch_start_time = time.time()
            
            # Handle patch-based data format
            if isinstance(batch_data, dict):
                # Patch-based format from PatchCIFAR10
                images = batch_data['image'].to(self.device)
                targets = batch_data['target'].to(self.device)
            else:
                # Standard format
                images, targets = batch_data
                images, targets = images.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for ViT stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate batch time
            if self.device == 'mps':
                torch.mps.synchronize()
            elif self.device.startswith('cuda'):
                torch.cuda.synchronize()
            
            batch_time = time.time() - batch_start_time
            
            # Update metrics
            train_metrics.update(
                loss=loss.item(),
                outputs=outputs,
                targets=targets,
                batch_time=batch_time
            )
            
            # Update progress bar
            current_metrics = train_metrics.get_metrics()
            progress_bar.set_postfix({
                'Loss': f"{current_metrics.get('loss', 0):.4f}",
                'Acc': f"{current_metrics.get('accuracy', 0):.3f}"
            })
            
            # Log batch metrics
            if batch_idx % self.config.log_interval == 0 and self.env['writer']:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.env['writer'].add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        return train_metrics.get_metrics()
    
    def validate_epoch(self, epoch: int) -> dict:
        """Validate for one epoch"""
        self.model.eval()
        self.evaluator.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1}/{self.config.epochs} [Val]",
            leave=False
        )
        
        with torch.no_grad():
            for batch_data in progress_bar:
                # Handle patch-based data format
                if isinstance(batch_data, dict):
                    images = batch_data['image'].to(self.device)
                    targets = batch_data['target'].to(self.device)
                else:
                    images, targets = batch_data
                    images, targets = images.to(self.device), targets.to(self.device)
                
                # Time the batch
                batch_start_time = time.time()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Calculate batch time
                if self.device == 'mps':
                    torch.mps.synchronize()
                elif self.device.startswith('cuda'):
                    torch.cuda.synchronize()
                
                batch_time = time.time() - batch_start_time
                
                # Update evaluator
                self.evaluator.update_batch(
                    outputs=outputs,
                    targets=targets,
                    batch_size=images.size(0),
                    batch_time=batch_time
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}"
                })
        
        # Compute comprehensive metrics
        all_metrics = self.evaluator.compute_all_metrics()
        
        # Add validation loss
        val_metrics = {
            'val_loss': total_loss / num_batches,
            'val_accuracy': all_metrics.get('performance', {}).get('accuracy', 0),
            'val_f1': all_metrics.get('performance', {}).get('f1_macro', 0),
            'efficiency': all_metrics.get('efficiency', {})
        }
        
        return val_metrics, all_metrics
    
    def _log_metrics_safely(self, writer, metrics: dict, step: int, phase: str):
        """Safely log metrics to TensorBoard"""
        for name, value in metrics.items():
            try:
                if isinstance(value, (int, float)):
                    writer.add_scalar(f'{phase}/{name}', value, step)
                elif torch.is_tensor(value) and value.ndim == 0:
                    writer.add_scalar(f'{phase}/{name}', value.item(), step)
                elif isinstance(value, dict):
                    for sub_name, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            writer.add_scalar(f'{phase}/{name}_{sub_name}', sub_value, step)
                        elif torch.is_tensor(sub_value) and sub_value.ndim == 0:
                            writer.add_scalar(f'{phase}/{name}_{sub_name}', sub_value.item(), step)
            except Exception as e:
                print(f"Warning: Could not log metric {name}: {e}")
    
    def train(self):
        """Complete training loop"""
        print(f"\n🚀 Starting ViT-Small training for {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_metrics = self.train_epoch(epoch)
            
            # Validation phase
            val_metrics, comprehensive_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Record history
            self.training_history['train_loss'].append(train_metrics.get('loss', 0))
            self.training_history['train_acc'].append(train_metrics.get('accuracy', 0))
            self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['val_acc'].append(val_metrics['val_accuracy'])
            self.training_history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Log to tensorboard
            if self.env['writer']:
                self._log_metrics_safely(
                    self.env['writer'],
                    train_metrics,
                    epoch,
                    'train'
                )
                self._log_metrics_safely(
                    self.env['writer'],
                    val_metrics,
                    epoch,
                    'val'
                )
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.epochs}:")
            print(f"  Train - Loss: {train_metrics.get('loss', 0):.4f}, "
                  f"Acc: {train_metrics.get('accuracy', 0):.4f}")
            print(f"  Val   - Loss: {val_metrics['val_loss']:.4f}, "
                  f"Acc: {val_metrics['val_accuracy']:.4f}")
            
            # Check for best model
            is_best = val_metrics['val_accuracy'] > self.best_accuracy
            if is_best:
                self.best_accuracy = val_metrics['val_accuracy']
                print(f"  🎉 New best accuracy: {self.best_accuracy:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                TrainingUtils.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    loss=val_metrics['val_loss'],
                    metrics=val_metrics,
                    checkpoint_dir=str(self.env['checkpoint_dir']),
                    model_name="vit_small",
                    is_best=is_best
                )
            
            # Early stopping check
            if self.early_stopping(val_metrics['val_accuracy'], self.model):
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
        
        # Final evaluation
        print(f"\n✅ Training completed!")
        print(f"Best validation accuracy: {self.best_accuracy:.4f}")
        
        # Save final results
        self.save_training_results(comprehensive_metrics)
        
        # Close tensorboard writer
        if self.env['writer']:
            self.env['writer'].close()
    
    def save_training_results(self, final_metrics: dict):
        """Save training results and metrics"""
        results = {
            'model_name': 'vit_small',
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'best_accuracy': self.best_accuracy,
            'final_metrics': final_metrics,
            'model_info': self.model.get_model_info()
        }
        
        # Save to JSON
        results_path = Path(self.config.checkpoint_dir) / "vit_small" / "metrics.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(results_path, 'w') as f:
            def convert_numpy(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                return obj
            
            json.dump(results, f, indent=2, default=convert_numpy)
        
        print(f"📊 Results saved to: {results_path}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ViT-Small baseline')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (smaller for ViT)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate (ViT optimal)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Create configuration
    config = ViTConfig()
    
    # Override with command line arguments
    if hasattr(args, 'epochs') and args.epochs != 50:
        config.epochs = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size != 32:
        config.batch_size = args.batch_size
    if hasattr(args, 'lr') and args.lr != 5e-4:
        config.learning_rate = args.lr
    if hasattr(args, 'device') and args.device != 'auto':
        config.device = args.device
    
    # Create and run trainer
    trainer = ViTSmallTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
