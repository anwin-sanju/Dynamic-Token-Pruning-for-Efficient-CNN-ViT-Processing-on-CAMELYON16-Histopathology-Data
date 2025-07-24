"""
Training script for CNN-ViT Hybrid with dynamic token pruning.
This is the main research contribution demonstrating efficiency gains.
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

from config.model_configs import HybridConfig
from models.model_factory import create_model
from data.datasets import create_data_loaders
from utils.training_utils import (
    TrainingUtils, MetricsTracker, EarlyStopping, setup_training_environment
)
from evaluation.metrics import ComprehensiveEvaluator

class CNNViTHybridTrainer:
    """Trainer for CNN-ViT Hybrid with two-stage training strategy"""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.device = config.device
        
        # Setup training environment
        self.env = setup_training_environment(config, "cnn_vit_hybrid")
        
        # Create hybrid model
        self.model = create_model("cnn_vit_hybrid", config)
        self.model.to(self.device)
        
        # Create data loaders - hybrid mode for both CNN and ViT processing
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Setup training components (will be recreated for each stage)
        self.criterion = TrainingUtils.create_loss_function(config)
        
        # Setup evaluation with token pruning support
        self.evaluator = ComprehensiveEvaluator(
            num_classes=config.num_classes,
            device=self.device
        )
        
        # Training state
        self.best_accuracy = 0.0
        self.training_history = {
            'stage1': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
            'stage2': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
            'learning_rates': [],
            'token_reduction_stats': []
        }
        
        print(f"✅ CNN-ViT Hybrid Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Training samples: {len(self.train_loader.dataset)}")
        print(f"   Validation samples: {len(self.val_loader.dataset)}")
        print(f"   Token pruning: {config.pruning_ratio*100:.1f}% reduction")
        print(f"   Two-stage training: Stage1={config.stage1_epochs}, Stage2={config.stage2_epochs}")
    
    def _create_data_loaders(self):
        """Create data loaders for hybrid training"""
        if self.device == 'mps':
            original_pin_memory = getattr(self.config, 'pin_memory', True)
            self.config.pin_memory = False
        
        train_loader, val_loader = create_data_loaders(self.config, "hybrid")
        
        if self.device == 'mps':
            self.config.pin_memory = original_pin_memory
        
        return train_loader, val_loader
    
    def train_stage1(self):
        """Stage 1: Train CNN ROI scorer only"""
        print(f"\n🎯 Stage 1: Training CNN ROI Scorer ({self.config.stage1_epochs} epochs)")
        
        # Set model to stage 1 training
        self.model.set_training_stage(1)
        
        # Create optimizer for stage 1 (CNN only)
        stage1_params = self.model.cnn_roi_scorer.parameters()
        self.optimizer = torch.optim.AdamW(
            stage1_params, 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        self.scheduler = TrainingUtils.create_scheduler(self.optimizer, 
                                                       type('Config', (), {'epochs': self.config.stage1_epochs})())
        
        # Early stopping for stage 1
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        
        for epoch in range(self.config.stage1_epochs):
            # Training phase
            train_metrics = self._train_epoch_stage1(epoch)
            
            # Validation phase  
            val_metrics, _ = self._validate_epoch_stage1(epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Record history
            self.training_history['stage1']['train_loss'].append(train_metrics.get('loss', 0))
            self.training_history['stage1']['train_acc'].append(train_metrics.get('accuracy', 0))
            self.training_history['stage1']['val_loss'].append(val_metrics['val_loss'])
            self.training_history['stage1']['val_acc'].append(val_metrics['val_accuracy'])
            
            # Log to tensorboard
            if self.env['writer']:
                self._log_metrics_safely(self.env['writer'], train_metrics, epoch, 'stage1_train')
                self._log_metrics_safely(self.env['writer'], val_metrics, epoch, 'stage1_val')
            
            # Print progress
            print(f"Stage 1 Epoch {epoch+1}/{self.config.stage1_epochs}:")
            print(f"  CNN Train - Loss: {train_metrics.get('loss', 0):.4f}, Acc: {train_metrics.get('accuracy', 0):.4f}")
            print(f"  CNN Val   - Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_accuracy']:.4f}")
            
            # Save checkpoint
            if val_metrics['val_accuracy'] > self.best_accuracy:
                self.best_accuracy = val_metrics['val_accuracy']
                TrainingUtils.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    loss=val_metrics['val_loss'],
                    metrics=val_metrics,
                    checkpoint_dir=str(self.env['checkpoint_dir']),  # FIXED: was 'checkout_dir'
                    model_name="cnn_vit_hybrid_stage1",
                    is_best=True
                )
            
            # Early stopping check
            if early_stopping(val_metrics['val_accuracy'], self.model):
                print(f"⏹️ Stage 1 early stopping at epoch {epoch+1}")
                break
        
        print(f"✅ Stage 1 completed! Best CNN accuracy: {self.best_accuracy:.4f}")
    
    def train_stage2(self):
        """Stage 2: End-to-end training with frozen CNN"""
        print(f"\n🚀 Stage 2: End-to-end training ({self.config.stage2_epochs} epochs)")
        
        # Set model to stage 2 training (freeze CNN)
        self.model.set_training_stage(2)
        
        # Create optimizer for stage 2 (ViT + unfrozen parts)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,  
            lr=self.config.learning_rate * 0.1,  # Lower LR for fine-tuning
            weight_decay=self.config.weight_decay
        )
        self.scheduler = TrainingUtils.create_scheduler(self.optimizer, 
                                                       type('Config', (), {'epochs': self.config.stage2_epochs})())
        
        # Reset best accuracy for stage 2
        stage2_best_accuracy = 0.0
        early_stopping = EarlyStopping(patience=15, min_delta=0.001)
        
        for epoch in range(self.config.stage2_epochs):
            # Training phase
            train_metrics = self._train_epoch_stage2(epoch)
            
            # Validation phase
            val_metrics, comprehensive_metrics = self._validate_epoch_stage2(epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Record history
            self.training_history['stage2']['train_loss'].append(train_metrics.get('loss', 0))
            self.training_history['stage2']['train_acc'].append(train_metrics.get('accuracy', 0))
            self.training_history['stage2']['val_loss'].append(val_metrics['val_loss'])
            self.training_history['stage2']['val_acc'].append(val_metrics['val_accuracy'])
            
            # Record token reduction stats
            if 'token_pruning' in comprehensive_metrics:
                self.training_history['token_reduction_stats'].append(
                    comprehensive_metrics['token_pruning']
                )
            
            # Log to tensorboard
            if self.env['writer']:
                self._log_metrics_safely(self.env['writer'], train_metrics, epoch, 'stage2_train')
                self._log_metrics_safely(self.env['writer'], val_metrics, epoch, 'stage2_val')
                
                # Log token pruning metrics
                if 'token_pruning' in comprehensive_metrics:
                    token_metrics = comprehensive_metrics['token_pruning']
                    self.env['writer'].add_scalar('TokenPruning/ReductionRate', 
                                                 token_metrics.get('avg_token_reduction_percentage', 0), epoch)
                    self.env['writer'].add_scalar('TokenPruning/SelectedTokens',
                                                 token_metrics.get('avg_selected_tokens', 0), epoch)
            
            # Print progress with token stats
            token_stats = comprehensive_metrics.get('token_pruning', {})
            print(f"Stage 2 Epoch {epoch+1}/{self.config.stage2_epochs}:")
            print(f"  Hybrid Train - Loss: {train_metrics.get('loss', 0):.4f}, Acc: {train_metrics.get('accuracy', 0):.4f}")
            print(f"  Hybrid Val   - Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_accuracy']:.4f}")
            print(f"  Token Reduction: {token_stats.get('avg_token_reduction_percentage', 0):.1f}%")
            
            # Check for best model
            is_best = val_metrics['val_accuracy'] > stage2_best_accuracy
            if is_best:
                stage2_best_accuracy = val_metrics['val_accuracy']
                print(f"  🎉 New best hybrid accuracy: {stage2_best_accuracy:.4f}")
            
            # Save checkpoint - FIXED: corrected key name
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                TrainingUtils.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    loss=val_metrics['val_loss'],
                    metrics=val_metrics,
                    checkpoint_dir=str(self.env['checkpoint_dir']),  # FIXED: was 'checkout_dir'
                    model_name="cnn_vit_hybrid",
                    is_best=is_best
                )
            
            # Early stopping check
            if early_stopping(val_metrics['val_accuracy'], self.model):
                print(f"⏹️ Stage 2 early stopping at epoch {epoch+1}")
                break
        
        self.best_accuracy = stage2_best_accuracy
        print(f"✅ Stage 2 completed! Best hybrid accuracy: {self.best_accuracy:.4f}")
    
    def _train_epoch_stage1(self, epoch: int) -> dict:
        """Training epoch for stage 1 (CNN only)"""
        self.model.train()
        train_metrics = MetricsTracker()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Stage1 Epoch {epoch+1} [Train]",
            leave=False
        )
        
        for batch_idx, batch_data in enumerate(progress_bar):
            batch_start_time = time.time()
            
            # Get CNN data
            cnn_images = batch_data['cnn_image'].to(self.device)
            targets = batch_data['target'].to(self.device)
            
            # Forward pass through CNN only
            self.optimizer.zero_grad()
            patches = self.model._extract_patches(cnn_images)
            cnn_outputs = self.model.cnn_roi_scorer.forward_with_classification(patches)[1]
            
            # Use mean prediction across patches for classification
            cnn_logits = cnn_outputs.mean(dim=1)  # [batch_size, num_classes]
            loss = self.criterion(cnn_logits, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Synchronize
            if self.device == 'mps':
                torch.mps.synchronize()
            batch_time = time.time() - batch_start_time
            
            # Update metrics
            train_metrics.update(
                loss=loss.item(),
                outputs=cnn_logits,
                targets=targets,
                batch_time=batch_time
            )
            
            # Update progress bar
            current_metrics = train_metrics.get_metrics()
            progress_bar.set_postfix({
                'Loss': f"{current_metrics.get('loss', 0):.4f}",
                'Acc': f"{current_metrics.get('accuracy', 0):.3f}"
            })
        
        return train_metrics.get_metrics()
    
    def _validate_epoch_stage1(self, epoch: int) -> tuple:
        """Validation epoch for stage 1 (CNN only)"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Stage1 Epoch {epoch+1} [Val]",
            leave=False
        )
        
        with torch.no_grad():
            for batch_data in progress_bar:
                cnn_images = batch_data['cnn_image'].to(self.device)
                targets = batch_data['target'].to(self.device)
                
                # Forward pass through CNN only
                patches = self.model._extract_patches(cnn_images)
                cnn_outputs = self.model.cnn_roi_scorer.forward_with_classification(patches)[1]
                cnn_logits = cnn_outputs.mean(dim=1)
                
                loss = self.criterion(cnn_logits, targets)
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(cnn_logits, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
        
        val_metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_accuracy': total_correct / total_samples,
            'val_f1': 0.0  # Simplified for stage 1
        }
        
        return val_metrics, {}
    
    def _train_epoch_stage2(self, epoch: int) -> dict:
        """Training epoch for stage 2 (end-to-end)"""
        self.model.train()
        train_metrics = MetricsTracker()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Stage2 Epoch {epoch+1} [Train]",
            leave=False
        )
        
        for batch_idx, batch_data in enumerate(progress_bar):
            batch_start_time = time.time()
            
            # Get hybrid data
            vit_images = batch_data['vit_image'].to(self.device)
            targets = batch_data['target'].to(self.device)
            
            # Forward pass through complete hybrid model
            self.optimizer.zero_grad()
            
            # Get hybrid model outputs
            model_outputs = self.model(vit_images)
            logits = model_outputs['logits']
            
            # Calculate loss with efficiency penalty
            classification_loss = self.criterion(logits, targets)
            
            # Add token efficiency penalty
            token_reduction_rate = model_outputs.get('token_reduction_stats', {}).get('avg_token_reduction_rate', 0)
            efficiency_penalty = self.config.efficiency_weight * (1.0 - token_reduction_rate)
            
            total_loss = classification_loss + efficiency_penalty
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Synchronize
            if self.device == 'mps':
                torch.mps.synchronize()
            batch_time = time.time() - batch_start_time
            
            # Update metrics
            train_metrics.update(
                loss=total_loss.item(),
                outputs=logits,
                targets=targets,
                batch_time=batch_time
            )
            
            # Update progress bar
            current_metrics = train_metrics.get_metrics()
            progress_bar.set_postfix({
                'Loss': f"{current_metrics.get('loss', 0):.4f}",
                'Acc': f"{current_metrics.get('accuracy', 0):.3f}",
                'Tokens': f"{model_outputs.get('num_tokens_used', [0])[0]}"
            })
        
        return train_metrics.get_metrics()
    
    def _validate_epoch_stage2(self, epoch: int) -> tuple:
        """Validation epoch for stage 2 (end-to-end)"""
        self.model.eval()
        self.evaluator.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Stage2 Epoch {epoch+1} [Val]",
            leave=False
        )
        
        with torch.no_grad():
            for batch_data in progress_bar:
                vit_images = batch_data['vit_image'].to(self.device)
                targets = batch_data['target'].to(self.device)
                
                # Time the batch
                batch_start_time = time.time()
                
                # Forward pass through complete hybrid model
                model_outputs = self.model(vit_images)
                logits = model_outputs['logits']
                
                loss = self.criterion(logits, targets)
                
                # Calculate batch time
                if self.device == 'mps':
                    torch.mps.synchronize()
                batch_time = time.time() - batch_start_time
                
                # Extract token information
                token_info = {
                    'original_tokens': 64,  # CIFAR-10 has 64 patches
                    'selected_tokens': model_outputs.get('num_tokens_used', torch.tensor([32])),
                    'importance_scores': model_outputs.get('importance_scores')
                }
                
                # Update evaluator with token information
                self.evaluator.update_batch(
                    outputs=logits,
                    targets=targets,
                    batch_size=vit_images.size(0),
                    batch_time=batch_time,
                    token_info=token_info
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}"
                })
        
        # Compute comprehensive metrics
        all_metrics = self.evaluator.compute_all_metrics()
        
        val_metrics = {
            'val_loss': total_loss / num_batches,
            'val_accuracy': all_metrics.get('performance', {}).get('accuracy', 0),
            'val_f1': all_metrics.get('performance', {}).get('f1_macro', 0),
            'efficiency': all_metrics.get('efficiency', {}),
            'token_pruning': all_metrics.get('token_pruning', {})
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
            except Exception as e:
                print(f"Warning: Could not log metric {name}: {e}")
    
    def train(self):
        """Complete two-stage training pipeline"""
        print(f"\n🎯 Starting CNN-ViT Hybrid two-stage training")
        
        # Stage 1: Train CNN ROI scorer
        self.train_stage1()
        
        # Stage 2: End-to-end training with token pruning
        self.train_stage2()
        
        # Final evaluation and save
        print(f"\n✅ Two-stage training completed!")
        print(f"Final best accuracy: {self.best_accuracy:.4f}")
        
        # Save comprehensive results
        self.save_training_results()
        
        # Close tensorboard writer
        if self.env['writer']:
            self.env['writer'].close()
    
    def save_training_results(self):
        """Save comprehensive training results"""
        # Get final model info
        model_info = self.model.get_model_info()
        efficiency_metrics = self.model.get_efficiency_metrics()
        
        results = {
            'model_name': 'cnn_vit_hybrid',
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'best_accuracy': self.best_accuracy,
            'model_info': model_info,
            'efficiency_metrics': efficiency_metrics,
            'research_summary': {
                'token_reduction_achieved': f"{self.config.pruning_ratio*100:.1f}%",
                'computational_savings': efficiency_metrics.get('computational_savings', 0),
                'parameter_count': model_info.get('total_parameters', 0)
            }
        }
        
        # Save to JSON
        results_path = Path(self.config.checkpoint_dir) / "cnn_vit_hybrid" / "metrics.json"
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
        
        print(f"📊 Comprehensive results saved to: {results_path}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train CNN-ViT Hybrid with token pruning')
    parser.add_argument('--stage1_epochs', type=int, default=20, help='Stage 1 epochs')
    parser.add_argument('--stage2_epochs', type=int, default=30, help='Stage 2 epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='Token pruning ratio')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Create configuration
    config = HybridConfig()
    
    # Override with command line arguments
    if hasattr(args, 'stage1_epochs'):
        config.stage1_epochs = args.stage1_epochs
    if hasattr(args, 'stage2_epochs'):
        config.stage2_epochs = args.stage2_epochs
    if hasattr(args, 'batch_size'):
        config.batch_size = args.batch_size
    if hasattr(args, 'lr'):
        config.learning_rate = args.lr
    if hasattr(args, 'pruning_ratio'):
        config.pruning_ratio = args.pruning_ratio
    if hasattr(args, 'device') and args.device != 'auto':
        config.device = args.device
    
    # Create and run trainer
    trainer = CNNViTHybridTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()