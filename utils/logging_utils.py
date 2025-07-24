"""
Enhanced logging utilities for CNN-ViT token pruning research.
Provides structured logging for experiments and analysis.
"""
import logging
import sys
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import torch

class ResearchLogger:
    """Structured logging for research experiments"""
    
    def __init__(self, experiment_name: str, log_dir: str = "results/logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self.log_file = self.log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Experiment metadata
        self.experiment_metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'log_file': str(self.log_file),
            'device': self._get_device_info()
        }
        
        self.log_experiment_start()
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information for logging"""
        device_info = {
            'pytorch_version': torch.__version__,
            'python_version': sys.version,
        }
        
        if torch.cuda.is_available():
            device_info.update({
                'cuda_available': True,
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
            })
        elif torch.backends.mps.is_available():
            device_info.update({
                'mps_available': True,
                'device_type': 'Apple Silicon'
            })
        else:
            device_info.update({
                'device_type': 'CPU'
            })
        
        return device_info
    
    def log_experiment_start(self):
        """Log experiment initialization"""
        self.logger.info("="*60)
        self.logger.info(f"🚀 STARTING EXPERIMENT: {self.experiment_name}")
        self.logger.info("="*60)
        self.logger.info(f"📅 Start Time: {self.experiment_metadata['start_time']}")
        self.logger.info(f"💻 Device: {self.experiment_metadata['device']}")
        self.logger.info(f"📄 Log File: {self.log_file}")
        self.logger.info("="*60)
    
    def log_model_info(self, model_name: str, model_info: Dict[str, Any]):
        """Log model architecture information"""
        self.logger.info(f"\n🏗️  MODEL: {model_name}")
        self.logger.info("-" * 40)
        
        for key, value in model_info.items():
            if isinstance(value, (int, float)):
                if 'parameter' in key.lower():
                    self.logger.info(f"   {key}: {value:,}")
                else:
                    self.logger.info(f"   {key}: {value}")
            else:
                self.logger.info(f"   {key}: {value}")
    
    def log_training_config(self, config: Dict[str, Any]):
        """Log training configuration"""
        self.logger.info(f"\n⚙️  TRAINING CONFIGURATION")
        self.logger.info("-" * 40)
        
        for key, value in config.items():
            self.logger.info(f"   {key}: {value}")
    
    def log_epoch_start(self, epoch: int, total_epochs: int, phase: str = "train"):
        """Log epoch start"""
        self.logger.info(f"\n📊 EPOCH {epoch+1}/{total_epochs} - {phase.upper()}")
        self.logger.info("-" * 30)
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float], 
                     duration: float, phase: str = "train"):
        """Log epoch completion"""
        self.logger.info(f"✅ EPOCH {epoch+1} {phase.upper()} COMPLETED")
        self.logger.info(f"   Duration: {duration:.2f}s")
        
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                if 'loss' in metric_name.lower():
                    self.logger.info(f"   {metric_name}: {value:.6f}")
                elif 'accuracy' in metric_name.lower():
                    self.logger.info(f"   {metric_name}: {value:.4f}")
                else:
                    self.logger.info(f"   {metric_name}: {value:.4f}")
            else:
                self.logger.info(f"   {metric_name}: {value}")
    
    def log_token_pruning_stats(self, epoch: int, pruning_stats: Dict[str, Any]):
        """Log token pruning statistics"""
        self.logger.info(f"\n🎯 TOKEN PRUNING STATS - Epoch {epoch+1}")
        self.logger.info("-" * 35)
        
        for key, value in pruning_stats.items():
            if isinstance(value, float):
                if 'percentage' in key.lower():
                    self.logger.info(f"   {key}: {value:.1f}%")
                else:
                    self.logger.info(f"   {key}: {value:.3f}")
            else:
                self.logger.info(f"   {key}: {value}")
    
    def log_checkpoint_save(self, epoch: int, model_name: str, 
                           checkpoint_path: str, is_best: bool = False):
        """Log checkpoint saving"""
        prefix = "🏆 BEST" if is_best else "💾"
        self.logger.info(f"{prefix} Checkpoint saved for {model_name}")
        self.logger.info(f"   Epoch: {epoch+1}")
        self.logger.info(f"   Path: {checkpoint_path}")
    
    def log_comparison_results(self, comparison_results: Dict[str, Any]):
        """Log model comparison results"""
        self.logger.info(f"\n🔬 MODEL COMPARISON RESULTS")
        self.logger.info("="*50)
        
        models = comparison_results.get('cross_analysis', {}).get('metrics_summary', {})
        
        for model_name, metrics in models.items():
            self.logger.info(f"\n📊 {model_name.upper()}")
            self.logger.info("-" * 20)
            
            accuracy = metrics.get('accuracy', 0)
            speed = metrics.get('samples_per_second', 0)
            params = metrics.get('total_parameters', 0)
            
            self.logger.info(f"   Accuracy: {accuracy:.1%}")
            self.logger.info(f"   Speed: {speed:.1f} samples/sec")
            self.logger.info(f"   Parameters: {params:,}")
            
            if 'token_reduction_percent' in metrics:
                reduction = metrics['token_reduction_percent']
                self.logger.info(f"   Token Reduction: {reduction:.1f}%")
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        """Log errors with context"""
        self.logger.error(f"❌ ERROR: {error_msg}")
        if exception:
            self.logger.error(f"   Exception: {str(exception)}")
            self.logger.error(f"   Type: {type(exception).__name__}")
    
    def log_warning(self, warning_msg: str):
        """Log warnings"""
        self.logger.warning(f"⚠️  WARNING: {warning_msg}")
    
    def log_success(self, success_msg: str):
        """Log success messages"""
        self.logger.info(f"✅ SUCCESS: {success_msg}")
    
    def log_research_insights(self, insights: Dict[str, Any]):
        """Log research insights and conclusions"""
        self.logger.info(f"\n🧬 RESEARCH INSIGHTS")
        self.logger.info("="*40)
        
        for category, insight_data in insights.items():
            self.logger.info(f"\n📋 {category.upper().replace('_', ' ')}")
            self.logger.info("-" * 25)
            
            if isinstance(insight_data, dict):
                for key, value in insight_data.items():
                    self.logger.info(f"   {key}: {value}")
            else:
                self.logger.info(f"   {insight_data}")
    
    def log_experiment_end(self, final_results: Dict[str, Any]):
        """Log experiment completion"""
        end_time = datetime.now().isoformat()
        
        self.logger.info("\n" + "="*60)
        self.logger.info(f"🏁 EXPERIMENT COMPLETED: {self.experiment_name}")
        self.logger.info("="*60)
        self.logger.info(f"📅 End Time: {end_time}")
        self.logger.info(f"📊 Final Results Summary:")
        
        if 'best_accuracy' in final_results:
            self.logger.info(f"   Best Accuracy: {final_results['best_accuracy']:.4f}")
        
        if 'total_training_time' in final_results:
            self.logger.info(f"   Total Training Time: {final_results['total_training_time']:.2f}s")
        
        if 'computational_savings' in final_results:
            self.logger.info(f"   Computational Savings: {final_results['computational_savings']:.1%}")
        
        self.logger.info(f"📄 Complete logs saved to: {self.log_file}")
        self.logger.info("="*60)
        
        # Update experiment metadata
        self.experiment_metadata.update({
            'end_time': end_time,
            'final_results': final_results
        })
        
        # Save experiment metadata
        metadata_file = self.log_dir / f"{self.experiment_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2, default=str)
    
    def close(self):
        """Close logger and handlers"""
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()

def setup_experiment_logging(experiment_name: str, log_dir: str = "results/logs") -> ResearchLogger:
    """Setup logging for an experiment"""
    return ResearchLogger(experiment_name, log_dir)

def log_system_info():
    """Log system information for reproducibility"""
    logger = logging.getLogger("system_info")
    
    logger.info("💻 SYSTEM INFORMATION")
    logger.info("-" * 30)
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Available: Yes")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    elif torch.backends.mps.is_available():
        logger.info(f"MPS Available: Yes (Apple Silicon)")
    else:
        logger.info(f"Device: CPU Only")
