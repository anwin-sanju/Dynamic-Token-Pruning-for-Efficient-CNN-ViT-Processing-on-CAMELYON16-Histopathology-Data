"""
Comprehensive evaluation metrics for CNN-ViT token pruning experiments.
"""
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import psutil
import os

class PerformanceMetrics:
    """Calculate classification performance metrics"""
    
    def __init__(self, num_classes: int = 10, average: str = 'macro'):
        self.num_classes = num_classes
        self.average = average
        self.reset()
    
    def reset(self):
        """Reset all stored values"""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               probabilities: Optional[torch.Tensor] = None):
        """Update metrics with batch results"""
        # Convert to numpy
        if predictions.is_cuda or str(predictions.device).startswith('mps'):
            predictions = predictions.cpu()
            targets = targets.cpu()
            if probabilities is not None:
                probabilities = probabilities.cpu()
        
        self.all_predictions.extend(predictions.numpy().flatten())
        self.all_targets.extend(targets.numpy().flatten())
        
        if probabilities is not None:
            self.all_probabilities.extend(probabilities.numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute all performance metrics"""
        if not self.all_predictions:
            return {}
        
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0)
        }
        
        # Add per-class metrics
        if self.num_classes <= 10:  # Only for manageable number of classes
            precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
            recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            for i in range(min(self.num_classes, len(precisions))):
                metrics[f'precision_class_{i}'] = precisions[i]
                metrics[f'recall_class_{i}'] = recalls[i]
                metrics[f'f1_class_{i}'] = f1s[i]
        
        # Add AUC if probabilities available
        if self.all_probabilities and self.num_classes <= 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, self.all_probabilities)
            except:
                pass
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        if not self.all_predictions:
            return np.array([])
        
        return confusion_matrix(self.all_targets, self.all_predictions)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report"""
        if not self.all_predictions:
            return ""
        
        return classification_report(self.all_targets, self.all_predictions)

class EfficiencyMetrics:
    """Calculate computational efficiency metrics"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset timing and memory measurements"""
        self.batch_times = []
        self.memory_usage = []
        self.flop_counts = []
        self.throughput_measurements = []
    
    def start_batch_timer(self):
        """Start timing a batch"""
        if self.device == 'mps':
            torch.mps.synchronize()
        elif self.device.startswith('cuda'):
            torch.cuda.synchronize()
        
        self.batch_start_time = time.time()
    
    def end_batch_timer(self, batch_size: int):
        """End timing a batch and record metrics"""
        if self.device == 'mps':
            torch.mps.synchronize()
        elif self.device.startswith('cuda'):
            torch.cuda.synchronize()
        
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)
        
        # Calculate throughput (samples per second)
        throughput = batch_size / batch_time if batch_time > 0 else 0
        self.throughput_measurements.append(throughput)
    
    def measure_memory_usage(self):
        """Measure current memory usage"""
        if self.device == 'mps':
            try:
                memory_mb = torch.mps.current_allocated_memory() / 1024 / 1024
            except:
                memory_mb = 0
        elif self.device.startswith('cuda'):
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            # CPU memory (approximate)
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
        
        self.memory_usage.append(memory_mb)
    
    def compute(self) -> Dict[str, float]:
        """Compute efficiency metrics"""
        metrics = {}
        
        if self.batch_times:
            metrics.update({
                'avg_batch_time': np.mean(self.batch_times),
                'std_batch_time': np.std(self.batch_times),
                'total_inference_time': np.sum(self.batch_times)
            })
        
        if self.throughput_measurements:
            metrics.update({
                'avg_throughput': np.mean(self.throughput_measurements),
                'max_throughput': np.max(self.throughput_measurements),
                'min_throughput': np.min(self.throughput_measurements)
            })
        
        if self.memory_usage:
            metrics.update({
                'avg_memory_mb': np.mean(self.memory_usage),
                'peak_memory_mb': np.max(self.memory_usage),
                'min_memory_mb': np.min(self.memory_usage)
            })
        
        return metrics

class TokenPruningMetrics:
    """Metrics specific to token pruning evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset token pruning measurements"""
        self.original_token_counts = []
        self.selected_token_counts = []
        self.importance_scores = []
        self.selection_consistency = []
    
    def update(self, original_tokens: int, selected_tokens: Union[int, torch.Tensor],
               importance_scores: Optional[torch.Tensor] = None):
        """Update token pruning metrics"""
        self.original_token_counts.append(original_tokens)
        
        if isinstance(selected_tokens, torch.Tensor):
            if selected_tokens.dim() == 0:  # Scalar tensor
                selected_tokens = selected_tokens.item()
            else:  # Count non-zero or valid tokens
                selected_tokens = (selected_tokens > 0).sum().item()
        
        self.selected_token_counts.append(selected_tokens)
        
        if importance_scores is not None:
            # Store importance statistics
            if importance_scores.is_cuda or str(importance_scores.device).startswith('mps'):
                importance_scores = importance_scores.cpu()
            
            self.importance_scores.append({
                'mean': importance_scores.mean().item(),
                'std': importance_scores.std().item(),
                'min': importance_scores.min().item(),
                'max': importance_scores.max().item()
            })
    
    def compute(self) -> Dict[str, float]:
        """Compute token pruning metrics"""
        if not self.original_token_counts or not self.selected_token_counts:
            return {}
        
        original = np.array(self.original_token_counts)
        selected = np.array(self.selected_token_counts)
        
        # Calculate reduction metrics
        reduction_rates = (original - selected) / original
        
        metrics = {
            'avg_original_tokens': np.mean(original),
            'avg_selected_tokens': np.mean(selected),
            'avg_token_reduction_rate': np.mean(reduction_rates),
            'avg_token_reduction_percentage': np.mean(reduction_rates) * 100,
            'std_token_reduction_rate': np.std(reduction_rates),
            'min_selected_tokens': np.min(selected),
            'max_selected_tokens': np.max(selected),
            'computational_savings': np.mean(reduction_rates)
        }
        
        # Add importance score statistics if available
        if self.importance_scores:
            importance_means = [score['mean'] for score in self.importance_scores]
            importance_stds = [score['std'] for score in self.importance_scores]
            
            metrics.update({
                'avg_importance_mean': np.mean(importance_means),
                'avg_importance_std': np.mean(importance_stds),
                'importance_consistency': 1.0 - np.std(importance_means)  # Higher = more consistent
            })
        
        return metrics

class ComprehensiveEvaluator:
    """Comprehensive evaluation combining all metric types"""
    
    def __init__(self, num_classes: int = 10, device: str = 'cpu'):
        self.performance_metrics = PerformanceMetrics(num_classes)
        self.efficiency_metrics = EfficiencyMetrics(device)
        self.token_metrics = TokenPruningMetrics()
        self.device = device
    
    def reset(self):
        """Reset all metrics"""
        self.performance_metrics.reset()
        self.efficiency_metrics.reset()
        self.token_metrics.reset()
    
    def update_batch(self, outputs: torch.Tensor, targets: torch.Tensor,
                    batch_size: int, batch_time: Optional[float] = None,
                    token_info: Optional[Dict[str, Any]] = None,
                    probabilities: Optional[torch.Tensor] = None):
        """Update all metrics with batch results"""
        # Convert outputs to predictions
        if outputs.dim() > 1 and outputs.size(1) > 1:
            # Multi-class
            _, predictions = torch.max(outputs, 1)
            if probabilities is None:
                probabilities = torch.softmax(outputs, dim=1)
        else:
            # Binary
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            if probabilities is None:
                probabilities = torch.sigmoid(outputs)
        
        # Update performance metrics
        self.performance_metrics.update(predictions, targets, probabilities)
        
        # Update efficiency metrics
        if batch_time is not None:
            self.efficiency_metrics.batch_times.append(batch_time)
            throughput = batch_size / batch_time if batch_time > 0 else 0
            self.efficiency_metrics.throughput_measurements.append(throughput)
        
        self.efficiency_metrics.measure_memory_usage()
        
        # Update token metrics if available
        if token_info:
            self.token_metrics.update(
                original_tokens=token_info.get('original_tokens', 0),
                selected_tokens=token_info.get('selected_tokens', 0),
                importance_scores=token_info.get('importance_scores')
            )
    
    def compute_all_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics"""
        results = {}
        
        # Performance metrics
        performance = self.performance_metrics.compute()
        if performance:
            results['performance'] = performance
        
        # Efficiency metrics
        efficiency = self.efficiency_metrics.compute()
        if efficiency:
            results['efficiency'] = efficiency
        
        # Token pruning metrics
        token_metrics = self.token_metrics.compute()
        if token_metrics:
            results['token_pruning'] = token_metrics
        
        # Combined metrics for comparison
        if 'performance' in results and 'efficiency' in results:
            results['combined'] = {
                'accuracy_per_second': (
                    results['performance'].get('accuracy', 0) / 
                    results['efficiency'].get('avg_batch_time', 1)
                ),
                'f1_per_mb': (
                    results['performance'].get('f1_macro', 0) / 
                    max(results['efficiency'].get('avg_memory_mb', 1), 1)
                )
            }
        
        return results
    
    def get_summary_report(self) -> str:
        """Generate human-readable summary report"""
        metrics = self.compute_all_metrics()
        
        report = "## Evaluation Summary\n\n"
        
        # Performance section
        if 'performance' in metrics:
            perf = metrics['performance']
            report += "### Performance Metrics\n"
            report += f"- **Accuracy**: {perf.get('accuracy', 0):.4f}\n"
            report += f"- **F1-Score (Macro)**: {perf.get('f1_macro', 0):.4f}\n"
            report += f"- **Precision (Macro)**: {perf.get('precision_macro', 0):.4f}\n"
            report += f"- **Recall (Macro)**: {perf.get('recall_macro', 0):.4f}\n\n"
        
        # Efficiency section
        if 'efficiency' in metrics:
            eff = metrics['efficiency']
            report += "### Efficiency Metrics\n"
            report += f"- **Average Throughput**: {eff.get('avg_throughput', 0):.2f} samples/sec\n"
            report += f"- **Average Batch Time**: {eff.get('avg_batch_time', 0):.4f} seconds\n"
            report += f"- **Peak Memory Usage**: {eff.get('peak_memory_mb', 0):.2f} MB\n\n"
        
        # Token pruning section
        if 'token_pruning' in metrics:
            token = metrics['token_pruning']
            report += "### Token Pruning Metrics\n"
            report += f"- **Token Reduction**: {token.get('avg_token_reduction_percentage', 0):.1f}%\n"
            report += f"- **Computational Savings**: {token.get('computational_savings', 0):.3f}\n"
            report += f"- **Average Selected Tokens**: {token.get('avg_selected_tokens', 0):.1f}\n\n"
        
        return report

def compare_models(model_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compare metrics across multiple models"""
    comparison = {
        'model_rankings': {},
        'relative_performance': {},
        'efficiency_analysis': {}
    }
    
    models = list(model_metrics.keys())
    
    # Rank models by different criteria
    criteria = ['accuracy', 'f1_macro', 'avg_throughput', 'computational_savings']
    
    for criterion in criteria:
        scores = {}
        for model_name, metrics in model_metrics.items():
            # Navigate nested metric structure
            score = 0
            if criterion in ['accuracy', 'f1_macro']:
                score = metrics.get('performance', {}).get(criterion, 0)
            elif criterion == 'avg_throughput':
                score = metrics.get('efficiency', {}).get(criterion, 0)
            elif criterion == 'computational_savings':
                score = metrics.get('token_pruning', {}).get(criterion, 0)
            
            scores[model_name] = score
        
        # Sort by score (descending)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        comparison['model_rankings'][criterion] = ranked
    
    return comparison
