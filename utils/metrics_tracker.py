"""
Advanced metrics tracking for CNN-ViT token pruning research.
Provides additional metrics beyond the basic evaluation framework.
"""
import torch
import numpy as np
import time
from typing import Dict, Any, List, Optional
from collections import defaultdict
import json
from pathlib import Path

class AdvancedMetricsTracker:
    """Advanced metrics tracking for research analysis"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all tracked metrics"""
        self.metrics_history = defaultdict(list)
        self.epoch_metrics = {}
        self.batch_metrics = defaultdict(list)
        self.timing_metrics = defaultdict(list)
        self.memory_metrics = defaultdict(list)
        self.token_metrics = defaultdict(list)
        
    def log_batch_metrics(self, epoch: int, batch_idx: int, metrics: Dict[str, Any]):
        """Log metrics for a single batch"""
        self.batch_metrics['epoch'].append(epoch)
        self.batch_metrics['batch_idx'].append(batch_idx)
        
        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = value.item() if value.numel() == 1 else value.cpu().numpy().mean()
            self.batch_metrics[key].append(value)
    
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """Log metrics for an entire epoch"""
        self.epoch_metrics[epoch] = {}
        
        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = value.item() if value.numel() == 1 else value.cpu().numpy().mean()
            self.epoch_metrics[epoch][key] = value
            self.metrics_history[key].append(value)
    
    def log_timing_metrics(self, phase: str, duration: float, samples_processed: int = 1):
        """Log timing information"""
        self.timing_metrics[f'{phase}_duration'].append(duration)
        self.timing_metrics[f'{phase}_throughput'].append(samples_processed / duration)
        self.timing_metrics[f'{phase}_samples'].append(samples_processed)
    
    def log_memory_usage(self, phase: str, memory_mb: float):
        """Log memory usage information"""
        self.memory_metrics[f'{phase}_memory_mb'].append(memory_mb)
    
    def log_token_pruning_stats(self, original_tokens: int, selected_tokens: int, 
                               importance_scores: Optional[torch.Tensor] = None):
        """Log token pruning statistics"""
        reduction_rate = (original_tokens - selected_tokens) / original_tokens
        
        self.token_metrics['original_tokens'].append(original_tokens)
        self.token_metrics['selected_tokens'].append(selected_tokens)
        self.token_metrics['reduction_rate'].append(reduction_rate)
        self.token_metrics['reduction_percentage'].append(reduction_rate * 100)
        
        if importance_scores is not None:
            if torch.is_tensor(importance_scores):
                importance_scores = importance_scores.cpu().numpy()
            
            self.token_metrics['importance_mean'].append(np.mean(importance_scores))
            self.token_metrics['importance_std'].append(np.std(importance_scores))
            self.token_metrics['importance_min'].append(np.min(importance_scores))
            self.token_metrics['importance_max'].append(np.max(importance_scores))
    
    def calculate_training_stability(self, metric_name: str, window_size: int = 10) -> Dict[str, float]:
        """Calculate training stability metrics"""
        if metric_name not in self.metrics_history:
            return {}
        
        values = np.array(self.metrics_history[metric_name])
        if len(values) < window_size:
            return {'stability_score': 0.0, 'trend': 'insufficient_data'}
        
        # Calculate moving average and variance
        moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
        moving_var = np.array([np.var(values[i:i+window_size]) for i in range(len(values)-window_size+1)])
        
        # Stability score (lower variance = more stable)
        stability_score = 1.0 / (1.0 + np.mean(moving_var))
        
        # Trend analysis
        if len(moving_avg) > 1:
            trend_slope = np.polyfit(range(len(moving_avg)), moving_avg, 1)[0]
            if trend_slope > 0.001:
                trend = 'improving'
            elif trend_slope < -0.001:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'
        
        return {
            'stability_score': stability_score,
            'trend': trend,
            'final_value': values[-1],
            'moving_average': moving_avg[-1] if len(moving_avg) > 0 else values[-1],
            'variance': np.var(values[-window_size:])
        }
    
    def get_efficiency_summary(self) -> Dict[str, Any]:
        """Get comprehensive efficiency summary"""
        summary = {}
        
        # Timing efficiency
        if 'train_duration' in self.timing_metrics:
            train_times = self.timing_metrics['train_duration']
            summary['training_efficiency'] = {
                'avg_epoch_time': np.mean(train_times),
                'total_training_time': np.sum(train_times),
                'avg_throughput': np.mean(self.timing_metrics.get('train_throughput', [0]))
            }
        
        # Memory efficiency
        if 'train_memory_mb' in self.memory_metrics:
            memory_usage = self.memory_metrics['train_memory_mb']
            summary['memory_efficiency'] = {
                'avg_memory_mb': np.mean(memory_usage),
                'peak_memory_mb': np.max(memory_usage),
                'memory_stability': np.std(memory_usage)
            }
        
        # Token pruning efficiency
        if 'reduction_percentage' in self.token_metrics:
            reductions = self.token_metrics['reduction_percentage']
            summary['token_pruning_efficiency'] = {
                'avg_reduction_percentage': np.mean(reductions),
                'reduction_stability': np.std(reductions),
                'max_reduction': np.max(reductions),
                'min_reduction': np.min(reductions)
            }
        
        return summary
    
    def get_convergence_analysis(self) -> Dict[str, Any]:
        """Analyze model convergence patterns"""
        analysis = {}
        
        # Loss convergence
        if 'loss' in self.metrics_history:
            loss_values = np.array(self.metrics_history['loss'])
            analysis['loss_convergence'] = {
                'converged': self._check_convergence(loss_values, decreasing=True),
                'final_loss': loss_values[-1],
                'improvement_rate': (loss_values[0] - loss_values[-1]) / len(loss_values)
            }
        
        # Accuracy convergence
        if 'accuracy' in self.metrics_history:
            acc_values = np.array(self.metrics_history['accuracy'])
            analysis['accuracy_convergence'] = {
                'converged': self._check_convergence(acc_values, decreasing=False),
                'final_accuracy': acc_values[-1],
                'improvement_rate': (acc_values[-1] - acc_values[0]) / len(acc_values)
            }
        
        return analysis
    
    def _check_convergence(self, values: np.ndarray, decreasing: bool = True, 
                          window: int = 5, threshold: float = 0.001) -> bool:
        """Check if a metric has converged"""
        if len(values) < window * 2:
            return False
        
        recent_values = values[-window:]
        previous_values = values[-window*2:-window]
        
        recent_mean = np.mean(recent_values)
        previous_mean = np.mean(previous_values)
        
        change = abs(recent_mean - previous_mean)
        relative_change = change / max(abs(previous_mean), 1e-8)
        
        return relative_change < threshold
    
    def export_metrics(self, filepath: str):
        """Export all metrics to JSON file"""
        export_data = {
            'metrics_history': dict(self.metrics_history),
            'epoch_metrics': self.epoch_metrics,
            'timing_metrics': dict(self.timing_metrics),
            'memory_metrics': dict(self.memory_metrics),
            'token_metrics': dict(self.token_metrics),
            'efficiency_summary': self.get_efficiency_summary(),
            'convergence_analysis': self.get_convergence_analysis()
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=convert_numpy)
        
        print(f"📊 Advanced metrics exported to: {filepath}")
    
    def print_summary_report(self):
        """Print a comprehensive summary report"""
        print("\n" + "="*60)
        print("🔬 ADVANCED METRICS SUMMARY REPORT")
        print("="*60)
        
        # Training progress
        if 'accuracy' in self.metrics_history:
            acc_values = self.metrics_history['accuracy']
            print(f"\n📈 Training Progress:")
            print(f"   Initial Accuracy: {acc_values[0]:.4f}")
            print(f"   Final Accuracy:   {acc_values[-1]:.4f}")
            print(f"   Total Improvement: {acc_values[-1] - acc_values[0]:.4f}")
        
        # Efficiency metrics
        efficiency = self.get_efficiency_summary()
        if efficiency:
            print(f"\n⚡ Efficiency Metrics:")
            
            if 'training_efficiency' in efficiency:
                train_eff = efficiency['training_efficiency']
                print(f"   Avg Epoch Time: {train_eff['avg_epoch_time']:.2f}s")
                print(f"   Avg Throughput: {train_eff['avg_throughput']:.1f} samples/sec")
            
            if 'token_pruning_efficiency' in efficiency:
                token_eff = efficiency['token_pruning_efficiency']
                print(f"   Avg Token Reduction: {token_eff['avg_reduction_percentage']:.1f}%")
                print(f"   Reduction Stability: {token_eff['reduction_stability']:.3f}")
        
        # Convergence analysis
        convergence = self.get_convergence_analysis()
        if convergence:
            print(f"\n🎯 Convergence Analysis:")
            if 'loss_convergence' in convergence:
                loss_conv = convergence['loss_convergence']
                print(f"   Loss Converged: {loss_conv['converged']}")
                print(f"   Final Loss: {loss_conv['final_loss']:.4f}")
            
            if 'accuracy_convergence' in convergence:
                acc_conv = convergence['accuracy_convergence']
                print(f"   Accuracy Converged: {acc_conv['converged']}")
                print(f"   Final Accuracy: {acc_conv['final_accuracy']:.4f}")
        
        print("="*60)

class TokenPruningAnalyzer:
    """Specialized analyzer for token pruning research"""
    
    def __init__(self):
        self.pruning_data = []
        self.attention_patterns = []
        self.selection_consistency = []
    
    def analyze_token_selection_consistency(self, selection_history: List[torch.Tensor]) -> Dict[str, float]:
        """Analyze how consistent token selection is across batches"""
        if len(selection_history) < 2:
            return {'consistency_score': 0.0}
        
        # Convert to numpy arrays
        selections = [sel.cpu().numpy() if torch.is_tensor(sel) else sel for sel in selection_history]
        
        # Calculate pairwise Jaccard similarities
        similarities = []
        for i in range(len(selections)-1):
            sel1, sel2 = set(selections[i]), set(selections[i+1])
            jaccard = len(sel1.intersection(sel2)) / len(sel1.union(sel2))
            similarities.append(jaccard)
        
        consistency_score = np.mean(similarities)
        
        return {
            'consistency_score': consistency_score,
            'avg_jaccard_similarity': consistency_score,
            'consistency_std': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities)
        }
    
    def analyze_computational_savings(self, original_flops: int, pruning_ratios: List[float]) -> Dict[str, float]:
        """Calculate computational savings from token pruning"""
        avg_pruning = np.mean(pruning_ratios)
        savings_ratio = avg_pruning
        
        # Estimate FLOPs reduction (simplified)
        # In ViT, self-attention has O(n^2) complexity where n is number of tokens
        original_tokens = 64  # CIFAR-10 with 4x4 patches
        avg_selected = original_tokens * (1 - avg_pruning)
        
        # Quadratic savings in attention computation
        attention_flop_savings = 1 - (avg_selected / original_tokens) ** 2
        
        # Linear savings in other operations
        linear_flop_savings = avg_pruning
        
        # Combined estimate (assuming 60% attention, 40% other operations)
        total_flop_savings = 0.6 * attention_flop_savings + 0.4 * linear_flop_savings
        
        return {
            'avg_token_reduction': avg_pruning,
            'estimated_flop_savings': total_flop_savings,
            'attention_savings': attention_flop_savings,
            'linear_op_savings': linear_flop_savings,
            'theoretical_speedup': 1 / (1 - total_flop_savings)
        }
