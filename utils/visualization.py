"""
Visualization utilities for CNN-ViT token pruning research.
Creates attention maps, importance heatmaps, and token selection visualizations.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from typing import Dict, Any, List, Tuple, Optional

class TokenPruningVisualizer:
    """Visualization tools for CNN-ViT dynamic token pruning research"""
    
    def __init__(self, save_dir: str = "results/visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib for publication-quality figures
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def visualize_cnn_importance_heatmap(self, image: torch.Tensor, 
                                       importance_scores: torch.Tensor,
                                       patch_size: int = 4,
                                       title: str = "CNN Importance Scores",
                                       save_path: str = None) -> plt.Figure:
        """Create heatmap showing CNN-predicted patch importance"""
        
        # Convert tensors to numpy
        if torch.is_tensor(image):
            image = image.cpu().numpy()
        if torch.is_tensor(importance_scores):
            importance_scores = importance_scores.cpu().numpy()
        
        # Handle image format [C, H, W] -> [H, W, C]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Denormalize image (assuming CIFAR-10 normalization)
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # Reshape importance scores to spatial grid
        grid_size = int(np.sqrt(len(importance_scores)))
        importance_grid = importance_scores.reshape(grid_size, grid_size)
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Importance heatmap
        im = ax2.imshow(importance_grid, cmap='hot', interpolation='nearest')
        ax2.set_title("CNN Importance Scores")
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # Overlay heatmap on image
        heatmap_resized = cv2.resize(importance_grid, (image.shape[1], image.shape[0]))
        overlay = ax3.imshow(image)
        ax3.imshow(heatmap_resized, alpha=0.6, cmap='hot')
        ax3.set_title("Importance Overlay")
        ax3.axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(self.save_dir / save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def visualize_token_selection(self, image: torch.Tensor,
                                importance_scores: torch.Tensor,
                                selected_indices: torch.Tensor,
                                patch_size: int = 4,
                                title: str = "Dynamic Token Selection",
                                save_path: str = None) -> plt.Figure:
        """Visualize which patches are selected for ViT processing"""
        
        # Convert to numpy
        if torch.is_tensor(image):
            image = image.cpu().numpy()
        if torch.is_tensor(importance_scores):
            importance_scores = importance_scores.cpu().numpy()
        if torch.is_tensor(selected_indices):
            selected_indices = selected_indices.cpu().numpy()
        
        # Handle image format
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Denormalize
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # Create selection mask
        grid_size = int(np.sqrt(len(importance_scores)))
        selection_mask = np.zeros(len(importance_scores))
        selection_mask[selected_indices] = 1
        selection_grid = selection_mask.reshape(grid_size, grid_size)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original image
        axes[0,0].imshow(image)
        axes[0,0].set_title("Original Image")
        axes[0,0].axis('off')
        
        # Importance scores
        im1 = axes[0,1].imshow(importance_scores.reshape(grid_size, grid_size), 
                              cmap='viridis', interpolation='nearest')
        axes[0,1].set_title("CNN Importance Scores")
        axes[0,1].axis('off')
        plt.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)
        
        # Selected patches
        im2 = axes[1,0].imshow(selection_grid, cmap='Reds', interpolation='nearest')
        axes[1,0].set_title(f"Selected Patches ({len(selected_indices)}/{len(importance_scores)})")
        axes[1,0].axis('off')
        plt.colorbar(im2, ax=axes[1,0], fraction=0.046, pad=0.04)
        
        # Combined visualization
        axes[1,1].imshow(image)
        # Overlay selection grid
        selection_resized = cv2.resize(selection_grid, (image.shape[1], image.shape[0]))
        axes[1,1].imshow(selection_resized, alpha=0.7, cmap='Reds')
        axes[1,1].set_title("Selected Regions Overlay")
        axes[1,1].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(self.save_dir / save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_model_comparison(self, comparison_results: Dict[str, Any], 
                            save_path: str = "model_comparison.png") -> plt.Figure:
        """Create comprehensive model comparison plots"""
        
        # Extract metrics
        models_data = comparison_results.get('models', {})
        cross_analysis = comparison_results.get('cross_analysis', {})
        metrics_summary = cross_analysis.get('metrics_summary', {})
        
        model_names = list(metrics_summary.keys())
        model_display_names = {
            'resnet18': 'ResNet18\n(CNN)',
            'vit_small': 'ViT-Small\n(Transformer)', 
            'cnn_vit_hybrid': 'CNN-ViT Hybrid\n(Our Method)'
        }
        
        # Extract metrics
        accuracies = [metrics_summary[model].get('accuracy', 0) * 100 for model in model_names]
        speeds = [metrics_summary[model].get('samples_per_second', 0) for model in model_names]
        parameters = [metrics_summary[model].get('total_parameters', 0) / 1e6 for model in model_names]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Colors for each model
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # 1. Accuracy comparison
        bars1 = ax1.bar([model_display_names[model] for model in model_names], 
                       accuracies, color=colors, alpha=0.8)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, max(accuracies) * 1.2)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Inference speed comparison
        bars2 = ax2.bar([model_display_names[model] for model in model_names], 
                       speeds, color=colors, alpha=0.8)
        ax2.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Samples/Second')
        
        for bar, speed in zip(bars2, speeds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{speed:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Parameter count comparison
        bars3 = ax3.bar([model_display_names[model] for model in model_names], 
                       parameters, color=colors, alpha=0.8)
        ax3.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Parameters (Millions)')
        
        for bar, param in zip(bars3, parameters):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{param:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        # 4. Efficiency analysis (Accuracy vs Speed)
        for i, model in enumerate(model_names):
            ax4.scatter(speeds[i], accuracies[i], s=parameters[i]*20, 
                       color=colors[i], alpha=0.7, 
                       label=model_display_names[model])
            
        ax4.set_xlabel('Inference Speed (Samples/Second)')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Efficiency Analysis\n(Bubble size = Model parameters)', 
                     fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.save_dir / save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_token_reduction_analysis(self, hybrid_results: Dict[str, Any],
                                    save_path: str = "token_reduction_analysis.png") -> plt.Figure:
        """Analyze token reduction effectiveness"""
        
        # Extract token reduction data
        training_history = hybrid_results.get('training_history', {})
        token_stats = training_history.get('token_reduction_stats', [])
        
        if not token_stats:
            print("No token reduction statistics found")
            return None
        
        # Extract metrics over training
        epochs = list(range(len(token_stats)))
        reduction_rates = [stat.get('avg_token_reduction_percentage', 0) for stat in token_stats]
        selected_tokens = [stat.get('avg_selected_tokens', 0) for stat in token_stats]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Token reduction over training
        ax1.plot(epochs, reduction_rates, 'b-', linewidth=2, marker='o')
        ax1.set_title('Token Reduction During Training', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Training Epoch (Stage 2)')
        ax1.set_ylabel('Token Reduction (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Selected tokens over training
        ax2.plot(epochs, selected_tokens, 'r-', linewidth=2, marker='s')
        ax2.axhline(y=64, color='gray', linestyle='--', alpha=0.7, label='Original (64 tokens)')
        ax2.set_title('Average Tokens Used During Training', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training Epoch (Stage 2)')
        ax2.set_ylabel('Average Tokens Selected')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.save_dir / save_path, dpi=300, bbox_inches='tight')
        
        return fig

def create_research_visualizations(results_dir: str = "results"):
    """Generate all research visualizations from saved results"""
    visualizer = TokenPruningVisualizer()
    results_path = Path(results_dir)
    
    print("🎨 Creating research visualizations...")
    
    # Load comparison results
    comparison_file = results_path / "comparison" / "comprehensive_analysis.json"
    if comparison_file.exists():
        import json
        with open(comparison_file, 'r') as f:
            comparison_results = json.load(f)
        
        # Create model comparison plot
        visualizer.plot_model_comparison(comparison_results)
        print("✅ Model comparison plot created")
        
        # Create token reduction analysis
        hybrid_results = comparison_results.get('models', {}).get('cnn_vit_hybrid', {})
        if hybrid_results:
            visualizer.plot_token_reduction_analysis(hybrid_results)
            print("✅ Token reduction analysis created")
    
    print(f"📊 All visualizations saved to: {visualizer.save_dir}")

if __name__ == "__main__":
    create_research_visualizations()
