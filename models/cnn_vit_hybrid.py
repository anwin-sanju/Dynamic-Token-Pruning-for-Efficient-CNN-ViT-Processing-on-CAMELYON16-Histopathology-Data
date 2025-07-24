"""
CNN-ViT Hybrid architecture with dynamic token pruning.
This is the main contribution of the research project.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from .resnet18 import ResNet18ROIScorer, PatchResNet18ROIScorer
from .vit_small import ViTSmallCustom
from utils.data_utils import select_top_k_patches, calculate_token_reduction_stats

class CNNViTHybrid(nn.Module):
    """
    Main CNN-ViT hybrid architecture with dynamic token pruning
    
    Architecture:
    1. CNN ROI Scorer: ResNet18 predicts importance scores for patches
    2. Dynamic Token Selection: Select top-k patches based on importance
    3. ViT Backbone: Process selected tokens with Vision Transformer
    4. Classification: Final prediction with efficiency metrics
    """
    
    def __init__(self, num_classes: int = 10, img_size: int = 32, patch_size: int = 4,
                 pruning_ratio: float = 0.5, selection_method: str = "top_k",
                 freeze_cnn_stage2: bool = True, embed_dim: int = 384,
                 vit_depth: int = 12, vit_heads: int = 6, min_tokens: int = 16):
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.pruning_ratio = pruning_ratio
        self.selection_method = selection_method
        self.min_tokens = min_tokens
        self.num_patches = (img_size // patch_size) ** 2
        
        # Calculate number of tokens to keep
        self.num_selected = max(min_tokens, int(self.num_patches * (1 - pruning_ratio)))
        
        # CNN ROI Scorer - predicts importance of patches
        self.cnn_roi_scorer = PatchResNet18ROIScorer(
            num_classes=num_classes,
            patch_size=patch_size,
            pretrained=True
        )
        
        # ViT Backbone - processes selected tokens
        self.vit_backbone = ViTSmallCustom(
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            max_tokens=self.num_selected
        )
        
        # Training configuration
        self.freeze_cnn_stage2 = freeze_cnn_stage2
        self.training_stage = 1  # 1: CNN pre-training, 2: End-to-end
        
        print(f"✅ CNNViTHybrid initialized:")
        print(f"   - Patches: {self.num_patches} → {self.num_selected} ({self.pruning_ratio*100:.1f}% reduction)")
        print(f"   - Selection: {selection_method}")
        print(f"   - ViT: {embed_dim}D, {vit_depth} layers, {vit_heads} heads")
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dynamic token pruning
        
        Args:
            x: Input images [batch_size, channels, height, width]
            return_attention: Whether to return attention weights for visualization
            
        Returns:
            Dictionary containing:
            - logits: Classification predictions
            - importance_scores: CNN importance predictions
            - selected_indices: Indices of selected patches
            - num_tokens_used: Number of tokens processed by ViT
            - token_reduction_stats: Statistics about token pruning
        """
        batch_size = x.shape[0]
        
        # Step 1: Extract patches from input images
        patches = self._extract_patches(x)  # [batch_size, num_patches, C, patch_size, patch_size]
        
        # Step 2: CNN ROI scoring
        if self.training_stage == 2 and self.freeze_cnn_stage2:
            with torch.no_grad():
                importance_scores = self.cnn_roi_scorer(patches)
        else:
            importance_scores = self.cnn_roi_scorer(patches)
        
        # Step 3: Dynamic token selection
        selected_patches, selected_indices = self._select_tokens(patches, importance_scores)
        
        # Step 4: ViT inference on selected tokens
        # Reconstruct images from selected patches for ViT processing
        selected_images = self._reconstruct_images_from_patches(selected_patches, selected_indices, batch_size)
        
        # ViT forward pass
        logits = self.vit_backbone(selected_images, selected_indices)
        
        # Step 5: Calculate token reduction statistics
        num_tokens_used = torch.full((batch_size,), self.num_selected, device=x.device)
        token_stats = calculate_token_reduction_stats(self.num_patches, num_tokens_used)
        
        results = {
            'logits': logits,
            'importance_scores': importance_scores,
            'selected_indices': selected_indices,
            'num_tokens_used': num_tokens_used,
            'token_reduction_stats': token_stats,
            'selected_patches': selected_patches  # For visualization
        }
        
        if return_attention:
            # Extract attention weights from ViT (simplified)
            results['attention_weights'] = None  # Implement if needed
        
        return results
    
    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patches from input images"""
        batch_size, channels, height, width = x.shape
        
        # Extract non-overlapping patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, -1, channels, self.patch_size, self.patch_size)
        
        return patches
    
    def _select_tokens(self, patches: torch.Tensor, importance_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select patches based on importance scores"""
        if self.selection_method == "top_k":
            selected_patches, selected_indices = select_top_k_patches(
                patches, importance_scores, self.num_selected
            )
        elif self.selection_method == "threshold":
            # Implement threshold-based selection
            threshold = importance_scores.mean() + 0.5 * importance_scores.std()
            mask = importance_scores >= threshold.unsqueeze(1)
            
            # Ensure minimum number of tokens
            selected_patches = []
            selected_indices = []
            
            for i in range(patches.shape[0]):
                valid_indices = torch.where(mask[i])[0]
                if len(valid_indices) < self.min_tokens:
                    _, top_indices = torch.topk(importance_scores[i], self.min_tokens)
                    valid_indices = top_indices
                elif len(valid_indices) > self.num_selected:
                    scores_subset = importance_scores[i][valid_indices]
                    _, top_subset = torch.topk(scores_subset, self.num_selected)
                    valid_indices = valid_indices[top_subset]
                
                selected_patches.append(patches[i, valid_indices])
                selected_indices.append(valid_indices)
            
            # Pad to consistent size for batch processing
            selected_patches, selected_indices = self._pad_selections(selected_patches, selected_indices)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        
        return selected_patches, selected_indices
    
    def _reconstruct_images_from_patches(self, selected_patches: torch.Tensor, 
                                       selected_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Reconstruct images from selected patches for ViT processing
        This is a simplified version - in practice, you might want to:
        1. Create sparse images with selected patches
        2. Use masking tokens for non-selected regions
        3. Preserve spatial relationships
        """
        # For now, we'll create a simplified reconstruction
        # In a full implementation, this would be more sophisticated
        device = selected_patches.device
        
        # Create images with selected patches placed at their original positions
        reconstructed = torch.zeros(batch_size, 3, self.img_size, self.img_size, device=device)
        
        patches_per_side = self.img_size // self.patch_size
        
        for b in range(batch_size):
            for i, patch_idx in enumerate(selected_indices[b]):
                if i >= selected_patches.shape[1]:  # Handle padding
                    break
                
                # Calculate patch position
                row = patch_idx // patches_per_side
                col = patch_idx % patches_per_side
                
                # Place patch in reconstructed image
                start_row = row * self.patch_size
                end_row = start_row + self.patch_size
                start_col = col * self.patch_size  
                end_col = start_col + self.patch_size
                
                reconstructed[b, :, start_row:end_row, start_col:end_col] = selected_patches[b, i]
        
        return reconstructed
    
    def _pad_selections(self, selected_patches_list, selected_indices_list):
        """Pad variable-length selections to consistent batch size"""
        batch_size = len(selected_patches_list)
        max_selected = max(len(indices) for indices in selected_indices_list)
        
        # Pad patches
        padded_patches = torch.zeros(batch_size, max_selected, 3, self.patch_size, self.patch_size,
                                   device=selected_patches_list[0].device)
        padded_indices = torch.zeros(batch_size, max_selected, dtype=torch.long,
                                   device=selected_indices_list[0].device)
        
        for i, (patches, indices) in enumerate(zip(selected_patches_list, selected_indices_list)):
            num_patches = len(indices)
            padded_patches[i, :num_patches] = patches
            padded_indices[i, :num_patches] = indices
        
        return padded_patches, padded_indices
    
    def set_training_stage(self, stage: int):
        """Set training stage (1: CNN pre-training, 2: End-to-end)"""
        self.training_stage = stage
        
        if stage == 2 and self.freeze_cnn_stage2:
            # Freeze CNN parameters for stage 2
            for param in self.cnn_roi_scorer.parameters():
                param.requires_grad = False
            print("🔒 CNN parameters frozen for stage 2 training")
        elif stage == 1:
            # Unfreeze CNN for stage 1
            for param in self.cnn_roi_scorer.parameters():
                param.requires_grad = True
            print("🔓 CNN parameters unfrozen for stage 1 training")
    
    def get_cnn_importance_map(self, x: torch.Tensor) -> torch.Tensor:
        """Get CNN importance heatmap for visualization"""
        patches = self._extract_patches(x)
        with torch.no_grad():
            importance_scores = self.cnn_roi_scorer(patches)
        
        # Reshape to spatial grid
        batch_size = x.shape[0]
        patches_per_side = self.img_size // self.patch_size
        importance_map = importance_scores.view(batch_size, patches_per_side, patches_per_side)
        
        return importance_map
    
    def get_efficiency_metrics(self, batch_size: int = 1) -> Dict[str, float]:
        """Calculate efficiency metrics"""
        original_tokens = self.num_patches
        selected_tokens = self.num_selected
        
        return {
            'original_tokens': original_tokens,
            'selected_tokens': selected_tokens,
            'pruning_ratio': self.pruning_ratio,
            'token_reduction_rate': (original_tokens - selected_tokens) / original_tokens,
            'computational_savings': 1 - (selected_tokens / original_tokens),
            'tokens_per_image': selected_tokens
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        cnn_info = self.cnn_roi_scorer.get_model_info()
        vit_info = self.vit_backbone.get_model_info()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'CNNViTHybrid',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'cnn_parameters': cnn_info['total_parameters'],
            'vit_parameters': vit_info['total_parameters'],
            'num_classes': self.num_classes,
            'pruning_ratio': self.pruning_ratio,
            'num_patches': self.num_patches,
            'num_selected': self.num_selected,
            'selection_method': self.selection_method,
            'training_stage': self.training_stage,
            'efficiency_metrics': self.get_efficiency_metrics()
        }

def create_hybrid_model(config) -> CNNViTHybrid:
    """
    Factory function to create CNN-ViT hybrid model from configuration
    
    Args:
        config: HybridConfig object
        
    Returns:
        Initialized CNNViTHybrid model
    """
    return CNNViTHybrid(
        num_classes=config.num_classes,
        img_size=config.img_size,
        patch_size=config.patch_size,
        pruning_ratio=config.pruning_ratio,
        selection_method=config.selection_method,
        freeze_cnn_stage2=config.freeze_cnn_stage2,
        embed_dim=getattr(config, 'embed_dim', 384),
        vit_depth=getattr(config, 'depth', 12),
        vit_heads=getattr(config, 'num_heads', 6),
        min_tokens=getattr(config, 'min_tokens', 16)
    )
