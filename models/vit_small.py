"""
ViT-Small implementation for baseline comparison and hybrid architecture.
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, Any
try:
    from timm import create_model
    from timm.models.vision_transformer import VisionTransformer
except ImportError:
    print("Warning: timm not installed. Install with: pip install timm")
    VisionTransformer = None

class ViTSmallBaseline(nn.Module):
    """Standard ViT-Small for baseline comparison"""
    
    def __init__(self, num_classes: int = 10, img_size: int = 32, patch_size: int = 4, 
                 pretrained: bool = True, embed_dim: int = 384):
        super().__init__()
        
        if VisionTransformer is None:
            raise ImportError("timm is required for ViT implementation")
        
        # Create ViT-Small model adapted for CIFAR-10
        self.model = create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            drop_rate=0.1,
            attn_drop_rate=0.1
        )
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        print(f"✅ ViTSmallBaseline initialized: {num_classes} classes, {self.num_patches} patches")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard ViT forward pass"""
        return self.model(x)
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """Extract attention weights for visualization"""
        # This is a simplified version - actual implementation depends on timm version
        self.model.eval()
        with torch.no_grad():
            # Forward through patch embedding
            x = self.model.patch_embed(x)
            x = x + self.model.pos_embed[:, 1:, :]
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.model.pos_drop(x)
            
            # Forward through transformer blocks
            for i, block in enumerate(self.model.blocks):
                if i == len(self.model.blocks) - 1 or i == layer_idx:
                    # Extract attention from the last or specified layer
                    attn_weights = block.attn(block.norm1(x), return_attention=True)
                    if isinstance(attn_weights, tuple):
                        x, attn_weights = attn_weights
                    else:
                        x = block(x)
                else:
                    x = block(x)
        
        return attn_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ViTSmallBaseline',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim,
            'num_patches': self.num_patches,
            'img_size': self.img_size,
            'patch_size': self.patch_size
        }

class ViTSmallCustom(nn.Module):
    """Custom ViT-Small for token pruning experiments"""
    
    def __init__(self, num_classes: int = 10, img_size: int = 32, patch_size: int = 4,
                 embed_dim: int = 384, depth: int = 12, num_heads: int = 6,
                 max_tokens: Optional[int] = None):
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2
        self.max_tokens = max_tokens or self.num_patches
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1)
            for _ in range(depth)
        ])
        
        # Layer normalization and classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Dropout
        self.pos_drop = nn.Dropout(0.1)
        
        self._init_weights()
        
        print(f"✅ ViTSmallCustom initialized: {embed_dim}D, {depth} layers, {num_heads} heads")
    
    def _init_weights(self):
        """Initialize weights"""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor, token_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional token selection
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            token_indices: Optional tensor [batch_size, num_selected] for token pruning
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        
        # Patch embedding: [batch_size, embed_dim, num_patches_h, num_patches_w]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        
        # Token selection if specified
        if token_indices is not None:
            x = self._select_tokens(x, token_indices)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings
        if token_indices is not None:
            # Custom positional embedding for selected tokens
            pos_embed = self._get_selected_pos_embed(token_indices, batch_size)
        else:
            pos_embed = self.pos_embed
        
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and classification
        x = self.norm(x)
        cls_output = x[:, 0]  # Use CLS token for classification
        logits = self.head(cls_output)
        
        return logits
    
    def _select_tokens(self, tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Select tokens based on importance indices"""
        batch_size, num_tokens, embed_dim = tokens.shape
        max_selected = indices.shape[1]
        
        # Expand batch indices
        batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1)
        batch_indices = batch_indices.expand(-1, max_selected)
        
        # Select tokens
        selected_tokens = tokens[batch_indices, indices]
        
        return selected_tokens
    
    def _get_selected_pos_embed(self, token_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Get positional embeddings for selected tokens"""
        # CLS token embedding (always index 0)
        cls_pos_embed = self.pos_embed[:, 0:1, :]  # [1, 1, embed_dim]
        
        # Selected token embeddings
        max_selected = token_indices.shape[1]
        selected_pos_embed = torch.zeros(batch_size, max_selected, self.embed_dim, 
                                       device=token_indices.device)
        
        for i in range(batch_size):
            # Get positional embeddings for selected indices (+1 for CLS token offset)
            selected_indices = token_indices[i] + 1
            selected_pos_embed[i] = self.pos_embed[0, selected_indices, :]
        
        # Concatenate CLS and selected positional embeddings
        cls_pos_embed = cls_pos_embed.expand(batch_size, -1, -1)
        pos_embed = torch.cat([cls_pos_embed, selected_pos_embed], dim=1)
        
        return pos_embed
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ViTSmallCustom',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'num_patches': self.num_patches,
            'max_tokens': self.max_tokens
        }

class TransformerBlock(nn.Module):
    """Standard Transformer block"""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x

def create_vit_model(model_type: str, config) -> nn.Module:
    """
    Factory function to create ViT variants
    
    Args:
        model_type: "baseline" or "custom"
        config: Configuration object
        
    Returns:
        Initialized ViT model
    """
    if model_type == "baseline":
        return ViTSmallBaseline(
            num_classes=config.num_classes,
            img_size=config.img_size,
            patch_size=config.patch_size,
            pretrained=getattr(config, 'pretrained', True),
            embed_dim=getattr(config, 'embed_dim', 384)
        )
    elif model_type == "custom":
        return ViTSmallCustom(
            num_classes=config.num_classes,
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=getattr(config, 'embed_dim', 384),
            depth=getattr(config, 'depth', 12),
            num_heads=getattr(config, 'num_heads', 6)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
