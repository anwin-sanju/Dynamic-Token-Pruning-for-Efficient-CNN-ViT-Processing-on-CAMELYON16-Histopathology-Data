"""
ResNet18 implementations for CNN baseline and ROI scoring.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional, Dict, Any

class ResNet18Baseline(nn.Module):
    """Standard ResNet18 for classification baseline"""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify for CIFAR-10 (smaller input size optimization)
        # Replace first conv layer to handle 32x32 images better
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove maxpool for small images (32x32)
        self.backbone.maxpool = nn.Identity()
        
        # Replace final classification layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
        self.num_classes = num_classes
        self.feature_dim = in_features
        
        print(f"✅ ResNet18Baseline initialized: {num_classes} classes, pretrained={pretrained}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification"""
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before final classification layer"""
        # Forward through all layers except final FC
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # This is now Identity()
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        
        return features
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ResNet18Baseline',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim
        }

class ResNet18ROIScorer(nn.Module):
    """ResNet18 modified for ROI importance scoring in hybrid architecture"""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = True, 
                 freeze_backbone: bool = False):
        super().__init__()
        
        # Load pretrained backbone
        backbone = models.resnet18(pretrained=pretrained)
        
        # Modify for CIFAR-10
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            features = self.backbone(dummy_input)
            self.feature_dim = features.view(features.size(0), -1).size(1)
        
        # Dual heads: classification + importance scoring
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.importance_scorer = nn.Sequential(
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid()  # Importance scores between 0 and 1
        )
        
        # Optional backbone freezing
        if freeze_backbone:
            self.freeze_backbone()
        
        self.num_classes = num_classes
        self._frozen = freeze_backbone
        
        print(f"✅ ResNet18ROIScorer initialized: {num_classes} classes, frozen_backbone={freeze_backbone}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both classification and importance scores
        
        Returns:
            classification_logits: [batch_size, num_classes]
            importance_scores: [batch_size, 1] - importance score per image
        """
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Dual outputs
        classification = self.classifier(features)
        importance = self.importance_scorer(features)
        
        return classification, importance
    
    def get_importance_only(self, x: torch.Tensor) -> torch.Tensor:
        """Get only importance scores (for inference efficiency)"""
        with torch.no_grad():
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
            importance = self.importance_scorer(features)
        return importance
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning only heads"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._frozen = True
        print("🔒 ResNet18ROIScorer backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for full training"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._frozen = False
        print("🔓 ResNet18ROIScorer backbone unfrozen")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ResNet18ROIScorer',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'backbone_frozen': self._frozen
        }

class PatchResNet18ROIScorer(nn.Module):
    """ResNet18 for scoring individual patches (for ViT token selection)"""
    
    def __init__(self, num_classes: int = 10, patch_size: int = 4, pretrained: bool = True):
        super().__init__()
        
        # For very small patches, use a lighter architecture
        if patch_size <= 4:
            # Custom lightweight CNN for tiny patches
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.feature_dim = 64
        else:
            # Modified ResNet18 for larger patches
            backbone = models.resnet18(pretrained=pretrained)
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            
            # Get feature dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, patch_size, patch_size)
                features = self.backbone(dummy_input)
                self.feature_dim = features.view(features.size(0), -1).size(1)
        
        # Importance scorer for patches
        self.importance_scorer = nn.Sequential(
            nn.Linear(self.feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Optional classification head (for auxiliary loss)
        self.classifier = nn.Linear(self.feature_dim, num_classes) if num_classes > 0 else None
        
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        print(f"✅ PatchResNet18ROIScorer initialized: patch_size={patch_size}, feature_dim={self.feature_dim}")
    
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Score importance of patches
        
        Args:
            patches: [batch_size, num_patches, channels, patch_size, patch_size]
            
        Returns:
            importance_scores: [batch_size, num_patches]
        """
        batch_size, num_patches, channels, h, w = patches.shape
        
        # Reshape for processing: [batch_size * num_patches, channels, h, w]
        patches_flat = patches.view(-1, channels, h, w)
        
        # Extract features
        features = self.backbone(patches_flat)
        features = features.view(-1, self.feature_dim)
        
        # Get importance scores
        importance = self.importance_scorer(features)  # [batch_size * num_patches, 1]
        
        # Reshape back: [batch_size, num_patches]
        importance = importance.view(batch_size, num_patches)
        
        return importance
    
    def forward_with_classification(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with both importance and classification"""
        if self.classifier is None:
            raise ValueError("Classification head not initialized")
        
        batch_size, num_patches, channels, h, w = patches.shape
        patches_flat = patches.view(-1, channels, h, w)
        
        # Extract features
        features = self.backbone(patches_flat)
        features = features.view(-1, self.feature_dim)
        
        # Get outputs
        importance = self.importance_scorer(features)
        classification = self.classifier(features)
        
        # Reshape
        importance = importance.view(batch_size, num_patches)
        classification = classification.view(batch_size, num_patches, self.num_classes)
        
        return importance, classification
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'PatchResNet18ROIScorer',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'patch_size': self.patch_size
        }

def create_resnet_model(model_type: str, config) -> nn.Module:
    """
    Factory function to create ResNet variants
    
    Args:
        model_type: "baseline", "roi_scorer", or "patch_scorer"
        config: Configuration object
        
    Returns:
        Initialized ResNet model
    """
    if model_type == "baseline":
        return ResNet18Baseline(
            num_classes=config.num_classes,
            pretrained=getattr(config, 'pretrained', True),
            dropout=getattr(config, 'dropout', 0.1)
        )
    elif model_type == "roi_scorer":
        return ResNet18ROIScorer(
            num_classes=config.num_classes,
            pretrained=getattr(config, 'cnn_pretrained', True),
            freeze_backbone=getattr(config, 'freeze_cnn_stage2', False)
        )
    elif model_type == "patch_scorer":
        return PatchResNet18ROIScorer(
            num_classes=config.num_classes,
            patch_size=getattr(config, 'patch_size', 4),
            pretrained=getattr(config, 'cnn_pretrained', True)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
