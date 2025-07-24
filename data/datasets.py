"""
Dataset classes for CIFAR-10 with patch-based processing capabilities.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, Any
from utils.data_utils import extract_patches, get_dataset_info

class CIFAR10Handler:
    """Main CIFAR-10 dataset handler with multiple loading modes"""
    
    def __init__(self, data_dir: str = "data/raw/cifar10", download: bool = True):
        self.data_dir = data_dir
        self.download = download
        self.dataset_info = get_dataset_info("cifar10")
        
        # Standard CIFAR-10 transforms for CNN baseline
        self.standard_train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.dataset_info['mean'], self.dataset_info['std'])
        ])
        
        self.standard_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.dataset_info['mean'], self.dataset_info['std'])
        ])
        
        # ViT-specific transforms (no data augmentation for patches)
        self.vit_train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.dataset_info['mean'], self.dataset_info['std'])
        ])
        
        self.vit_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.dataset_info['mean'], self.dataset_info['std'])
        ])
    
    def get_standard_loaders(self, batch_size: int = 64, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        """Get standard CIFAR-10 dataloaders for CNN baseline"""
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=self.download,
            transform=self.standard_train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=self.download,
            transform=self.standard_test_transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"✅ Standard CIFAR-10 loaders created: {len(train_dataset)} train, {len(test_dataset)} test")
        return train_loader, test_loader
    
    def get_patch_loaders(self, patch_size: int = 4, batch_size: int = 64, 
                         num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        """Get patch-based dataloaders for ViT and hybrid models"""
        train_dataset = PatchCIFAR10(
            root=self.data_dir,
            train=True,
            download=self.download,
            patch_size=patch_size,
            transform=self.vit_train_transform
        )
        
        test_dataset = PatchCIFAR10(
            root=self.data_dir,
            train=False,
            download=self.download,
            patch_size=patch_size,
            transform=self.vit_test_transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"✅ Patch-based CIFAR-10 loaders created: patch_size={patch_size}, {train_dataset.num_patches} patches/image")
        return train_loader, test_loader

class PatchCIFAR10(torchvision.datasets.CIFAR10):
    """CIFAR-10 dataset with patch extraction for ViT training"""
    
    def __init__(self, root: str, train: bool = True, transform=None, 
                 target_transform=None, download: bool = False, patch_size: int = 4):
        super().__init__(root, train, transform, target_transform, download)
        self.patch_size = patch_size
        self.img_size = 32
        
        if self.img_size % patch_size != 0:
            raise ValueError(f"Image size {self.img_size} must be divisible by patch size {patch_size}")
        
        self.num_patches = (self.img_size // patch_size) ** 2
        self.patches_per_side = self.img_size // patch_size
        
        print(f"PatchCIFAR10 initialized: {self.num_patches} patches per image ({self.patches_per_side}x{self.patches_per_side} grid)")
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns:
            Dictionary containing:
            - 'image': Original transformed image [C, H, W]
            - 'patches': Extracted patches [num_patches, C, patch_size, patch_size]
            - 'target': Class label
            - 'num_patches': Number of patches (constant)
            - 'patch_positions': Spatial positions of patches
        """
        # Get original image and target
        img, target = super().__getitem__(index)
        
        # Extract patches from the transformed image
        patches = self._extract_patches(img)
        
        # Create patch position information (for positional embeddings)
        patch_positions = self._get_patch_positions()
        
        return {
            'image': img,
            'patches': patches,
            'target': target,
            'num_patches': self.num_patches,
            'patch_positions': patch_positions
        }
    
    def _extract_patches(self, img: torch.Tensor) -> torch.Tensor:
        """Extract non-overlapping patches from image"""
        # img shape: [C, H, W]
        C, H, W = img.shape
        p = self.patch_size
        
        # Reshape to patches: [num_patches, C, patch_size, patch_size]
        patches = img.unfold(1, p, p).unfold(2, p, p)  # [C, H/p, W/p, p, p]
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()  # [H/p, W/p, C, p, p]
        patches = patches.view(-1, C, p, p)  # [num_patches, C, p, p]
        
        return patches
    
    def _get_patch_positions(self) -> torch.Tensor:
        """Get 2D positions of patches for positional embeddings"""
        positions = []
        for i in range(self.patches_per_side):
            for j in range(self.patches_per_side):
                positions.append([i, j])
        return torch.tensor(positions, dtype=torch.long)

class HybridCIFAR10(torchvision.datasets.CIFAR10):
    """
    Extended CIFAR-10 dataset for CNN-ViT hybrid training
    CRITICAL FIX: Direct inheritance from base CIFAR10 to avoid transform conflicts
    """
    
    def __init__(self, root: str, train: bool = True, transform=None,
                 target_transform=None, download: bool = False, patch_size: int = 4,
                 cnn_transform=None):
        # Initialize base CIFAR10 class WITHOUT any transforms initially
        super().__init__(root, train, None, target_transform, download)
        
        # Store transforms separately
        self.vit_transform = transform
        self.cnn_transform = cnn_transform or transform
        self.patch_size = patch_size
        
        # Calculate patch information
        self.img_size = 32
        if self.img_size % patch_size != 0:
            raise ValueError(f"Image size {self.img_size} must be divisible by patch size {patch_size}")
        
        self.num_patches = (self.img_size // patch_size) ** 2
        self.patches_per_side = self.img_size // patch_size
        
        print(f"HybridCIFAR10 initialized: {self.num_patches} patches per image ({self.patches_per_side}x{self.patches_per_side} grid)")
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns data for both CNN and ViT processing
        CRITICAL FIX: Get raw PIL image first, then apply transforms separately
        """
        # Get raw PIL image and target from base CIFAR10 class
        # This ensures we always start with a PIL Image
        img, target = super().__getitem__(index)
        
        # Verify we have a PIL Image (safety check)
        if not hasattr(img, 'mode'):  # PIL Images have .mode attribute
            raise TypeError(f"Expected PIL Image, got {type(img)}. Check dataset setup.")
        
        # Apply transforms to the same PIL image separately
        cnn_img = self.cnn_transform(img)
        vit_img = self.vit_transform(img)
        
        # Extract patches for ViT from the transformed image
        patches = self._extract_patches(vit_img)
        patch_positions = self._get_patch_positions()
        
        return {
            'cnn_image': cnn_img,  # For CNN ROI scoring
            'vit_image': vit_img,  # For ViT processing
            'patches': patches,    # Extracted patches
            'target': target,
            'num_patches': self.num_patches,
            'patch_positions': patch_positions
        }
    
    def _extract_patches(self, img: torch.Tensor) -> torch.Tensor:
        """Extract non-overlapping patches from image"""
        # img shape: [C, H, W]
        C, H, W = img.shape
        p = self.patch_size
        
        # Reshape to patches: [num_patches, C, patch_size, patch_size]
        patches = img.unfold(1, p, p).unfold(2, p, p)  # [C, H/p, W/p, p, p]
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()  # [H/p, W/p, C, p, p]
        patches = patches.view(-1, C, p, p)  # [num_patches, C, p, p]
        
        return patches
    
    def _get_patch_positions(self) -> torch.Tensor:
        """Get 2D positions of patches for positional embeddings"""
        positions = []
        for i in range(self.patches_per_side):
            for j in range(self.patches_per_side):
                positions.append([i, j])
        return torch.tensor(positions, dtype=torch.long)

def create_data_loaders(config, loader_type: str = "standard") -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create appropriate data loaders based on configuration
    
    Args:
        config: Configuration object (BaseConfig or subclass)
        loader_type: "standard" for CNN, "patch" for ViT, "hybrid" for CNN-ViT
        
    Returns:
        train_loader, test_loader
    """
    handler = CIFAR10Handler(data_dir=config.data_dir, download=True)
    
    # Disable pin_memory for MPS devices to avoid warnings
    use_pin_memory = not (config.device == 'mps')
    
    if loader_type == "standard":
        return handler.get_standard_loaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
    elif loader_type == "patch":
        return handler.get_patch_loaders(
            patch_size=config.patch_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
    elif loader_type == "hybrid":
        # For hybrid training using the completely fixed HybridCIFAR10 dataset
        train_dataset = HybridCIFAR10(
            root=config.data_dir,
            train=True,
            download=True,
            patch_size=config.patch_size,
            transform=handler.vit_train_transform,
            cnn_transform=handler.standard_train_transform
        )
        
        test_dataset = HybridCIFAR10(
            root=config.data_dir,
            train=False,
            download=True,
            patch_size=config.patch_size,
            transform=handler.vit_test_transform,
            cnn_transform=handler.standard_test_transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=use_pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=use_pin_memory
        )
        
        print(f"✅ Hybrid CIFAR-10 loaders created for CNN-ViT training")
        return train_loader, test_loader
    else:
        raise ValueError(f"Unknown loader_type: {loader_type}")
