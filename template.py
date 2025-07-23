#!/usr/bin/env python3
"""
Modified script to create folder structure and empty files for CNN-ViT Token Pruning project.
Assumes root directory already exists - only creates subdirectories and files.
"""

import os
from pathlib import Path

def create_project_structure():
    """Create subdirectories and empty files only (assumes root exists)"""
    
    # Use current directory as project root (assuming you're already in cnn_vit_token_pruning/)
    project_root = Path(".")
    
    print(f"🚀 Setting up project structure in: {project_root.absolute()}")
    
    # Define all subdirectories (no root creation)
    directories = [
        "models",
        "experiments", 
        "evaluation",
        "config",
        "utils",
        "data/raw/cifar10",
        "data/processed/cifar10", 
        "data/metadata",
        "results/resnet18/checkpoints",
        "results/resnet18/logs",
        "results/vit_small/checkpoints",
        "results/vit_small/logs", 
        "results/cnn_vit_hybrid/checkpoints",
        "results/cnn_vit_hybrid/logs",
        "results/comparison/visualizations",
        "notebooks",
        "scripts"
    ]
    
    # Create subdirectories
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {full_path}")
    
    # Define all files to create (empty)
    files = [
        # Model files
        "models/__init__.py",
        "models/resnet18.py", 
        "models/vit_small.py",
        "models/cnn_vit_hybrid.py",
        "models/model_factory.py",
        
        # Experiment files
        "experiments/__init__.py",
        "experiments/train_resnet18.py",
        "experiments/train_vit_small.py", 
        "experiments/train_hybrid.py",
        "experiments/compare_models.py",
        
        # Evaluation files
        "evaluation/__init__.py",
        "evaluation/metrics.py",
        "evaluation/evaluator.py",
        "evaluation/comparator.py",
        
        # Config files
        "config/__init__.py",
        "config/base_config.py",
        "config/model_configs.py",
        "config/training_configs.py",
        
        # Utility files  
        "utils/__init__.py",
        "utils/data_utils.py",
        "utils/logging_utils.py",
        "utils/visualization.py",
        "utils/metrics_tracker.py",
        
        # Data files
        "data/__init__.py",
        "data/datasets.py",
        "data/transforms.py",
        
        # Results files (JSON placeholders)
        "results/resnet18/metrics.json",
        "results/vit_small/metrics.json", 
        "results/cnn_vit_hybrid/metrics.json",
        "results/comparison/performance_comparison.json",
        "results/comparison/efficiency_analysis.json",
        
        # Notebook files
        "notebooks/01_data_exploration.ipynb",
        "notebooks/02_model_training.ipynb",
        "notebooks/03_results_analysis.ipynb",
        
        # Script files
        "scripts/setup_environment.sh",
        "scripts/download_data.py",
        "scripts/run_experiments.sh",
        
        # Project files (in root)
        "README.md",
        "requirements.txt", 
        ".gitignore",
        "setup.py"
    ]
    
    # Create empty files
    for file_path in files:
        full_path = project_root / file_path
        # Only create if file doesn't already exist
        if not full_path.exists():
            full_path.touch()
            print(f"📄 Created file: {full_path}")
        else:
            print(f"📄 File already exists: {full_path}")
    
    print(f"\n✅ Created {len(directories)} directories and {len(files)} files")
    print(f"📂 Project root: {project_root.absolute()}")

def show_structure():
    """Display the created structure"""
    project_root = Path(".")
    print(f"\n📁 Project structure in current directory:")
    
    def print_tree(path, prefix="", is_last=True):
        if path.name.startswith('.') and path.name not in ['.gitignore']:
            return
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{path.name}")
        
        if path.is_dir():
            children = sorted([p for p in path.iterdir() 
                             if not p.name.startswith('.') or p.name == '.gitignore'])
            for i, child in enumerate(children):
                is_last_child = i == len(children) - 1
                extension = "    " if is_last else "│   "
                print_tree(child, prefix + extension, is_last_child)
    
    # Show current directory contents
    children = sorted([p for p in project_root.iterdir() 
                      if not p.name.startswith('.') or p.name == '.gitignore'])
    for i, child in enumerate(children):
        is_last_child = i == len(children) - 1
        print_tree(child, "", is_last_child)

if __name__ == "__main__":
    print("🚀 Setting up CNN-ViT Token Pruning project structure...")
    print("📂 Working in existing root directory")
    create_project_structure()
    show_structure()
