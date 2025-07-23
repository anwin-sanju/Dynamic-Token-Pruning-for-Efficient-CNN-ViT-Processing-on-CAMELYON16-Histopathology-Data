#!/usr/bin/env python3
"""
Setup script for CNN-ViT Dynamic Token Pruning research project.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cnn-vit-token-pruning",
    version="0.1.0",
    author="Anwin Sanju",
    author_email="dev.anwinsanju@gmail.com",
    description="CNN-ViT Dynamic Token Pruning for Efficient Medical Image Analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/anwin-sanju/Dynamic-Token-Pruning-for-Efficient-CNN-ViT-Processing-on-CAMELYON16-Histopathology-Data",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.20.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.7.0",
        ],
        "viz": [
            "plotly>=5.10.0",
            "dash>=2.6.0",
            "streamlit>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-resnet18=experiments.train_resnet18:main",
            "train-vit=experiments.train_vit_small:main",
            "train-hybrid=experiments.train_hybrid:main",
            "compare-models=experiments.compare_models:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    keywords=[
        "deep learning",
        "vision transformer",
        "token pruning",
        "medical imaging",
        "histopathology",
        "CNN",
        "ViT",
        "efficient inference",
        "CAMELYON",
        "research"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/cnn-vit-token-pruning/issues",
        "Source": "https://github.com/yourusername/cnn-vit-token-pruning",
        "Documentation": "https://github.com/yourusername/cnn-vit-token-pruning/wiki",
    },
)