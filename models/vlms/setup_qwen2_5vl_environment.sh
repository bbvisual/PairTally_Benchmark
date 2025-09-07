#!/bin/bash

# Qwen2.5-VL Environment Setup Script
# This script sets up the conda environment for Qwen2.5-VL evaluation

echo "Setting up Qwen2.5-VL Environment"
echo "================================="

# Create conda environment
echo "Creating conda environment: qwen2_5vl"
conda create -n qwen2_5vl python=3.10 -y

echo "Activating environment..."
conda activate qwen2_5vl

echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing transformers and related packages..."
pip install transformers>=4.37.0

echo "Installing Qwen-VL utils..."
pip install qwen-vl-utils

echo "Installing other required packages..."
pip install \
    matplotlib \
    pillow \
    numpy \
    scipy \
    datasets \
    accelerate \
    sentencepiece \
    protobuf

echo "Installing flash-attention (optional for better performance)..."
pip install flash-attn --no-build-isolation

echo ""
echo "Environment setup completed!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate qwen2_5vl"
echo ""
echo "To test the setup, run:"
echo "  python -c \"from transformers import Qwen2_5_VLForConditionalGeneration; print('Qwen2.5-VL setup successful!')\"" 