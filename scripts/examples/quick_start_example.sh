#!/bin/bash

# DICTA25 - Quick Start Example
# This script demonstrates how to get started with the DICTA25 benchmark

echo "=========================================="
echo "DICTA25 Quick Start Example"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Project root: $PROJECT_ROOT"
echo ""

echo "This example will:"
echo "1. Set up one environment (CountGD)"
echo "2. Prepare the dataset"
echo "3. Run a single model evaluation"
echo "4. Generate basic results"
echo ""

read -p "Do you want to continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting..."
    exit 1
fi

echo ""
echo "Step 1: Setting up CountGD environment..."
echo "========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not available. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create CountGD environment
echo "Creating CountGD environment..."
if conda info --envs | grep -q "^countgd "; then
    echo "✓ CountGD environment already exists"
else
    conda create -n countgd python=3.9.19 -y
    echo "✓ Created CountGD environment"
fi

# Activate and install basic requirements
eval "$(conda shell.bash hook)"
conda activate countgd

echo "Installing basic requirements..."
pip install torch torchvision torchaudio numpy matplotlib pillow tqdm

echo "✓ CountGD environment setup completed"
echo ""

echo "Step 2: Preparing dataset..."
echo "============================="

cd "$PROJECT_ROOT/dataset"

if [ -f "annotations/parsed_annotations.json" ]; then
    echo "✓ Dataset annotations found"
else
    echo "❌ Dataset annotations not found!"
    echo "Please ensure the dataset is properly downloaded and placed in the dataset/ directory"
    exit 1
fi

# Get basic dataset statistics
echo "Dataset statistics:"
python tools/get_annotation_statistics.py annotations/parsed_annotations.json | head -20

echo ""
echo "Step 3: Model evaluation setup..."
echo "=================================="

echo "For a complete model evaluation, you would:"
echo "1. Clone the original model repository (e.g., CountGD)"
echo "2. Download pre-trained weights"
echo "3. Copy evaluation scripts from models/countgd/"
echo "4. Run the evaluation"
echo ""

echo "Example commands for CountGD:"
echo "git clone https://github.com/niki-amini-naieni/CountGD.git"
echo "cd CountGD"
echo "# Download weights and setup as per models/countgd/README.md"
echo "cp $PROJECT_ROOT/models/countgd/evaluate_DICTA25_combined.py ."
echo "python evaluate_DICTA25_combined.py"

echo ""
echo "Step 4: Results analysis..."
echo "==========================="

cd "$PROJECT_ROOT"

echo "After running evaluations, you can:"
echo "1. Analyze results: python scripts/analysis/generate_accuracy_summary.py"
echo "2. Create plots: python scripts/visualization/create_model_comparison_plot.py"
echo "3. Generate tables: python scripts/analysis/generate_latex_tables.py"

echo ""
echo "=========================================="
echo "Quick Start Example Completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Follow the detailed setup instructions in models/{model_name}/README.md"
echo "2. Download required model weights"
echo "3. Run full evaluations using scripts/evaluation/run_all_evaluations.sh"
echo ""
echo "For questions, see the main README.md or individual model documentation."

conda deactivate
