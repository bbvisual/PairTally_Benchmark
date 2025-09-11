#!/bin/bash

# DICTA25 - Setup All Model Environments
# This script sets up conda environments for all models used in the benchmark
# Usage: ./setup_all_environments.sh

echo "=========================================="
echo "DICTA25 - Environment Setup"
echo "=========================================="
echo "This script will set up conda environments for all 9 models"
echo "Start time: $(date)"
echo ""

# Function to setup environment with error handling
setup_environment() {
    local env_name=$1
    local requirements_file=$2
    local python_version=$3
    local description=$4
    
    echo "===========================================" 
    echo "Setting up $env_name environment"
    echo "Description: $description"
    echo "Python version: $python_version"
    echo "Requirements: $requirements_file"
    echo "==========================================="
    
    # Check if environment already exists
    if conda info --envs | grep -q "^$env_name "; then
        echo "Environment '$env_name' already exists. Skipping..."
        return 0
    fi
    
    # Create conda environment
    if conda create -n "$env_name" python="$python_version" -y; then
        echo "Created conda environment: $env_name"
        
        # Activate environment and install requirements
        if eval "$(conda shell.bash hook)" && conda activate "$env_name"; then
            echo "Activated environment: $env_name"
            
            if [ -f "$requirements_file" ]; then
                echo "Installing requirements from $requirements_file..."
                if pip install -r "$requirements_file"; then
                    echo "Requirements installed successfully"
                else
                    echo "Failed to install requirements"
                    return 1
                fi
            else
                echo "Warning: Requirements file not found: $requirements_file"
                echo "Manual setup required for this environment"
            fi
            
            conda deactivate
            echo "$env_name environment setup completed"
        else
            echo "Failed to activate environment: $env_name"
            return 1
        fi
    else
        echo "Failed to create environment: $env_name"
        return 1
    fi
    
    echo ""
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
REQ_DIR="$PROJECT_ROOT/requirements"

echo "Project root: $PROJECT_ROOT"
echo "Requirements directory: $REQ_DIR"
echo ""

# Create requirements directory if it doesn't exist
mkdir -p "$REQ_DIR/environments"

echo "Setting up environments for all models..."
echo ""

# Object Counting Models
echo "=========================================="
echo "OBJECT COUNTING MODELS"
echo "=========================================="

# 1. CountGD
setup_environment "countgd" "$REQ_DIR/environments/countgd_requirements.txt" "3.9.19" "CountGD - Multi-Modal Open-World Counting"

# 2. DAVE
setup_environment "dave" "$REQ_DIR/environments/dave_requirements.txt" "3.8" "DAVE - Detect-and-Verify Paradigm"

# 3. GeCo
setup_environment "geco_test" "$REQ_DIR/environments/geco_requirements.txt" "3.8" "GeCo - Unified Low-Shot Counting"

# 4. LearningToCountEverything
setup_environment "learningtocount" "$REQ_DIR/environments/learningtocount_requirements.txt" "3.8" "Learning To Count Everything"

# 5. LOCA
setup_environment "loca" "$REQ_DIR/environments/loca_requirements.txt" "3.8" "LOCA - Low-Shot Object Counting"

# Vision-Language Models
echo "=========================================="
echo "VISION-LANGUAGE MODELS"
echo "=========================================="

# 6. Qwen2.5-VL
setup_environment "qwen2_5vl" "$REQ_DIR/environments/qwen2_5vl_requirements.txt" "3.9" "Qwen2.5-VL Vision-Language Model"

# 7. Llama Vision & InternVL3 (shared environment)
setup_environment "llama-vision" "$REQ_DIR/environments/llama_vision_requirements.txt" "3.9" "Llama Vision & InternVL3 Models"

# 8. Ovis2
setup_environment "ovis2-34b" "$REQ_DIR/environments/ovis2_requirements.txt" "3.9" "Ovis2 Vision-Language Model"

echo "=========================================="
echo "ENVIRONMENT SETUP SUMMARY"
echo "=========================================="
echo "All environments have been set up!"
echo ""
echo "Created environments:"
echo "1. countgd (CountGD)"
echo "2. dave (DAVE)"
echo "3. geco_test (GeCo)"
echo "4. learningtocount (LearningToCountEverything)"
echo "5. loca (LOCA)"
echo "6. qwen2_5vl (Qwen2.5-VL)"
echo "7. llama-vision (Llama Vision & InternVL3)"
echo "8. ovis2-34b (Ovis2)"
echo ""
echo "Next steps:"
echo "1. Download model weights for each model"
echo "2. Update configuration files with correct paths"
echo "3. Run evaluations: ./scripts/evaluation/run_all_evaluations.sh"
echo ""
echo "For detailed setup instructions for each model, see:"
echo "- models/countgd/README.md"
echo "- models/dave/README.md"
echo "- models/geco/README.md"
echo "- models/learningtocount/README.md"
echo "- models/loca/README.md"
echo "- models/vlms/README.md"
echo ""
echo "End time: $(date)"
