#!/bin/bash

# DICTA25 - Run All Model Evaluations
# This script runs evaluations for all 9 models on the DICTA25 dataset
# Usage: ./run_all_evaluations.sh

echo "=========================================="
echo "DICTA25 - Complete Evaluation Pipeline"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Set up paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DATASET_PATH="$PROJECT_ROOT/dataset"
RESULTS_PATH="$PROJECT_ROOT/results"

echo "Project root: $PROJECT_ROOT"
echo "Dataset path: $DATASET_PATH"
echo "Results path: $RESULTS_PATH"
echo ""

# Create results directory
mkdir -p "$RESULTS_PATH"

# Initialize counters
TOTAL_MODELS=9
SUCCESSFUL_MODELS=0
FAILED_MODELS=0

# Function to run evaluation with error handling
run_evaluation() {
    local model_name=$1
    local script_path=$2
    local env_name=$3
    
    echo "===========================================" 
    echo "Running $model_name Evaluation"
    echo "Environment: $env_name"
    echo "Script: $script_path"
    echo "Start time: $(date)"
    echo "==========================================="
    
    if eval "$(conda shell.bash hook)" && conda activate "$env_name"; then
        echo "Successfully activated environment: $env_name"
        
        if bash "$script_path"; then
            echo "$model_name evaluation completed successfully!"
            echo "End time: $(date)"
            ((SUCCESSFUL_MODELS++))
        else
            echo "ERROR: $model_name evaluation failed!"
            echo "End time: $(date)"
            ((FAILED_MODELS++))
        fi
        
        conda deactivate
    else
        echo "ERROR: Failed to activate environment: $env_name"
        ((FAILED_MODELS++))
    fi
    
    echo ""
}

echo "Planning to run evaluations for $TOTAL_MODELS models:"
echo "1. CountGD (countgd environment)"
echo "2. DAVE (dave environment)"
echo "3. GeCo (geco_test environment)"
echo "4. LearningToCountEverything (learningtocount environment)"
echo "5. LOCA (loca environment)"
echo "6. Qwen2.5-VL (qwen2_5vl environment)"
echo "7. InternVL3 (llama-vision environment)"
echo "8. Llama Vision (llama-vision environment)"
echo "9. Ovis2 (ovis2-34b environment)"
echo ""

# Object Counting Models
echo "=========================================="
echo "OBJECT COUNTING MODELS"
echo "=========================================="

# 1. CountGD
run_evaluation "CountGD" "$PROJECT_ROOT/models/countgd/run_combined_eval.sh" "countgd"

# 2. DAVE
run_evaluation "DAVE" "$PROJECT_ROOT/models/dave/run_combined_eval.sh" "dave"

# 3. GeCo
run_evaluation "GeCo" "$PROJECT_ROOT/models/geco/run_combined_eval.sh" "geco_test"

# 4. LearningToCountEverything
run_evaluation "LearningToCountEverything" "$PROJECT_ROOT/models/learningtocount/run_combined_eval.sh" "learningtocount"

# 5. LOCA
run_evaluation "LOCA" "$PROJECT_ROOT/models/loca/run_combined_eval.sh" "loca"

# Vision-Language Models
echo "=========================================="
echo "VISION-LANGUAGE MODELS"
echo "=========================================="

# 6. Qwen2.5-VL
run_evaluation "Qwen2.5-VL" "$PROJECT_ROOT/models/vlms/run_qwen2_5vl_evaluation.sh" "qwen2_5vl"

# 7. InternVL3
run_evaluation "InternVL3" "$PROJECT_ROOT/scripts/evaluation/run_internvl3.sh" "llama-vision"

# 8. Llama Vision
run_evaluation "Llama Vision" "$PROJECT_ROOT/scripts/evaluation/run_llama_vision.sh" "llama-vision"

# 9. Ovis2
run_evaluation "Ovis2" "$PROJECT_ROOT/scripts/evaluation/run_ovis2.sh" "ovis2-34b"

# Final summary
echo "========================================="
echo "FINAL EVALUATION SUMMARY"
echo "========================================="
echo "Total models: $TOTAL_MODELS"
echo "Successful: $SUCCESSFUL_MODELS"
echo "Failed: $FAILED_MODELS"
echo ""

if [ $SUCCESSFUL_MODELS -eq $TOTAL_MODELS ]; then
    echo "ALL EVALUATIONS COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Next steps:"
    echo "1. Run analysis: ./scripts/evaluation/analyze_results.sh"
    echo "2. Generate plots: ./scripts/visualization/create_comparison_plots.sh"
    echo "3. Create tables: ./scripts/analysis/generate_summary_tables.sh"
    
    exit 0
elif [ $SUCCESSFUL_MODELS -gt 0 ]; then
    echo "PARTIAL SUCCESS: $SUCCESSFUL_MODELS/$TOTAL_MODELS models completed"
    echo "Check individual model logs above for failure details."
    exit 1
else
    echo "ALL EVALUATIONS FAILED!"
    echo "Check environment setup and model dependencies."
    exit 2
fi

echo ""
echo "Results saved in: $RESULTS_PATH"
echo "End time: $(date)"
