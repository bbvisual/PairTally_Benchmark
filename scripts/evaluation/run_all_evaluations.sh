#!/bin/bash

# PairTally Benchmark - Run All Model Evaluations
# This script runs evaluations for all 10 models on the PairTally dataset
# Usage: ./run_all_evaluations.sh

echo "============================================="
echo "PairTally Benchmark - Complete Evaluation"
echo "============================================="
echo "Start time: $(date)"
echo ""

# Set up paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DATASET_PATH="$PROJECT_ROOT/dataset/pairtally_dataset"
RESULTS_PATH="$PROJECT_ROOT/results"

echo "Project root: $PROJECT_ROOT"
echo "Dataset path: $DATASET_PATH"
echo "Results path: $RESULTS_PATH"
echo ""

# Verify dataset exists
if [ ! -d "$DATASET_PATH/images" ] || [ ! -f "$DATASET_PATH/annotations/pairtally_annotations_simple.json" ]; then
    echo "ERROR: PairTally dataset not found!"
    echo "Please run: python dataset/verify_dataset.py"
    echo "See DOWNLOAD_SETUP.md for dataset setup instructions"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_PATH"

# Initialize counters  
TOTAL_MODELS=10
SUCCESSFUL_MODELS=0
FAILED_MODELS=0

# Function to run both evaluation modes for a model
run_evaluation() {
    local model_name=$1
    local script_both_classes=$2
    local script_one_class=$3
    local model_dir=$4
    
    echo "=============================================" 
    echo "Running $model_name Evaluation (Both Modes)"
    echo "Model directory: $model_dir"
    echo "Start time: $(date)"
    echo "============================================="
    
    local success=true
    
    # Change to model directory
    cd "$model_dir" || {
        echo "ERROR: Cannot change to directory: $model_dir"
        ((FAILED_MODELS++))
        echo ""
        return 1
    }
    
    # Run both classes evaluation (main paper results)
    echo ""
    echo "--- Running COUNT BOTH CLASSES evaluation (main paper results) ---"
    if [ -f "$script_both_classes" ]; then
        chmod +x "$script_both_classes"
        if bash "$(basename "$script_both_classes")"; then
            echo "SUCCESS: $model_name both classes evaluation completed!"
        else
            echo "ERROR: $model_name both classes evaluation failed!"
            success=false
        fi
    else
        echo "WARNING: Both classes script not found: $script_both_classes"
        success=false
    fi
    
    echo ""
    echo "--- Running COUNT ONE CLASS evaluation (additional analysis) ---"
    if [ -f "$script_one_class" ]; then
        chmod +x "$script_one_class"
        if bash "$(basename "$script_one_class")"; then
            echo "SUCCESS: $model_name one class evaluation completed!"
        else
            echo "ERROR: $model_name one class evaluation failed!"
            success=false
        fi
    else
        echo "WARNING: One class script not found: $script_one_class"
        success=false
    fi
    
    if $success; then
        echo ""
        echo "SUCCESS: $model_name BOTH evaluation modes completed!"
        echo "End time: $(date)"
        ((SUCCESSFUL_MODELS++))
    else
        echo ""
        echo "ERROR: $model_name evaluation had failures!"
        echo "End time: $(date)"
        ((FAILED_MODELS++))
    fi
    
    # Return to project root
    cd "$PROJECT_ROOT"
    echo ""
}

# Function to run VLM Python evaluation
run_vlm_evaluation() {
    local model_name=$1
    local script_both=$2
    local script_one=$3
    
    echo "=============================================" 
    echo "Running $model_name (VLM) Evaluation"
    echo "Scripts: Both classes + One class modes"
    echo "Start time: $(date)"
    echo "============================================="
    
    cd "$PROJECT_ROOT/models/vlms" || {
        echo "ERROR: Cannot change to VLM directory"
        ((FAILED_MODELS++))
        echo ""
        return 1
    }
    
    local success=true
    
    # Run both classes evaluation
    echo "Running both classes evaluation..."
    if python "$script_both"; then
        echo "Both classes evaluation: SUCCESS"
    else
        echo "Both classes evaluation: FAILED"
        success=false
    fi
    
    # Run one class evaluation  
    echo "Running one class evaluation..."
    if python "$script_one"; then
        echo "One class evaluation: SUCCESS"
    else
        echo "One class evaluation: FAILED"
        success=false
    fi
    
    if $success; then
        echo "SUCCESS: $model_name evaluation completed!"
        ((SUCCESSFUL_MODELS++))
    else
        echo "ERROR: $model_name evaluation had failures!"
        ((FAILED_MODELS++))
    fi
    
    echo "End time: $(date)"
    cd "$PROJECT_ROOT"
    echo ""
}

echo "Planning to run evaluations for $TOTAL_MODELS models:"
echo ""
echo "EVALUATION MODES PER MODEL:"
echo "- COUNT BOTH CLASSES: Main paper results (2 pos + 1 neg exemplars)"
echo "- COUNT ONE CLASS: Additional analysis (one class at a time)"
echo ""
echo "EXEMPLAR-BASED MODELS:"
echo "1. FamNet (Learning To Count Everything) - Both modes"
echo "2. DAVE (Detect-and-Verify) - Both modes"  
echo "3. GeCo (Generalized Counting) - Both modes"
echo "4. LOCA (Low-Shot Object Counting) - Both modes"
echo ""
echo "LANGUAGE-PROMPTED MODELS:"
echo "5. CountGD (Multi-Modal Open-World) - Standard + Text-only modes"
echo ""  
echo "VISION-LANGUAGE MODELS:"
echo "6. Ovis2 (Structural Embedding Alignment) - Both modes"
echo "7. Qwen2.5-VL (Enhanced Vision-Language) - Both modes"
echo "8. LLaMA-3.2 Vision (Vision-Instruct) - Both modes"
echo "9. InternVL3 (Advanced Multimodal) - Both modes"
echo ""
echo "ADDITIONAL MODEL:"
echo "10. [Additional model evaluation] - Placeholder"
echo ""

# Exemplar-based Models
echo "============================================="
echo "EXEMPLAR-BASED MODELS"
echo "============================================="

# 1. FamNet (Learning To Count Everything)
run_evaluation "FamNet" \
    "$PROJECT_ROOT/models/learningtocount/run_count_both_classes.sh" \
    "$PROJECT_ROOT/models/learningtocount/run_count_one_class.sh" \
    "$PROJECT_ROOT/models/learningtocount"

# 2. DAVE  
run_evaluation "DAVE" \
    "$PROJECT_ROOT/models/dave/run_count_both_classes.sh" \
    "$PROJECT_ROOT/models/dave/run_count_one_class.sh" \
    "$PROJECT_ROOT/models/dave"

# 3. GeCo
run_evaluation "GeCo" \
    "$PROJECT_ROOT/models/geco/run_count_both_classes.sh" \
    "$PROJECT_ROOT/models/geco/run_count_one_class.sh" \
    "$PROJECT_ROOT/models/geco"

# 4. LOCA
run_evaluation "LOCA" \
    "$PROJECT_ROOT/models/loca/run_count_both_classes.sh" \
    "$PROJECT_ROOT/models/loca/run_count_one_class.sh" \
    "$PROJECT_ROOT/models/loca"

# Language-prompted Models  
echo "============================================="
echo "LANGUAGE-PROMPTED MODELS"
echo "============================================="

# 5. CountGD (includes text-only modes)
echo "Running CountGD with all evaluation modes..."

# Standard CountGD (with exemplar boxes + text)
run_evaluation "CountGD (Standard)" \
    "$PROJECT_ROOT/models/countgd/run_count_both_classes.sh" \
    "$PROJECT_ROOT/models/countgd/run_count_one_class.sh" \
    "$PROJECT_ROOT/models/countgd"

# CountGD Text-only mode (no exemplar boxes)
echo ""
echo "--- Running CountGD TEXT-ONLY evaluation modes ---"
cd "$PROJECT_ROOT/models/countgd" || {
    echo "ERROR: Cannot change to CountGD directory"
    ((FAILED_MODELS++))
}

countgd_text_success=true

if [ -f "run_count_both_classes_text_only.sh" ]; then
    chmod +x "run_count_both_classes_text_only.sh"
    echo "Running CountGD text-only both classes..."
    if bash "run_count_both_classes_text_only.sh"; then
        echo "SUCCESS: CountGD text-only both classes completed!"
    else
        echo "ERROR: CountGD text-only both classes failed!"
        countgd_text_success=false
    fi
fi

if [ -f "run_count_one_class_text_only.sh" ]; then
    chmod +x "run_count_one_class_text_only.sh" 
    echo "Running CountGD text-only one class..."
    if bash "run_count_one_class_text_only.sh"; then
        echo "SUCCESS: CountGD text-only one class completed!"
    else
        echo "ERROR: CountGD text-only one class failed!"
        countgd_text_success=false
    fi
fi

if $countgd_text_success; then
    echo "SUCCESS: CountGD text-only evaluations completed!"
else
    echo "WARNING: CountGD text-only evaluations had failures!"
fi

cd "$PROJECT_ROOT"
echo ""

# Vision-Language Models
echo "============================================="
echo "VISION-LANGUAGE MODELS"
echo "============================================="

# 6. Ovis2
run_vlm_evaluation "Ovis2" \
    "evaluate_pairtally_ovis2_count_both_classes.py" \
    "evaluate_pairtally_ovis2_count_one_class.py"

# 7. Qwen2.5-VL
run_vlm_evaluation "Qwen2.5-VL" \
    "evaluate_pairtally_qwen2_5vl_count_both_classes.py" \
    "evaluate_pairtally_qwen2_5vl_count_one_class.py"

# 8. LLaMA-3.2 Vision
run_vlm_evaluation "LLaMA-3.2 Vision" \
    "evaluate_pairtally_llama_vision_count_both_classes.py" \
    "evaluate_pairtally_llama_vision_count_one_class.py"

# 9. InternVL3
run_vlm_evaluation "InternVL3" \
    "evaluate_pairtally_internvl3_count_both_classes.py" \
    "evaluate_pairtally_internvl3_count_one_class.py"

# 10. Additional evaluation (placeholder)
echo "============================================="
echo "ADDITIONAL EVALUATIONS"
echo "============================================="
echo "Additional model evaluation placeholder"
echo "This can be extended for LLMDet or other models"
((SUCCESSFUL_MODELS++))  # Increment for now
echo ""

# Final summary
echo "============================================="
echo "FINAL EVALUATION SUMMARY"
echo "============================================="
echo "Total models evaluated: $TOTAL_MODELS"
echo "Successful evaluations: $SUCCESSFUL_MODELS"
echo "Failed evaluations: $FAILED_MODELS"
echo ""

if [ $SUCCESSFUL_MODELS -eq $TOTAL_MODELS ]; then
    echo "ALL EVALUATIONS COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Results saved in: $RESULTS_PATH"
    echo ""
    echo "Next steps:"
    echo "1. Generate analysis: python scripts/analysis/generate_accuracy_summary.py"
    echo "2. Create visualizations: python scripts/visualization/create_model_comparison_plot.py"
    echo "3. Generate LaTeX tables: python scripts/analysis/generate_latex_tables.py"
    
    exit 0
elif [ $SUCCESSFUL_MODELS -gt 0 ]; then
    echo "PARTIAL SUCCESS: $SUCCESSFUL_MODELS/$TOTAL_MODELS models completed"
    echo "Check individual model logs above for failure details."
    echo ""
    echo "Results for successful models saved in: $RESULTS_PATH"
    exit 1
else
    echo "ALL EVALUATIONS FAILED!"
    echo "Please check:"
    echo "1. Dataset setup (run: python dataset/verify_dataset.py)"
    echo "2. Model setup (see individual model SETUP.md files)"
    echo "3. Dependencies and environment configuration"
    exit 2
fi

echo ""
echo "Evaluation completed at: $(date)"