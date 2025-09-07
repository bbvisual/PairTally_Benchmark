#!/bin/bash

# LearningToCountEverything Combined Evaluation Script for DICTA25
# This script runs combined evaluation with 2 positive + 1 negative exemplars on the final dataset

echo "🚀 Starting LearningToCountEverything Combined Evaluation (2 Positive + 1 Negative Exemplars) on FINAL DATASET..."

# Default settings
MODEL_PATH="/home/khanhnguyen/DICTA25/LearningToCountEverything/data/pretrainedModels/FamNet_Save1.pth"
MODEL_NAME="LearningToCountEverything"
BASE_DATA_PATH="/home/khanhnguyen/DICTA25/final_dataset_default"
OUTPUT_LIMIT=""
ADAPT_FLAG=""
GRADIENT_STEPS="100"
LEARNING_RATE="1e-7"
WEIGHT_MINCOUNT="1e-9"
WEIGHT_PERTURBATION="1e-4"
GPU_ID="0"

# Only run on the final dataset
DATASETS=("final_dataset_default")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --base_data_path)
            BASE_DATA_PATH="$2"
            shift 2
            ;;
        --output_limit)
            OUTPUT_LIMIT="--output_limit $2"
            shift 2
            ;;
        --adapt)
            ADAPT_FLAG="--adapt"
            shift
            ;;
        --gradient_steps)
            GRADIENT_STEPS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --weight_mincount)
            WEIGHT_MINCOUNT="$2"
            shift 2
            ;;
        --weight_perturbation)
            WEIGHT_PERTURBATION="$2"
            shift 2
            ;;
        --gpu_id)
            GPU_ID="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_name NAME         Model name (default: LearningToCountEverything)"
            echo "  --model_path PATH         Model path (default: /home/khanhnguyen/DICTA25/LearningToCountEverything/FamNet_Save500.pth)"
            echo "  --base_data_path P        Base path for dataset (default: /home/khanhnguyen/DICTA25/final_dataset_default)"
            echo "  --output_limit N          Limit processing to N images (for testing)"
            echo "  --adapt                   Enable test-time adaptation"
            echo "  --gradient_steps N        Number of gradient steps for adaptation (default: 100)"
            echo "  --learning_rate LR        Learning rate for adaptation (default: 1e-7)"
            echo "  --weight_mincount W       Weight for mincount loss (default: 1e-9)"
            echo "  --weight_perturbation W   Weight for perturbation loss (default: 1e-4)"
            echo "  --gpu_id ID               GPU ID to use (default: 0, use -1 for CPU)"
            echo "  --help                    Show this help message"
            echo ""
            echo "Example: $0 --output_limit 10 --adapt"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "📂 Model path: $MODEL_PATH"
echo "🎯 Model name: $MODEL_NAME"
echo "📊 Base data path: $BASE_DATA_PATH"
echo "🎯 Dataset to evaluate: final_dataset_default"
echo "🔗 Evaluation type: Combined (2 positive + 1 negative exemplars)"
echo "🔧 Test-time adaptation: $([ -n "$ADAPT_FLAG" ] && echo "Enabled" || echo "Disabled")"
echo "💻 GPU ID: $GPU_ID"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file $MODEL_PATH does not exist!"
    exit 1
fi

echo "✅ Model file found: $MODEL_PATH"
echo ""

# Define paths for final dataset
DATA_PATH="$BASE_DATA_PATH"
ANNOTATION_FILE="$DATA_PATH/annotations/annotation_FSC147_384.json"
IMAGE_DIR="$DATA_PATH/images_384_VarV2"

# Check if data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ Warning: Data path $DATA_PATH does not exist!"
    echo "Expected structure:"
    echo "$DATA_PATH/"
    echo "├── images_384_VarV2/"
    echo "└── annotations/"
    echo "    └── annotation_FSC147_384.json"
    echo "Skipping final_dataset_default..."
    echo ""
    exit 1
fi

# Check if annotation file exists
if [ ! -f "$ANNOTATION_FILE" ]; then
    echo "❌ Warning: Annotation file $ANNOTATION_FILE does not exist!"
    echo "Skipping final_dataset_default..."
    echo ""
    exit 1
fi

# Check if image directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ Warning: Image directory $IMAGE_DIR does not exist!"
    echo "Skipping final_dataset_default..."
    echo ""
    exit 1
fi

echo "✅ Annotation file found: $ANNOTATION_FILE"
echo "✅ Image directory found: $IMAGE_DIR"

# Create results directories for combined evaluation with clear identifiers
mkdir -p "/home/khanhnguyen/DICTA25-RESULTS/LearningToCountEverything-qualitative-combined/final_dataset_default"
mkdir -p "/home/khanhnguyen/DICTA25-RESULTS/LearningToCountEverything-quantitative-combined/final_dataset_default"

echo "📁 Results will be saved to:"
echo "   Qualitative: /home/khanhnguyen/DICTA25-RESULTS/LearningToCountEverything-qualitative-combined/final_dataset_default/"
echo "   Quantitative: /home/khanhnguyen/DICTA25-RESULTS/LearningToCountEverything-quantitative-combined/final_dataset_default/"
echo ""

# Run combined evaluation
echo "🏃 Running combined evaluation (2 positive + 1 negative exemplars)..."
python evaluate_DICTA25_combined.py \
    --annotation_file "$ANNOTATION_FILE" \
    --image_dir "$IMAGE_DIR" \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --gradient_steps "$GRADIENT_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_mincount "$WEIGHT_MINCOUNT" \
    --weight_perturbation "$WEIGHT_PERTURBATION" \
    --gpu_id "$GPU_ID" \
    $ADAPT_FLAG \
    $OUTPUT_LIMIT

if [ $? -eq 0 ]; then
    echo "🎉 LearningToCountEverything combined evaluation completed successfully for final_dataset_default!"
    echo "📁 Qualitative results saved to: /home/khanhnguyen/DICTA25-RESULTS/LearningToCountEverything-qualitative-combined/final_dataset_default/"
    echo "📊 Quantitative results saved to: /home/khanhnguyen/DICTA25-RESULTS/LearningToCountEverything-quantitative-combined/final_dataset_default/"
else
    echo "❌ Combined evaluation failed for final_dataset_default with exit code $?"
    exit 1
fi

echo ""
echo "========================================"
echo "🎉 Combined evaluation completed!"
echo "========================================"
echo ""
echo "📋 Summary:"
echo "   Dataset: final_dataset_default"
echo "   Model: $MODEL_NAME"
echo "   Evaluation Type: Combined (2 positive + 1 negative exemplars)"
echo "   Test-time adaptation: $([ -n "$ADAPT_FLAG" ] && echo "Enabled" || echo "Disabled")"
echo "   Results Location: /home/khanhnguyen/DICTA25-RESULTS/LearningToCountEverything-*-combined/final_dataset_default/"
echo ""
echo "📄 Files created:"
echo "   - combined_inference_data.json"
echo "   - combined_inference_data.pkl"
echo "   - {model_name}_combined_quantitative_results.json"
echo "   - {model_name}_combined_summary.txt"

# Visualization (optional)
CREATE_VIS="n"
if [[ "$CREATE_VIS" =~ ^[Yy]$ ]]; then
    results_dir="/home/khanhnguyen/DICTA25-RESULTS/LearningToCountEverything-qualitative-combined/final_dataset_default"
    if [ -d "$results_dir" ]; then
        python visualize_ltce_combined_results.py \
            --results_dir "$results_dir" \
            --sample_size 5 2>/dev/null || echo "  Warning: Visualization script not found or failed for final_dataset_default"
    fi
    echo "✅ Visualizations created for final_dataset_default!"
    echo "   Location: /home/khanhnguyen/DICTA25-RESULTS/LearningToCountEverything-qualitative-combined/final_dataset_default/visualizations/"
fi

echo "🏁 All combined evaluations complete!"
