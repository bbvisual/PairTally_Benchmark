#!/bin/bash

# This script runs custom evaluation with 2 positive + 1 negative exemplars on the final dataset

echo " Starting LearningToCountEverything Combined Evaluation (2 Positive + 1 Negative Exemplars) on FINAL DATASET..."

# Default settings
MODEL_PATH="LearningToCountEverything/data/pretrainedModels/FamNet_Save1.pth"
MODEL_NAME="LearningToCountEverything"
BASE_DATA_PATH="../../dataset/pairtally_dataset"
OUTPUT_LIMIT=""
ADAPT_FLAG=""
GRADIENT_STEPS="100"
LEARNING_RATE="1e-7"
WEIGHT_MINCOUNT="1e-9"
WEIGHT_PERTURBATION="1e-4"
GPU_ID="0"

# Only run on the final dataset
DATASETS=("pairtally_dataset")

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
            echo "  --model_path PATH         Model path "
            echo "  --base_data_path P        Base path for dataset "
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

echo " Model path: $MODEL_PATH"
echo " Model name: $MODEL_NAME"
echo " Base data path: $BASE_DATA_PATH"
echo " Dataset to evaluate: pairtally_dataset"
echo "🔗 Evaluation type: Combined (2 positive + 1 negative exemplars)"
echo "🔧 Test-time adaptation: $([ -n "$ADAPT_FLAG" ] && echo "Enabled" || echo "Disabled")"
echo "💻 GPU ID: $GPU_ID"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo " Error: Model file $MODEL_PATH does not exist!"
    exit 1
fi

echo " Model file found: $MODEL_PATH"
echo ""

# Define paths for final dataset
DATA_PATH="$BASE_DATA_PATH"
ANNOTATION_FILE="$DATA_PATH/annotations/pairtally_annotations_simple.json"
IMAGE_DIR="$DATA_PATH/images"

# Check if data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo " Warning: Data path $DATA_PATH does not exist!"
    echo "Expected structure:"
    echo "$DATA_PATH/"
    echo "├── images/"
    echo "└── annotations/"
    echo "    └── pairtally_annotations_simple.json"
    echo "Skipping pairtally_dataset..."
    echo ""
    exit 1
fi

# Check if annotation file exists
if [ ! -f "$ANNOTATION_FILE" ]; then
    echo " Warning: Annotation file $ANNOTATION_FILE does not exist!"
    echo "Skipping pairtally_dataset..."
    echo ""
    exit 1
fi

# Check if image directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo " Warning: Image directory $IMAGE_DIR does not exist!"
    echo "Skipping pairtally_dataset..."
    echo ""
    exit 1
fi

echo " Annotation file found: $ANNOTATION_FILE"
echo " Image directory found: $IMAGE_DIR"

# Create results directories for custom evaluation with clear identifiers
mkdir -p "../../results/LearningToCountEverything-qualitative-custom/pairtally_dataset"
mkdir -p "../../results/LearningToCountEverything-quantitative-custom/pairtally_dataset"

echo " Results will be saved to:"
echo "   Qualitative: ../../results/LearningToCountEverything-qualitative-custom/pairtally_dataset/"
echo "   Quantitative: ../../results/LearningToCountEverything-quantitative-custom/pairtally_dataset/"
echo ""

# Run custom evaluation
echo "🏃 Running custom evaluation (positive exemplars only, count 1 class)..."
python evaluate_pairtally_count_one_class.py \
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
    echo " LearningToCountEverything custom evaluation completed successfully for pairtally_dataset!"
    echo " Qualitative results saved to: ../../results/LearningToCountEverything-qualitative-custom/pairtally_dataset/"
    echo " Quantitative results saved to: ../../results/LearningToCountEverything-quantitative-custom/pairtally_dataset/"
else
    echo " Combined evaluation failed for pairtally_dataset with exit code $?"
    exit 1
fi

echo ""
echo "========================================"
echo " Combined evaluation completed!"
echo "========================================"
echo ""
echo " Summary:"
echo "   Dataset: pairtally_dataset"
echo "   Model: $MODEL_NAME"
echo "   Evaluation Type: Combined (2 positive + 1 negative exemplars)"
echo "   Test-time adaptation: $([ -n "$ADAPT_FLAG" ] && echo "Enabled" || echo "Disabled")"
echo "   Results Location: ../../results/LearningToCountEverything-*-custom/pairtally_dataset/"
echo ""
echo " Files created:"
echo "   - custom_inference_data.json"
echo "   - custom_inference_data.pkl"
echo "   - {model_name}_custom_quantitative_results.json"
echo "   - {model_name}_custom_summary.txt"

# Visualization (optional)
CREATE_VIS="n"
if [[ "$CREATE_VIS" =~ ^[Yy]$ ]]; then
    results_dir="../../results/LearningToCountEverything-qualitative-custom/pairtally_dataset"
    if [ -d "$results_dir" ]; then
        python visualize_ltce_custom_results.py \
            --results_dir "$results_dir" \
            --sample_size 5 2>/dev/null || echo "  Warning: Visualization script not found or failed for pairtally_dataset"
    fi
    echo " Visualizations created for pairtally_dataset!"
    echo "   Location: ../../results/LearningToCountEverything-qualitative-custom/pairtally_dataset/visualizations/"
fi

echo "🏁 All custom evaluations complete!"
