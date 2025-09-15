#!/bin/bash

# CountGD Count Both Classes Text-Only Evaluation Script for PairTally
# This script runs combined evaluation with text-only prompts (no exemplar boxes)
# Text-only CountGD model (combined prompts only)

echo "Starting CountGD Count Both Classes Text-Only Evaluation..."

# ===== CONFIGURATION SECTION =====
# Change BASE_DATA_PATH here and everything else will update automatically
BASE_DATA_PATH="../../dataset/pairtally_dataset"

# Automatically derive dataset name from the base path
DATASET_NAME=$(basename "$BASE_DATA_PATH")

# Other settings
MODEL_PATH="CountGD/checkpoints"
MODEL_NAME="checkpoint_fsc147_best"
CONFIG_FILE="CountGD/config/cfg_fsc147_vit_b.py"
CONFIDENCE_THRESH="0.3"
OUTPUT_LIMIT=""

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
        --config_file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --base_data_path)
            BASE_DATA_PATH="$2"
            # Automatically update dataset name when base path changes
            DATASET_NAME=$(basename "$BASE_DATA_PATH")
            shift 2
            ;;
        --confidence_thresh)
            CONFIDENCE_THRESH="$2"
            shift 2
            ;;
        --output_limit)
            OUTPUT_LIMIT="--output_limit $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_name NAME     Model name (default: checkpoint_fsc147_best)"
            echo "  --model_path PATH     Model path (default: CountGD/checkpoints)"
            echo "  --config_file FILE    Config file (default: CountGD/config/cfg_fsc147_vit_b.py)"
            echo "  --base_data_path P    Base path for dataset (default: ../../dataset/pairtally_dataset)"
            echo "  --confidence_thresh T Confidence threshold (default: 0.3)"
            echo "  --output_limit N      Limit processing to N images (for testing)"
            echo "  --help               Show this help message"
            echo ""
            echo "Example: $0 --output_limit 10"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Activate the countgd conda environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate countgd

# Set CUDA device (change as needed)
export CUDA_VISIBLE_DEVICES=0

echo "Model path: $MODEL_PATH"
echo "Model name: $MODEL_NAME"
echo "Config file: $CONFIG_FILE"
echo "Base data path: $BASE_DATA_PATH"
echo "Dataset to evaluate: $DATASET_NAME"
echo "Confidence threshold: $CONFIDENCE_THRESH"
echo "Mode: Combined Text-Only (combined prompts without exemplar boxes)"

# Check if model exists
if [ ! -f "$MODEL_PATH/$MODEL_NAME.pth" ]; then
    echo "Error: Model file $MODEL_PATH/$MODEL_NAME.pth does not exist!"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE does not exist!"
    exit 1
fi

echo "Model file found: $MODEL_PATH/$MODEL_NAME.pth"
echo "Config file found: $CONFIG_FILE"
echo ""

# Define paths for final dataset
DATA_PATH="$BASE_DATA_PATH"
ANNOTATION_FILE="$DATA_PATH/annotations/pairtally_annotations_simple.json"
IMAGE_FOLDER="$DATA_PATH/images"

# Check if data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo "Warning: Data path $DATA_PATH does not exist!"
    echo "Expected structure:"
    echo "$DATA_PATH/"
    echo "├── images/"
    echo "└── annotations/"
    echo "    └── pairtally_annotations_simple.json"
    echo "Skipping $DATASET_NAME..."
    echo ""
    exit 1
fi

# Check if annotation file exists
if [ ! -f "$ANNOTATION_FILE" ]; then
    echo "Warning: Annotation file $ANNOTATION_FILE does not exist!"
    echo "Skipping $DATASET_NAME..."
    echo ""
    exit 1
fi

# Check if image directory exists
if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "Warning: Image directory $IMAGE_FOLDER does not exist!"
    echo "Skipping $DATASET_NAME..."
    echo ""
    exit 1
fi

echo "Annotation file found: $ANNOTATION_FILE"
echo "Image directory found: $IMAGE_FOLDER"

# Create output directory
OUTPUT_DIR="./outputs"
mkdir -p "$OUTPUT_DIR"

echo "========================================" 
echo "Running CountGD Count Both Classes Text-Only Evaluation"
echo "========================================"
echo "Model: CountGD Combined Text-Only (combined prompts without exemplar boxes)"
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $MODEL_PATH/$MODEL_NAME.pth"
echo "Confidence threshold: $CONFIDENCE_THRESH"
echo "========================================"

# Run the combined text-only evaluation
python evaluate_pairtally_count_both_classes_text_only.py \
    --annotation_file "$ANNOTATION_FILE" \
    --image_folder "$IMAGE_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --config "$CONFIG_FILE" \
    --pretrain_model_path "$MODEL_PATH/$MODEL_NAME.pth" \
    --confidence_thresh "$CONFIDENCE_THRESH" \
    --save_path "$OUTPUT_DIR" \
    --device cuda \
    --eval \
    $OUTPUT_LIMIT

if [ $? -eq 0 ]; then
    echo "CountGD combined text-only evaluation completed successfully for $DATASET_NAME!"
    echo "Results saved to:"
    echo "  Predictions: $OUTPUT_DIR/${DATASET_NAME}_combined_text_only_predictions/"
    echo "  Quantitative: $OUTPUT_DIR/${DATASET_NAME}_combined_text_only_quantitative/"
    echo "  Visualizations: $OUTPUT_DIR/${DATASET_NAME}_combined_text_only_visualizations/"
else
    echo "Evaluation failed for $DATASET_NAME with exit code $?"
    exit 1
fi

echo "Evaluation completed for $DATASET_NAME!"
echo "========================================"
echo "All evaluations completed!"
echo "========================================"

echo "CountGD Combined Text-Only evaluation complete!"