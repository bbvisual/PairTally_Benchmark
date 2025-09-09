#!/bin/bash

# CountGD Text-Only Custom Qualitative Evaluation Script for PairTally
# This script runs custom evaluation with text-only prompts on the final dataset only
# Text-only CountGD model (without exemplar boxes)

echo "🚀 Starting CountGD Text-Only Custom Qualitative Evaluation on FINAL DATASET..."

# ===== CONFIGURATION SECTION =====
# Change BASE_DATA_PATH here and everything else will update automatically
BASE_DATA_PATH="./pairtally_dataset"

# Automatically derive dataset name from the base path
DATASET_NAME=$(basename "$BASE_DATA_PATH")

# Other settings
MODEL_PATH="./CountGD/checkpoints"
MODEL_NAME="checkpoint_fsc147_best"
CONFIG_FILE="./CountGD/config/cfg_fsc147_vit_b.py"
CONFIDENCE_THRESH="0.23"
OUTPUT_LIMIT=""

# Automatically set dataset array
DATASETS=("$DATASET_NAME")

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
            DATASETS=("$DATASET_NAME")
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
            echo "  --model_path PATH     Model path (default: ./CountGD/checkpoints)"
            echo "  --config_file FILE    Config file (default: ./CountGD/config/cfg_fsc147_vit_b.py)"
            echo "  --base_data_path P    Base path for dataset (default: ./final_dataset)"
            echo "  --confidence_thresh T Confidence threshold (default: 0.23)"
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

echo "📂 Model path: $MODEL_PATH"
echo "🎯 Model name: $MODEL_NAME"
echo "📁 Config file: $CONFIG_FILE"
echo "📊 Base data path: $BASE_DATA_PATH"
echo "🎯 Dataset to evaluate: $DATASET_NAME"
echo "🎚️ Confidence threshold: $CONFIDENCE_THRESH"
echo "📝 Mode: Text-Only (no exemplar boxes)"

# Check if model exists
if [ ! -f "$MODEL_PATH/$MODEL_NAME.pth" ]; then
    echo "❌ Error: Model file $MODEL_PATH/$MODEL_NAME.pth does not exist!"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file $CONFIG_FILE does not exist!"
    exit 1
fi

echo "✅ Model file found: $MODEL_PATH/$MODEL_NAME.pth"
echo "✅ Config file found: $CONFIG_FILE"
echo ""

# Define paths for final dataset
DATA_PATH="$BASE_DATA_PATH"
ANNOTATION_FILE="$DATA_PATH/annotations/pairtally_annotations_simple.json"
IMAGE_DIR="$DATA_PATH/images"

# Check if data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ Warning: Data path $DATA_PATH does not exist!"
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
    echo "❌ Warning: Annotation file $ANNOTATION_FILE does not exist!"
    echo "Skipping $DATASET_NAME..."
    echo ""
    exit 1
fi

# Check if image directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ Warning: Image directory $IMAGE_DIR does not exist!"
    echo "Skipping $DATASET_NAME..."
    echo ""
    exit 1
fi

echo "✅ Annotation file found: $ANNOTATION_FILE"
echo "✅ Image directory found: $IMAGE_DIR"

# Create results directories for this dataset
mkdir -p "../../results/CountGD-TextOnly-qualitative/$DATASET_NAME"
mkdir -p "../../results/CountGD-TextOnly-quantitative/$DATASET_NAME"

# Run custom evaluation with parameters for CountGD text-only model
python evaluate_PairTally_text_only.py \
    --annotations_file "$ANNOTATION_FILE" \
    --images_folder "$IMAGE_DIR" \
    --config "$CONFIG_FILE" \
    --pretrain_model_path "$MODEL_PATH/$MODEL_NAME.pth" \
    --confidence_thresh "$CONFIDENCE_THRESH" \
    --device "cuda" \
    --dataset_name "$DATASET_NAME" \
    $OUTPUT_LIMIT

if [ $? -eq 0 ]; then
    echo "🎉 CountGD text-only custom evaluation completed successfully for $DATASET_NAME!"
    echo "📁 Results saved to: ../../results/CountGD-TextOnly-qualitative/$DATASET_NAME/"
else
    echo "❌ Evaluation failed for $DATASET_NAME with exit code $?"
    exit 1
fi

echo "Evaluation completed for $DATASET_NAME!"
echo "========================================"
echo "All evaluations completed!"
echo "========================================"

# Visualization (optional)
CREATE_VIS="n"
if [[ "$CREATE_VIS" =~ ^[Yy]$ ]]; then
    results_dir="../../results/CountGD-TextOnly-qualitative/$DATASET_NAME"
    if [ -d "$results_dir" ]; then
        python visualize_countgd_results.py \
            --results_dir "$results_dir" \
            --sample_size 5 2>/dev/null || echo "  Warning: Visualization script not found or failed for $DATASET_NAME"
    fi
    echo "✅ Visualizations created for $DATASET_NAME!"
    echo "   Location: ../../results/CountGD-TextOnly-qualitative/$DATASET_NAME/visualizations/"
fi

echo "🎉 All evaluations complete!"
