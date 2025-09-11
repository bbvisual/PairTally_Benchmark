#!/bin/bash

# DAVE Custom Qualitative Evaluation Script for DICTA25
# This script runs custom evaluation with positive/negative exemplar handling on the final dataset only
# Configured for DAVE 3-shot model

echo " Starting DAVE 3-Shot Custom Qualitative Evaluation (Positive/Negative Exemplars) on FINAL DATASET..."

# Default settings
MODEL_PATH="/home/khanhnguyen/DICTA25/DAVE/MODEL_folder"
MODEL_NAME="DAVE_3_shot"
BASE_DATA_PATH="/home/khanhnguyen/DICTA25/final_dataset"
OUTPUT_LIMIT=""

# Only run on the final dataset
DATASETS=("final_dataset")

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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_name NAME     Model name (default: DAVE_3_shot)"
            echo "  --model_path PATH     Model path (default: /home/khanhnguyen/DICTA25/DAVE/MODEL_folder)"
            echo "  --base_data_path P    Base path for dataset (default: /home/khanhnguyen/DICTA25/final_dataset)"
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

# Activate the dave conda environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate dave

# Set CUDA device (change as needed)
export CUDA_VISIBLE_DEVICES=0

echo " Model path: $MODEL_PATH"
echo " Model name: $MODEL_NAME"
echo " Base data path: $BASE_DATA_PATH"
echo " Dataset to evaluate: final_dataset"

# Check if model exists
if [ ! -f "$MODEL_PATH/$MODEL_NAME.pth" ]; then
    echo " Error: Model file $MODEL_PATH/$MODEL_NAME.pth does not exist!"
    exit 1
fi

echo " Model file found: $MODEL_PATH/$MODEL_NAME.pth"
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
    echo "Skipping final_dataset..."
    echo ""
    exit 1
fi

# Check if annotation file exists
if [ ! -f "$ANNOTATION_FILE" ]; then
    echo " Warning: Annotation file $ANNOTATION_FILE does not exist!"
    echo "Skipping final_dataset..."
    echo ""
    exit 1
fi

# Check if image directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo " Warning: Image directory $IMAGE_DIR does not exist!"
    echo "Skipping final_dataset..."
    echo ""
    exit 1
fi

echo " Annotation file found: $ANNOTATION_FILE"
echo " Image directory found: $IMAGE_DIR"

# Create results directories for this dataset
mkdir -p "/home/khanhnguyen/DICTA25-RESULTS/DAVE-qualitative/final_dataset"
mkdir -p "/home/khanhnguyen/DICTA25-RESULTS/DAVE-quantitative/final_dataset"

# Run custom evaluation with parameters for DAVE 3-shot model
python evaluate_DICTA25_custom.py \
    --annotation_file "$ANNOTATION_FILE" \
    --image_dir "$IMAGE_DIR" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --backbone resnet50 \
    --image_size 512 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --emb_dim 256 \
    --num_heads 8 \
    --kernel_dim 3 \
    --num_objects 3 \
    --reduction 8 \
    --dropout 0.1 \
    --pre_norm \
    --use_query_pos_emb \
    --use_objectness \
    --use_appearance \
    --d_s 1.0 \
    --m_s 0.0 \
    --i_thr 0.55 \
    --d_t 3.0 \
    --s_t 0.008 \
    --egv 0.132 \
    --prompt_shot \
    --batch_size 1 \
    --num_workers 0 \
    $OUTPUT_LIMIT

if [ $? -eq 0 ]; then
    echo " DAVE 3-shot custom evaluation completed successfully for final_dataset!"
    echo " Results saved to: /home/khanhnguyen/DICTA25-RESULTS/DAVE-qualitative/final_dataset/"
else
    echo " Evaluation failed for final_dataset with exit code $?"
    exit 1
fi

echo "Evaluation completed for final_dataset!"
echo "========================================"
echo "All evaluations completed!"
echo "========================================"

# Visualization (optional)
CREATE_VIS="n"
if [[ "$CREATE_VIS" =~ ^[Yy]$ ]]; then
    results_dir="/home/khanhnguyen/DICTA25-RESULTS/DAVE-qualitative/final_dataset"
    if [ -d "$results_dir" ]; then
        python visualize_dave_results.py \
            --results_dir "$results_dir" \
            --sample_size 5 2>/dev/null || echo "  Warning: Visualization script not found or failed for final_dataset"
    fi
    echo " Visualizations created for final_dataset!"
    echo "   Location: /home/khanhnguyen/DICTA25-RESULTS/DAVE-qualitative/final_dataset/visualizations/"
fi

echo " All evaluations complete!"
