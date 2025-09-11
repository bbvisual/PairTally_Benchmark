#!/bin/bash

# DAVE Combined Evaluation Script for DICTA25
# This script runs combined evaluation with 2 positive + 1 negative exemplars on the final dataset

echo " Starting DAVE Combined Evaluation (2 Positive + 1 Negative Exemplars) on FINAL DATASET..."

# Default settings
MODEL_PATH="DAVE/pretrained_models"
MODEL_NAME="DAVE_3_shot"
BASE_DATA_PATH="../../dataset/pairtally_dataset"
OUTPUT_LIMIT=""

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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_name NAME     Model name (default: DAVE_3_shot)"
            echo "  --model_path PATH     Model path (default: ./pretrained_models)"
            echo "  --base_data_path P    Base path for dataset (default: ../../../DICTA25/pairtally_dataset)"
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
echo " Dataset to evaluate: pairtally_dataset"
echo "🔗 Evaluation type: Combined (2 positive + 1 negative exemplars)"

# Check if model exists
if [ ! -f "$MODEL_PATH/$MODEL_NAME.pth" ]; then
    echo " Error: Model file $MODEL_PATH/$MODEL_NAME.pth does not exist!"
    exit 1
fi

echo " Model file found: $MODEL_PATH/$MODEL_NAME.pth"

# Check if verification file exists
if [ ! -f "$MODEL_PATH/verification.pth" ]; then
    echo "  Warning: Verification file $MODEL_PATH/verification.pth does not exist!"
    echo "   DAVE model requires both main and verification checkpoints"
else
    echo " Verification file found: $MODEL_PATH/verification.pth"
fi

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

# Create results directories for combined evaluation with clear identifiers
mkdir -p "../../results/DAVE-qualitative-combined/pairtally_dataset"
mkdir -p "../../results/DAVE-quantitative-combined/pairtally_dataset"

echo " Results will be saved to:"
echo "   Qualitative: ../../results/DAVE-qualitative-combined/pairtally_dataset/"
echo "   Quantitative: ../../results/DAVE-quantitative-combined/pairtally_dataset/"
echo ""

# Run combined evaluation with DAVE 3-shot parameters
echo "🏃 Running combined evaluation (2 positive + 1 negative exemplars)..."
python evaluate_pairtally_count_both_classes.py \
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
    echo " DAVE combined evaluation completed successfully for pairtally_dataset!"
    echo " Qualitative results saved to: ../../results/DAVE-qualitative-combined/pairtally_dataset/"
    echo " Quantitative results saved to: ../../results/DAVE-quantitative-combined/pairtally_dataset/"
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
echo "   Results Location: ../../results/DAVE-*-combined/pairtally_dataset/"
echo ""
echo " Files created:"
echo "   - combined_inference_data.json"
echo "   - combined_inference_data.pkl"
echo "   - {model_name}_combined_quantitative_results.json"
echo "   - {model_name}_combined_summary.txt"

# Visualization (optional)
CREATE_VIS="n"
if [[ "$CREATE_VIS" =~ ^[Yy]$ ]]; then
    results_dir="../../results/DAVE-qualitative-combined/pairtally_dataset"
    if [ -d "$results_dir" ]; then
        python visualize_dave_combined_results.py \
            --results_dir "$results_dir" \
            --sample_size 5 2>/dev/null || echo "  Warning: Visualization script not found or failed for pairtally_dataset"
    fi
    echo " Visualizations created for pairtally_dataset!"
    echo "   Location: ../../results/DAVE-qualitative-combined/pairtally_dataset/visualizations/"
fi

echo "🏁 All combined evaluations complete!"
