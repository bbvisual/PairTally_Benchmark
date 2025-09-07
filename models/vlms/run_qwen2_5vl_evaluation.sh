#!/bin/bash

# Qwen2.5-VL DICTA25 Evaluation Script
# This script demonstrates how to run the Qwen2.5-VL evaluation on DICTA25 dataset

echo "Qwen2.5-VL DICTA25 Evaluation Runner"
echo "===================================="

# Set common parameters
BASE_DATA_PATH="/home/khanhnguyen/DICTA25"
DATASET_NAME="test_bbx_frames"
MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"

# Basic evaluation (limited output for testing)
echo "Running basic Qwen2.5-VL evaluation (limited to 3 images for testing)..."
python evaluate_DICTA25_qwen2_5vl.py \
    --model_id "$MODEL_ID" \
    --base_data_path "$BASE_DATA_PATH" \
    --dataset_name "$DATASET_NAME" \
    --output_limit 3 \
    --temperature 0.1 \
    --max_new_tokens 256

echo ""
echo "Basic evaluation completed!"
echo ""

# Evaluation with flash attention enabled
echo "Running Qwen2.5-VL evaluation with flash attention..."
python evaluate_DICTA25_qwen2_5vl.py \
    --model_id "$MODEL_ID" \
    --base_data_path "$BASE_DATA_PATH" \
    --dataset_name "$DATASET_NAME" \
    --output_limit 3 \
    --temperature 0.1 \
    --max_new_tokens 256 \
    --use_flash_attention

echo ""
echo "Flash attention evaluation completed!"
echo ""

# Full evaluation (uncomment to run on entire dataset)
# echo "Running full Qwen2.5-VL evaluation on entire dataset..."
# python evaluate_DICTA25_qwen2_5vl.py \
#     --model_id "$MODEL_ID" \
#     --base_data_path "$BASE_DATA_PATH" \
#     --dataset_name "$DATASET_NAME" \
#     --temperature 0.1 \
#     --max_new_tokens 256 \
#     --use_flash_attention

echo "All evaluations completed!"
echo ""
echo "Results can be found in:"
echo "  - Qualitative: /home/khanhnguyen/DICTA25-RESULTS/Qwen2_5VL-qualitative/"
echo "  - Quantitative: /home/khanhnguyen/DICTA25-RESULTS/Qwen2_5VL-quantitative/" 