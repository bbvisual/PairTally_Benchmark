# Vision-Language Models (VLMs) Setup Instructions

## Prerequisites

VLMs are typically accessed through Hugging Face transformers and don't require separate model downloads. However, you need to install the appropriate dependencies and may need Hugging Face authentication for some models.

## Supported Models

This directory contains evaluation scripts for 4 Vision-Language Models:

1. **Ovis2** - Structural Embedding Alignment for Multimodal LLM (16B)
2. **Qwen2.5-VL** - Enhanced Vision-Language Model (7B)  
3. **LLaMA-3.2** - Vision-Instruct Model (11B)
4. **InternVL3** - Advanced Multimodal Model (14B)

## 1. Setup Directory Structure

```
models/vlms/
├── evaluate_pairtally_ovis2_count_both_classes.py
├── evaluate_pairtally_ovis2_count_one_class.py
├── evaluate_pairtally_qwen2_5vl_count_both_classes.py
├── evaluate_pairtally_qwen2_5vl_count_one_class.py
├── evaluate_pairtally_llama_vision_count_both_classes.py
├── evaluate_pairtally_llama_vision_count_one_class.py
├── evaluate_pairtally_internvl3_count_both_classes.py
├── evaluate_pairtally_internvl3_count_one_class.py
└── SETUP.md                                     # This file
```

## 2. Install Dependencies

Install the required packages for VLM evaluation:

```bash
# Core dependencies
pip install torch torchvision transformers
pip install pillow opencv-python numpy
pip install accelerate bitsandbytes

# Model-specific dependencies
pip install qwen-vl-utils  # For Qwen2.5-VL
pip install sentencepiece  # For LLaMA models

# Optional: For better performance
pip install flash-attn --no-build-isolation
```

## 3. Hugging Face Setup (if needed)

Some models may require Hugging Face authentication:

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login (if required for model access)
huggingface-cli login
```

## 4. Run Evaluation

VLMs can be run directly without downloading separate model files:

```bash
# Evaluate specific models
python evaluate_pairtally_qwen2_5vl_count_both_classes.py
python evaluate_pairtally_ovis2_count_one_class.py
python evaluate_pairtally_llama_vision_count_both_classes.py
python evaluate_pairtally_internvl3_count_one_class.py
```

## Expected Model Performance

Based on the paper results:

| Model | MAE | RMSE |
|-------|-----|------|
| **Ovis2** | 111.56 | 174.16 |
| **Qwen2.5-VL** | 99.88 | 174.93 |
| **LLaMA-3.2** | 97.56 | 175.80 |
| **InternVL3** | 115.98 | 179.89 |

## Evaluation Modes

Each model has two evaluation scripts:

1. **Count Both Classes** (`*_count_both_classes.py`):
   - Asks model to count both object classes in the image
   - Uses natural language prompts
   - Main results reported in paper

2. **Count One Class** (`*_count_one_class.py`):
   - Asks model to count one class at a time
   - Two separate runs per image
   - Useful for per-class analysis

## GPU Requirements

VLMs require significant GPU memory:
- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB+ GPU memory
- **For large models**: 24GB+ GPU memory

## Troubleshooting

- **CUDA out of memory**: Reduce batch size or use model quantization
- **Model download errors**: Check internet connection and Hugging Face access
- **Import errors**: Install all required dependencies
- **Authentication errors**: Login to Hugging Face if required

## Model-Specific Notes

### Qwen2.5-VL
- Requires `qwen-vl-utils` package
- May need specific tokenizer setup

### LLaMA-3.2
- May require Meta AI access permissions
- Use `sentencepiece` for tokenization

### Ovis2
- Newer model, ensure latest transformers version
- May require specific configuration

### InternVL3
- Large model, requires significant GPU memory
- Consider using quantization for smaller GPUs

## Directory Structure After Setup

```
models/vlms/
├── outputs/                         # Created automatically
├── [evaluation scripts]             # Already present
└── [model cache]                    # Downloaded automatically to ~/.cache/huggingface/
```
