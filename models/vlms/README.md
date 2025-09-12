# Vision-Language Models (VLMs) Evaluation

This directory contains evaluation scripts for 4 state-of-the-art Vision-Language Models on the PairTally benchmark dataset.

## Models Evaluated

### 1. Qwen2.5-VL-7B-Instruct
- **Developer**: Alibaba Cloud
- **Model Size**: 7B parameters
- **HuggingFace**: `Qwen/Qwen2.5-VL-7B-Instruct`
- **Architecture**: Enhanced vision-language model with improved perception

### 2. InternVL3-8B  
- **Developer**: OpenGVLab
- **Model Size**: 8B parameters
- **HuggingFace**: `OpenGVLab/InternVL2-8B`
- **Architecture**: Advanced multimodal model for visual-linguistic tasks

### 3. LLaMA-3.2-11B-Vision-Instruct
- **Developer**: Meta
- **Model Size**: 11B parameters
- **HuggingFace**: `meta-llama/Llama-3.2-11B-Vision-Instruct` 
- **Architecture**: Vision-instruct model with multimodal capabilities

### 4. Ovis2-Llama3-8B
- **Developer**: AIDC-AI
- **Model Size**: 8B parameters (3B variant available)
- **HuggingFace**: `AIDC-AI/Ovis1.6-Llama3.2-3B`
- **Architecture**: Structural embedding alignment for multimodal LLMs

## Evaluation Files

### Python Evaluation Scripts
Each model has two evaluation modes:

#### Both Classes Mode (Main Paper Results)
- `evaluate_pairtally_qwen2_5vl_count_both_classes.py`
- `evaluate_pairtally_internvl3_count_both_classes.py`
- `evaluate_pairtally_llama_vision_count_both_classes.py`
- `evaluate_pairtally_ovis2_count_both_classes.py`

#### One Class Mode (Additional Analysis)
- `evaluate_pairtally_qwen2_5vl_count_one_class.py`
- `evaluate_pairtally_internvl3_count_one_class.py`
- `evaluate_pairtally_llama_vision_count_one_class.py`
- `evaluate_pairtally_ovis2_count_one_class.py`

### Configuration Files
- `requirements.txt` - Common dependencies for all VLMs
- `SETUP.md` - Detailed setup instructions
- `README.md` - This documentation file

## Quick Start

### 1. Install Dependencies
```bash
# Install common VLM dependencies
pip install -r requirements.txt

# Additional model-specific packages
pip install qwen-vl-utils          # For Qwen2.5-VL
pip install sentencepiece         # For LLaMA models
```

### 2. Authentication (if required)
```bash
# Login to HuggingFace for gated models (LLaMA)
huggingface-cli login
```

### 3. Run Individual Model Evaluation
```bash
# Example: Qwen2.5-VL evaluation
python evaluate_pairtally_qwen2_5vl_count_both_classes.py
python evaluate_pairtally_qwen2_5vl_count_one_class.py

# Example: LLaMA-3.2 Vision evaluation  
python evaluate_pairtally_llama_vision_count_both_classes.py
python evaluate_pairtally_llama_vision_count_one_class.py
```

### 4. Run All VLM Evaluations
```bash
# From repository root
./scripts/evaluation/run_all_evaluations.sh
```

## Evaluation Protocol

### Input Format
VLMs receive natural language prompts without visual exemplars:
- **Both classes**: "Count the {class1} and {class2} in this image"
- **One class**: "How many {object_class} are there in this image?"

### Generation Parameters
- **Temperature**: 0.1 (for consistent numerical responses)
- **Max tokens**: 50 (sufficient for numerical answers)
- **Decoding**: Greedy decoding for deterministic results

### Response Processing
- Automatic extraction of numerical answers from text responses
- Handles various formats: "5", "five", "I see 5 objects", etc.
- Robust parsing with fallback strategies

## Expected Performance

Based on PairTally benchmark results:

| Model | MAE | RMSE | Performance Notes |
|-------|-----|------|------------------|
| **Ovis2** | 111.56 | 174.16 | Structural embedding approach |
| **Qwen2.5-VL** | 99.88 | 174.93 | Enhanced vision-language model |
| **LLaMA-3.2** | 97.56 | 175.80 | Best VLM performance |
| **InternVL3** | 115.98 | 179.89 | Advanced multimodal model |

**Key Finding**: All VLMs show significantly higher error rates compared to exemplar-based models, highlighting challenges in fine-grained counting tasks.

## Evaluation Modes

### 1. Count Both Classes Mode
- **Purpose**: Main benchmark evaluation
- **Task**: Count both object classes simultaneously  
- **Prompt**: Natural language describing both target objects
- **Results**: Primary paper metrics (MAE, RMSE)

### 2. Count One Class Mode
- **Purpose**: Additional analysis
- **Task**: Count one object class at a time
- **Prompt**: Single-class counting instructions
- **Results**: Per-class performance insights

## Output Structure

Results are saved to `outputs/` directory:
```
outputs/
├── {model_name}_both_classes_results.json    # Main benchmark results
├── {model_name}_one_class_results.json       # Per-class analysis
├── {model_name}_summary.txt                  # Performance summary
└── logs/
    ├── {model_name}_both_classes.log
    └── {model_name}_one_class.log
```

## System Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (for smaller models)
- **RAM**: 16GB system memory
- **Storage**: 50GB free space for model weights
- **Python**: 3.8+ with PyTorch 2.0+

### Recommended Requirements
- **GPU**: 16GB+ VRAM (RTX 4090, A100)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ SSD for optimal performance

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size or use model quantization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**Model Download Errors**
```bash
# Check internet connection and HuggingFace access
huggingface-cli whoami
```

**Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt --upgrade
```

**Authentication Errors**
```bash
# Re-login to HuggingFace
huggingface-cli logout
huggingface-cli login
```

### Performance Optimization

1. **Memory Management**
   - Use `torch.cuda.empty_cache()` between evaluations
   - Enable gradient checkpointing for larger models
   - Consider using quantized models (4-bit/8-bit)

2. **Speed Optimization**
   - Use appropriate batch sizes for your hardware
   - Enable mixed precision training if supported
   - Consider using Flash Attention for supported models

## Research Insights

### Key Findings from PairTally Evaluation

1. **VLM Limitations**: All VLMs show poor performance on fine-grained counting
2. **Semantic Understanding**: Models struggle with distinguishing similar object variants
3. **Prompt Dependency**: Performance varies significantly with prompt formulation
4. **Scale Effects**: Larger models don't necessarily perform better on counting tasks

### Failure Mode Analysis

**Common VLM Errors**:
- Over-counting objects in cluttered scenes
- Confusion between similar object categories
- Inconsistent numerical reasoning
- Hallucination of objects not present in images

## Citation

If you use these VLM evaluations, please cite:

```bibtex
@inproceedings{nguyen2025pairtally,
  title={Can Current AI Models Count What We Mean, Not What They See? 
         A Benchmark and Systematic Evaluation},
  author={Nguyen, Gia Khanh and Huang, Yifeng and Hoai, Minh},
  booktitle={Digital Image Computing: Techniques and Applications (DICTA)},
  year={2025}
}
```

### Model Citations

**Qwen2.5-VL**:
```bibtex
@article{qwen2vl2024,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and others},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}
```

**InternVL3**:
```bibtex
@article{chen2023internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and others},
  journal={arXiv preprint arXiv:2312.14238},
  year={2023}
}
```

**LLaMA-3.2 Vision**:
```bibtex
@article{llama3_2024,
  title={The Llama 3 Herd of Models},
  author={Llama Team},
  journal={arXiv preprint},
  year={2024}
}
```

**Ovis2**:
```bibtex
@article{ovis2024,
  title={Ovis: Structural Embedding Alignment for Multimodal Large Language Model},
  author={[Ovis Authors]},
  journal={arXiv preprint},
  year={2024}
}
```