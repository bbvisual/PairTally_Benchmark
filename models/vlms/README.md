# Vision-Language Models (VLMs) Evaluation

This directory contains evaluation scripts for 4 state-of-the-art Vision-Language Models on the DICTA25 dataset.

## Models Evaluated

### 1. Qwen2.5-VL-7B-Instruct
- **Developer**: Alibaba Cloud
- **Model**: Qwen2.5-VL-7B-Instruct
- **HuggingFace**: `Qwen/Qwen2.5-VL-7B-Instruct`

### 2. InternVL2.5-8B
- **Developer**: OpenGVLab
- **Model**: InternVL2.5-8B
- **HuggingFace**: `OpenGVLab/InternVL2-8B`

### 3. Llama-3.2-11B-Vision-Instruct
- **Developer**: Meta
- **Model**: Llama-3.2-11B-Vision-Instruct
- **HuggingFace**: `meta-llama/Llama-3.2-11B-Vision-Instruct`

### 4. Ovis2-Llama3-8B
- **Developer**: AIDC-AI
- **Model**: Ovis2-Llama3-8B
- **HuggingFace**: `AIDC-AI/Ovis1.6-Llama3.2-3B`

## Setup Instructions

### Environment Setup

Each VLM requires its own conda environment due to different dependency requirements:

#### 1. Qwen2.5-VL Environment
```bash
conda create -n qwen2_5vl python=3.9
conda activate qwen2_5vl
pip install -r requirements/qwen2_5vl_requirements.txt
```

#### 2. InternVL3 Environment  
```bash
conda create -n llama-vision python=3.9
conda activate llama-vision
pip install -r requirements/internvl_requirements.txt
```

#### 3. Llama Vision Environment
```bash
# Uses same environment as InternVL3
conda activate llama-vision
```

#### 4. Ovis2 Environment
```bash
conda create -n ovis2-34b python=3.9
conda activate ovis2-34b
pip install -r requirements/ovis2_requirements.txt
```

### Authentication Setup

Some models require HuggingFace authentication:

```bash
# Login to HuggingFace (required for Llama models)
huggingface-cli login
```

## Files in this Directory

### Evaluation Scripts
- `evaluate_DICTA25_qwen2_5vl_combined.py` - Qwen2.5-VL evaluation
- `evaluate_DICTA25_internvl3_combined.py` - InternVL3 evaluation  
- `evaluate_DICTA25_llama_vision_combined.py` - Llama Vision evaluation
- `evaluate_DICTA25_ovis2_combined.py` - Ovis2 evaluation

### Setup Scripts
- `setup_qwen2_5vl_environment.sh` - Qwen2.5-VL environment setup
- `run_qwen2_5vl_evaluation.sh` - Run Qwen2.5-VL evaluation

### Utility Scripts
- `monitor_and_kill_if_low_mem.sh` - Memory monitoring utility
- `test_qwen_vision.py` - Test script for Qwen models
- `test_llama_internvl_vision.py` - Test script for Llama/InternVL models

### Requirements
- `requirements.txt` - Common requirements
- Individual model requirements in `/requirements/` directory

## Running Evaluations

### Individual Model Evaluation

#### Qwen2.5-VL
```bash
conda activate qwen2_5vl
python evaluate_DICTA25_qwen2_5vl_combined.py \
    --base_data_path /path/to/DICTA25-Can-AI-Models-Count-Release/dataset \
    --dataset_name pairtally_dataset \
    --save_results
```

#### InternVL3
```bash
conda activate llama-vision
python evaluate_DICTA25_internvl3_combined.py \
    --base_data_path /path/to/DICTA25-Can-AI-Models-Count-Release/dataset \
    --dataset_name pairtally_dataset \
    --save_results
```

#### Llama Vision
```bash
conda activate llama-vision
python evaluate_DICTA25_llama_vision_combined.py \
    --base_data_path /path/to/DICTA25-Can-AI-Models-Count-Release/dataset \
    --dataset_name pairtally_dataset \
    --save_results
```

#### Ovis2
```bash
conda activate ovis2-34b
python evaluate_DICTA25_ovis2_combined.py \
    --base_data_path /path/to/DICTA25-Can-AI-Models-Count-Release/dataset \
    --dataset_name pairtally_dataset \
    --save_results
```

### Batch Evaluation

Run all VLM evaluations sequentially:
```bash
# Copy the batch script from main evaluation directory
cp ../../scripts/evaluation/run_all_vlm_evaluations.sh .
./run_all_vlm_evaluations.sh
```

## Evaluation Parameters

### Common Parameters
- **Temperature**: 0.1 (low for consistent counting)
- **Max tokens**: 20 (sufficient for numerical responses)
- **Prompt format**: "How many {object_class} are there in this image?"
- **Response parsing**: Numerical extraction from text responses

### Model-Specific Parameters

#### Qwen2.5-VL
- **Model ID**: `Qwen/Qwen2.5-VL-7B-Instruct`
- **Processor**: Qwen2VLProcessor
- **Generation config**: Temperature=0.1, max_new_tokens=20

#### InternVL3
- **Model ID**: `OpenGVLab/InternVL2-8B`
- **Image processing**: Custom InternVL processor
- **Generation**: Greedy decoding with low temperature

#### Llama Vision
- **Model ID**: `meta-llama/Llama-3.2-11B-Vision-Instruct`
- **Processor**: MllamaProcessor
- **Authentication**: Requires HuggingFace access token

#### Ovis2
- **Model ID**: `AIDC-AI/Ovis1.6-Llama3.2-3B`
- **Processor**: Custom Ovis processor
- **Memory**: Requires careful memory management

## Output Structure

Results for each VLM are saved to:
```
/path/to/results/{Model}-DICTA25-Results/
├── {Model}-quantitative/          # Quantitative metrics
│   └── pairtally_dataset/
│       ├── {Model}_quantitative_results.json
│       ├── {Model}_quantitative_results.pkl
│       └── {Model}_summary.txt
└── {Model}-qualitative/           # Qualitative results
    └── pairtally_dataset/
        ├── positive_qualitative_data.json
        ├── negative_qualitative_data.json
        ├── complete_qualitative_data.json
        └── complete_qualitative_data.pkl
```

## Expected Performance

VLM performance comparison on DICTA25:

| Model | Overall MAE | Overall RMSE | INTER MAE | INTRA MAE |
|-------|-------------|--------------|-----------|-----------|
| Qwen2.5-VL | X.XX | X.XX | X.XX | X.XX |
| InternVL3 | X.XX | X.XX | X.XX | X.XX |
| Llama Vision | X.XX | X.XX | X.XX | X.XX |
| Ovis2 | X.XX | X.XX | X.XX | X.XX |

## Key Evaluation Aspects

### 1. Text-Only Counting
- VLMs receive only text prompts describing objects to count
- No visual exemplars provided (unlike object counting models)
- Tests pure semantic understanding

### 2. Response Parsing
- Robust parsing of numerical answers from text responses
- Handles various response formats: "5", "five", "I count 5 objects"
- Logs parsing decisions for analysis

### 3. Error Analysis
- Systematic analysis of failure modes
- Comparison of INTER vs INTRA performance
- Category-specific performance analysis

### 4. Prompt Engineering
- Simple, consistent prompts across all models
- Format: "How many {object_class} are there in this image?"
- No few-shot examples or complex prompt engineering

## Troubleshooting

### Common Issues

1. **Memory Errors**: 
   - Use memory monitoring script: `./monitor_and_kill_if_low_mem.sh`
   - Reduce batch size or enable gradient checkpointing

2. **Authentication Errors**:
   - Ensure HuggingFace login: `huggingface-cli login`
   - Verify access to gated models (Llama)

3. **Environment Conflicts**:
   - Use separate conda environments for each model
   - Check CUDA compatibility between models

4. **Model Loading Errors**:
   - Verify model IDs are correct
   - Check internet connection for model downloads
   - Ensure sufficient disk space for model weights

### Performance Tips

1. **Memory Optimization**:
   - Use `torch.cuda.empty_cache()` between evaluations
   - Enable mixed precision if supported
   - Monitor GPU memory usage

2. **Speed Optimization**:
   - Use appropriate batch sizes for your hardware
   - Enable model compilation if available
   - Consider using quantized models for faster inference

## Citation

If you use these VLM evaluations in your research, please cite the original model papers and our benchmark:

```bibtex
# Our paper
@inproceedings{your-name2025dicta25,
  title={Can Current AI Models Count What We Mean, Not What They See? A Benchmark and Systematic Evaluation},
  author={Your Name and Co-authors},
  booktitle={Proceedings of DICTA},
  year={2025}
}

# Qwen2.5-VL
@article{qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and others},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}

# InternVL
@article{chen2023internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and others},
  journal={arXiv preprint arXiv:2312.14238},
  year={2023}
}

# Llama Vision
@article{llama3_2,
  title={The Llama 3 Herd of Models},
  author={Llama Team},
  journal={arXiv preprint},
  year={2024}
}
```