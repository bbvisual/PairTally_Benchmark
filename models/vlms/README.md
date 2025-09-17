# Vision-Language Models (VLMs) for Object Counting

This directory contains the evaluation setup for 4 Vision-Language Models on the PairTally dataset.

## Supported Models

1. **Ovis2** - Structural Embedding Alignment for Multimodal LLM (16B)
2. **Qwen2.5-VL** - Enhanced Vision-Language Model (7B)  
3. **LLaMA-3.2** - Vision-Instruct Model (11B)
4. **InternVL3** - Advanced Multimodal Model (14B)

## Setup Instructions

### 1. Environment Setup
```bash
# Create conda environment
conda create -n vlms python=3.9
conda activate vlms

# Install core dependencies
pip install -r requirements.txt

# Or install manually:
pip install torch>=2.0.0 torchvision>=0.15.0
pip install transformers>=4.45.0 accelerate>=0.21.0
pip install pillow numpy matplotlib scipy
pip install flash-attn sentencepiece timm
```

### 2. Hugging Face Setup
Some models may require Hugging Face authentication:

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login (if required for model access)
huggingface-cli login
```

### 3. GPU Requirements
VLMs require significant GPU memory:
- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB+ GPU memory
- **For large models**: 24GB+ GPU memory

## PairTally Evaluation

### Files in this Directory

**Evaluation Scripts:**

**Ovis2 (16B):**
- `evaluate_pairtally_ovis2_count_both_classes.py` - Combined evaluation
- `evaluate_pairtally_ovis2_count_one_class.py` - Single-class evaluation

**Qwen2.5-VL (7B):**
- `evaluate_pairtally_qwen2_5vl_count_both_classes.py` - Combined evaluation
- `evaluate_pairtally_qwen2_5vl_count_one_class.py` - Single-class evaluation

**LLaMA-3.2 Vision (11B):**
- `evaluate_pairtally_llama_vision_count_both_classes.py` - Combined evaluation
- `evaluate_pairtally_llama_vision_count_one_class.py` - Single-class evaluation

**InternVL3 (14B):**
- `evaluate_pairtally_internvl3_count_both_classes.py` - Combined evaluation
- `evaluate_pairtally_internvl3_count_one_class.py` - Single-class evaluation

**Additional Files:**
- `requirements.txt` - Python dependencies
- `SETUP.md` - Detailed setup instructions
- `README.md` - This file

### Running Evaluation

VLMs are accessed through Hugging Face and don't require separate model downloads. The scripts automatically use the dataset at `../../dataset/pairtally_dataset/`.

**Dataset Structure Expected:**
```
../../dataset/pairtally_dataset/
├── annotations/
│   └── pairtally_annotations_simple.json
└── images/
    └── [image files]
```

**Run Evaluation:**

**Option 1: Combined Mode (Both classes simultaneously)**
```bash
python evaluate_pairtally_qwen2_5vl_count_both_classes.py
python evaluate_pairtally_ovis2_count_both_classes.py
python evaluate_pairtally_llama_vision_count_both_classes.py
python evaluate_pairtally_internvl3_count_both_classes.py
```

**Option 2: Single-Class Mode (One class at a time)**
```bash
python evaluate_pairtally_qwen2_5vl_count_one_class.py
python evaluate_pairtally_ovis2_count_one_class.py
python evaluate_pairtally_llama_vision_count_one_class.py
python evaluate_pairtally_internvl3_count_one_class.py
```

**For Testing (limit to N images, add to script):**
```python
# Add --output_limit parameter to the script if supported
```

### Evaluation Modes

**Combined Mode** (`*_count_both_classes.py`):
- Asks model to count **both object classes** in the image using natural language
- Uses text-only prompts describing both object types
- Tests ability to distinguish between different object types through language understanding
- More challenging as model must identify and count multiple object types simultaneously

**Single-Class Mode** (`*_count_one_class.py`):
- Asks model to count **one class at a time** using natural language
- Uses separate text prompts for each object class
- Simpler task focusing on counting accuracy for a single object type
- Two separate runs per image (one for each object class)

### Evaluation Parameters

The evaluation uses the following key parameters:
- **Models**: Hugging Face transformers (auto-downloaded)
- **Input**: Natural language prompts + images
- **No visual exemplars**: VLMs rely purely on language descriptions
- **Device**: CUDA (configurable via `CUDA_VISIBLE_DEVICES`)
- **Inference**: Text generation with structured output parsing

### Output Structure

Results are saved to `../../results/` with the following structure:

**Single-Class Mode:**
```
../../results/{ModelName}-quantitative/pairtally_dataset/
├── {ModelName}_quantitative_results.json
├── {ModelName}_quantitative_results.pkl
└── {ModelName}_summary.txt

../../results/{ModelName}-qualitative/pairtally_dataset/
├── positive_qualitative_data.json
├── negative_qualitative_data.json
└── complete_qualitative_data.json
```

**Combined Mode:**
```
../../results/{ModelName}-quantitative-combined/pairtally_dataset/
├── {ModelName}_combined_quantitative_results.json
├── {ModelName}_combined_quantitative_results.pkl
└── {ModelName}_combined_summary.txt

../../results/{ModelName}-qualitative-combined/pairtally_dataset/
└── {ModelName}_combined_detailed_results.json
```

Where `{ModelName}` is one of: `Ovis2`, `Qwen2_5VL`, `LLaMA_Vision`, `InternVL3`

### Model-Specific Notes

**Qwen2.5-VL:**
- Efficient 7B parameter model
- Good balance of performance and memory usage
- May require `qwen-vl-utils` package

**LLaMA-3.2 Vision:**
- 11B parameter vision-instruct model
- May require Meta AI access permissions
- Uses `sentencepiece` for tokenization

**Ovis2:**
- Large 16B parameter model
- Structural embedding alignment approach
- Requires latest transformers version

**InternVL3:**
- Advanced 14B parameter multimodal model
- Large model requiring significant GPU memory
- Consider using quantization for smaller GPUs

**Performance Tips:**
- Use `CUDA_VISIBLE_DEVICES` to specify GPU
- Monitor GPU memory usage during evaluation
- Consider using gradient checkpointing for memory efficiency
- Results are automatically saved to `../../results/`
- VLM models are cached in `~/.cache/huggingface/`

### Key Differences from Other Models

Unlike traditional counting models, VLMs:
1. **No visual exemplars**: Rely purely on language descriptions
2. **Natural language interface**: Use text prompts instead of bounding boxes
3. **Generative approach**: Generate text responses that need parsing
4. **Large model size**: Require significant computational resources
5. **Zero-shot capability**: Can count objects without training on counting datasets

### Citations

If you use any of these VLMs in your research, please cite the respective papers:

**Qwen2.5-VL:**
```bibtex
@article{qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and others},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}
```

**LLaMA-3.2:**
```bibtex
@misc{llama32,
  title={Llama 3.2: Revolutionizing edge AI and vision with open, customizable models},
  author={Meta AI},
  year={2024}
}
```

**InternVL3:**
```bibtex
@article{internvl3,
  title={InternVL 2.0: Scaling Vision-Language Models to 34B Parameters},
  author={Chen, Zhe and others},
  journal={arXiv preprint},
  year={2024}
}
```

**Ovis2:**
```bibtex
@article{lu2024ovis,
  title={Ovis: Structural embedding alignment for multimodal large language model},
  author={Lu, Shiyin and Li, Yang and Chen, Qing-Guo and Xu, Zhao and Luo, Weihua and Zhang, Kaifu and Ye, Han-Jia},
  journal={arXiv:2405.20797},
  year={2024}
}
```