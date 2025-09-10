# PairTally: Can Current AI Models Count What We Mean, Not What They See?

<p align="center">
  <img src="img/teaser.png" alt="PairTally Dataset Examples" width="100%">
</p>

<p align="center">
  <strong>A Benchmark for Fine-Grained Visual Counting</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx">Paper</a> •
  <a href="#installation">Installation</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

Visual counting requires more than object detection—it demands understanding user intent and discriminating between visually similar objects. While humans effortlessly count specific items based on subtle attributes, current AI systems struggle with this fundamental task.

**PairTally** is the first benchmark specifically designed to evaluate fine-grained visual counting capabilities. Each image contains two object types requiring discrimination based on color, size, or texture/shape differences, revealing critical gaps in current computer vision models.

## Key Contributions

- **Systematic Benchmark**: 681 high-resolution images with controlled pairs of visually similar objects
- **Comprehensive Evaluation**: Assessment of 10 state-of-the-art models across three counting paradigms
- **Fine-grained Analysis**: Attribute-specific evaluation revealing model strengths and limitations
- **Performance Gap**: Quantitative evidence that best models (53.07 MAE) significantly underperform humans (~5 MAE)

## Dataset

### Statistics

| Property | Value |
|----------|-------|
| Total Images | 681 |
| Object Categories | 54 |
| Subcategories | 98 |
| Objects per Image | ~200 (average) |
| Image Resolution | High (varied) |

### Structure

```
PairTally/
├── images/                    # 681 high-resolution images
├── annotations/
│   ├── inter_category/        # 350 images - Different object types
│   └── intra_category/        # 331 images - Same object, different attributes
└── evaluation/                # Evaluation scripts and tools
```

### Attribute Distribution

| Attribute Type | Percentage | Examples |
|---------------|------------|----------|
| Color | 43.5% | Black vs. white game pieces |
| Shape/Texture | 42.5% | Spiral vs. penne pasta |
| Size | 14.0% | Large vs. small marbles |

## Installation

```bash
# Clone repository
git clone https://github.com/bbvisual/PairTally_Benchmark.git
cd PairTally_Benchmark

# Create environment
conda create -n pairtally python=3.8
conda activate pairtally

# Install dependencies
pip install -r requirements.txt

# Download dataset
python download_dataset.py

# Verify installation
python verify_dataset.py
```

## Quick Start

```python
from pairtally import PairTallyDataset

# Initialize dataset
dataset = PairTallyDataset(data_dir='./dataset')

# Load sample
image, annotation = dataset[0]

# Access annotation details
positive_class = annotation['positive_prompt']  # e.g., "red poker chips"
negative_class = annotation['negative_prompt']  # e.g., "blue poker chips"
positive_count = len(annotation['points'])
negative_count = len(annotation['negative_points'])
```

## Evaluation

### Running Evaluation

```python
from pairtally import evaluate_model

# Load your model
model = YourCountingModel()

# Evaluate on full dataset
results = evaluate_model(
    model=model,
    dataset=dataset,
    subset='all'  # Options: 'all', 'inter', 'intra'
)

# Access metrics
print(f"MAE: {results['mae']:.2f}")
print(f"RMSE: {results['rmse']:.2f}")
print(f"NAE: {results['nae']:.2f}")
```

### Supported Model Types

| Category | Models | Input Type |
|----------|--------|------------|
| Exemplar-based | FamNet, DAVE, GeCo, LoCA | Visual exemplars |
| Language-prompted | CountGD, LLMDet | Text descriptions |
| Vision-Language | Qwen-VL, LLaMA-Vision, InternVL | Multi-modal |

## Results

### Overall Performance

| Model | Type | MAE ↓ | RMSE ↓ | Parameters |
|-------|------|-------|--------|------------|
| **GeCo** | Exemplar | **53.07** | **98.00** | 126M |
| CountGD | Hybrid | 57.33 | 108.93 | 172M |
| LoCA | Exemplar | 62.78 | 136.76 | 95M |
| DAVE | Exemplar | 69.49 | 130.42 | 118M |
| FamNet | Exemplar | 88.30 | 148.42 | 82M |
| Mean Baseline | - | 98.56 | 151.45 | - |
| Qwen2.5-VL | VLM | 99.88 | 174.93 | 7B |
| Human Performance* | - | ~5 | ~8 | - |

*Estimated based on subset evaluation

### Attribute-Specific Performance (NAE)

| Model | Color | Size | Texture/Shape |
|-------|-------|------|---------------|
| DAVE | 0.738 | 1.293 | 0.693 |
| GeCo | 0.791 | 1.345 | 0.946 |
| LoCA | 0.799 | 1.244 | 1.007 |
| CountGD | 0.856 | 1.402 | 0.793 |

### Key Findings

1. **Performance Gap**: Best models achieve 10× higher error than estimated human performance
2. **Attribute Sensitivity**: Color variations are most distinguishable (NAE: 0.74-0.86), size differences most challenging (NAE: 1.24-1.40)
3. **Model Limitations**: Vision-Language Models significantly underperform specialized counting methods
4. **Generalization Issues**: Models often count all objects, ignoring specified attributes

## Demo Notebook

Explore the dataset and evaluation pipeline through our interactive notebook:

```bash
jupyter notebook PairTally_Demo_Notebook.ipynb
```

The notebook provides:
- Dataset visualization and statistics
- Sample annotations and bounding boxes
- Model evaluation pipeline
- Performance analysis tools

## Citation

```bibtex
@inproceedings{nguyen2025pairtally,
  title={Can Current AI Models Count What We Mean, Not What They See? 
         A Benchmark and Systematic Evaluation},
  author={Nguyen, Gia Khanh and Huang, Yifeng and Hoai, Minh},
  booktitle={Digital Image Computing: Techniques and Applications (DICTA)},
  year={2025}
}
```

## License

This dataset is released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license for academic research purposes.

## Contact

**Gia Khanh Nguyen**  
Australian Institute for Machine Learning  
University of Adelaide  
Email: giakhanh.nguyen@adelaide.edu.au

## Acknowledgments

We thank the Australian Institute for Machine Learning and Stony Brook University for supporting this research. Special thanks to all contributors who helped with data collection and annotation.

---

<p align="center">
  <a href="https://github.com/bbvisual/PairTally_Benchmark">GitHub</a> •
  <a href="https://arxiv.org/abs/xxxx.xxxxx">Paper</a> •
  <a href="https://drive.google.com/file/d/1TnenXS4yFicjo81NnmClfzgc8ltmmeBv/view">Dataset</a>
</p>