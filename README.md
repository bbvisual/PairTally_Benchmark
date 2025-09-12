# PairTally: A Benchmark Dataset

<p align="center">
  <img src="img/teaser.png" alt="PairTally Dataset Examples" width="100%">
</p>

<p align="center">
  <strong>Can Current AI Models Count What We Mean, Not What They See?</strong><br>
  <em>Gia Khanh Nguyen<sup>1</sup>, Yifeng Huang<sup>2</sup>, Minh Hoai<sup>1</sup></em><br>
  <sup>1</sup>Australian Institute for Machine Learning, University of Adelaide<br>
  <sup>2</sup>Stony Brook University
</p>

<p align="center">
  <a href="#citation">Paper</a> •
  <a href="#installation">Installation</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a>
</p>

---
## Key Contributions

- **First Fine-Grained Counting Benchmark**: Introduces the first dataset specifically designed to test subtle within-class distinctions, with 681 high-resolution, controlled pairs.
- **Comprehensive Evaluation**: Assessment of 10 state-of-the-art models across three counting paradigms
- **Diagnostic Analysis**: Reveals critical limitations in current vision models for intent-driven counting
- **Real-world Relevance**: Addresses scenarios where accurate counting depends on subtle visual distinctions

## Dataset

### Statistics

| Property | Value |
|----------|-------|
| Total Images | 681 |
| Object Categories | 54 |
| Subcategories | 98 |
| Supercategories | 5 (Food, Fun, Household, Office, Other) |
| Inter-category pairs | 50 |
| Intra-category pairs | 47 |

<p align="center">
  <img src="img/statistics.png" alt="PairTally Dataset Statistics" width="100%">
</p>

### Structure

```
PairTally/
├── images/                    # 681 high-resolution images
├── annotations/
│   ├── pairtally_annotations_simple.json
│   ├── pairtally_annotations_inter_simple.json
│   ├── pairtally_annotations_intra_simple.json
│   └── image_metadata.json
└── evaluation/                # Evaluation scripts and tools
```

### Attribute Distribution

| Attribute Type | Percentage | Examples |
|---------------|------------|----------|
| Color | 43.5% | Black vs. white checker pieces |
| Shape/Texture | 42.5% | Spiral vs. penne pasta |
| Size | 14.1% | Large vs. small marbles |

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

### Model Categories Evaluated

| Category | Models | Input Type |
|----------|--------|------------|
| Exemplar-based | FamNet, DAVE, GeCo, LoCA | 3 bounding box exemplars |
| Language-prompted | CountGD, LLMDet | Text prompts (category names) |
| Vision-Language | Ovis2, Qwen2.5-VL, LLaMA-3.2, InternVL3 | Natural language instructions |

## Results

### Overall Performance (All Objects Counting)

| Model | Type | MAE ↓ | RMSE ↓ |
|-------|------|-------|--------|
| Mean Baseline | - | 98.56 | 151.45 |
| Median Baseline | - | 89.12 | 159.01 |
| **GeCo** | Exemplar | **53.07** | **98.00** |
| CountGD | Text+Exemplar | 57.33 | 108.93 |
| CountGD (Text) | Text-only | 160.46 | 220.64 |
| LoCA | Exemplar | 62.78 | 136.76 |
| DAVE | Exemplar | 69.49 | 130.42 |
| FamNet | Exemplar | 88.30 | 148.42 |
| LLaMA-3.2 | VLM | 97.56 | 175.80 |
| Qwen2.5-VL | VLM | 99.88 | 174.93 |
| LLMDet | Text-only | 107.84 | 177.66 |
| Ovis2 | VLM | 111.56 | 174.16 |
| InternVL3 | VLM | 115.98 | 179.89 |

### Performance with Distractors (Inter vs Intra)

| Model | Inter MAE ↓ | Intra MAE ↓ | Inter NAE ↓ | Intra NAE ↓ | Inter CI ↑ | Intra CI ↑ |
|-------|-------------|-------------|-------------|-------------|------------|------------|
| Mean Baseline | 39.42 | 66.71 | 0.714 | 0.535 | 0.969 | 0.977 |
| Median Baseline | 37.38 | 57.25 | 1.587 | 0.776 | 0.970 | 0.987 |
| CountGD | 39.78 | 56.54 | 0.673 | 0.906 | 0.934 | 0.977 |
| GeCo | 45.05 | 54.80 | 0.777 | 0.935 | 0.985 | 0.986 |
| DAVE | 46.27 | 46.75 | 0.779 | 0.797 | 0.982 | 0.980 |
| Qwen2.5-VL | 46.35 | 67.86 | 0.598 | 0.712 | 0.927 | 0.983 |
| LLaMA-3.2 | 49.14 | 58.73 | 0.730 | 0.740 | 0.949 | 0.992 |
| CountGD (Text) | 50.23 | 53.93 | 0.712 | 0.841 | 0.914 | 0.971 |
| Ovis2 | 56.87 | 74.24 | 0.711 | 0.736 | 0.981 | 0.974 |
| InternVL3 | 55.89 | 71.47 | 0.667 | 0.721 | 0.987 | 0.999 |
| LoCA | 71.89 | 57.45 | 1.177 | 0.950 | 0.806 | 0.999 |
| FamNet | 66.97 | 74.75 | 1.363 | 1.440 | 0.893 | 0.959 |
| LLMDet | 78.72 | 142.08 | 0.661 | 1.060 | 0.985 | 0.990 |

### Attribute-Specific Performance (Intra-category)

| Model | Color MAE ↓ | Color RMSE ↓ | Color NAE ↓ | Size MAE ↓ | Size RMSE ↓ | Size NAE ↓ | Texture MAE ↓ | Texture RMSE ↓ | Texture NAE ↓ |
|-------|-------------|--------------|-------------|------------|-------------|-------------|---------------|----------------|---------------|
| Mean Baseline | 55.16 | 83.51 | 1.698 | 25.81 | 36.14 | 1.154 | 43.75 | 59.38 | 0.838 |
| Median Baseline | 49.37 | 88.01 | 0.967 | 24.71 | 36.75 | 0.586 | 40.89 | 61.71 | 0.720 |
| **DAVE** | 63.44 | 89.16 | **0.738** | 33.26 | 39.31 | 1.293 | 34.14 | 43.00 | **0.693** |
| CountGD (Text) | 64.51 | 98.99 | 0.760 | 38.95 | 45.61 | 1.410 | 48.06 | 73.20 | **0.735** |
| CountGD | 75.40 | 117.32 | 0.856 | 36.30 | 42.35 | 1.402 | 43.95 | 57.42 | 0.793 |
| GeCo | 63.40 | 88.77 | 0.791 | 35.06 | 41.13 | 1.345 | 52.53 | 74.55 | 0.946 |
| LoCA | 65.37 | 95.24 | 0.799 | 33.34 | 39.66 | **1.244** | 57.33 | 91.11 | 1.007 |
| FamNet | 84.92 | 117.33 | 1.296 | 56.32 | 75.44 | 1.859 | 70.45 | 90.64 | 1.448 |
| LLMDet | 118.29 | 151.01 | 2.12 | 68.68 | 82.73 | 4.18 | 89.33 | 116.06 | 2.04 |

### Key Findings

1. **Model Limitations**: Even best-performing models achieve MAE > 50, indicating substantial room for improvement
2. **Distractor Sensitivity**: Most models struggle more with intra-category pairs than inter-category discrimination  
3. **Attribute Hierarchy**: Color differences are most distinguishable, size differences most challenging
4. **VLM Performance**: Large vision-language models underperform specialized counting methods
5. **Overcounting Bias**: Models frequently count all objects rather than following specific prompts

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
  <a href="#citation">Paper</a> •
  <a href="https://drive.google.com/file/d/1TnenXS4yFicjo81NnmClfzgc8ltmmeBv/view">Dataset</a>
</p>
