# Can Current AI Models Count What We Mean, Not What They See?

**A Benchmark and Systematic Evaluation**

[![Paper](https://img.shields.io/badge/Paper-DICTA2025-blue)](https://your-paper-link-here)
[![Dataset](https://img.shields.io/badge/Dataset-PairTally-green)](https://your-dataset-link-here)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Overview

This repository contains the code and data for our DICTA 2025 paper "Can Current AI Models Count What We Mean, Not What They See? A Benchmark and Systematic Evaluation". We introduce **PairTally**, a benchmark dataset specifically designed to evaluate fine-grained visual counting, and systematically evaluate 10 state-of-the-art models across different counting paradigms.

### Key Contributions

1. **PairTally Dataset**: 681 high-resolution images with object pairs, designed for fine-grained counting evaluation
2. **Systematic Evaluation**: Benchmarking of 10 models including 4 exemplar-based counters, 2 language-prompted detectors, and 4 Vision-Language Models (VLMs)
3. **Fine-grained Analysis**: First systematic evaluation of INTER-category vs INTRA-category counting with attribute-specific analysis (color, size, texture/shape)

## Models Evaluated

### Exemplar-Based Counting Models (4)
- **FamNet** - Learning to Count Everything (CVPR 2021) - First exemplar-guided counting model
- **DAVE** - A Detect-and-Verify Paradigm for Low-Shot Counting (CVPR 2024)  
- **GeCo** - A Novel Unified Architecture for Low-Shot Counting (NeurIPS 2024)
- **LOCA** - Low-Shot Object Counting with Iterative Prototype Adaptation

### Language-Prompted Detectors (2)
- **CountGD** - Multi-Modal Open-World Counting (NeurIPS 2024) - Text + exemplar prompts
- **LLMDet** - Learning Strong Open-Vocabulary Object Detectors - Text-only prompts

### Vision-Language Models (4)
- **Ovis2** - Structural Embedding Alignment for Multimodal LLM (16B)
- **Qwen2.5-VL** - Enhanced Vision-Language Model (7B)
- **LLaMA-3.2** - Vision-Instruct Model (11B)
- **InternVL3** - Advanced Multimodal Model (14B)

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/DICTA25-Can-AI-Models-Count-Release.git
cd DICTA25-Can-AI-Models-Count-Release

# Set up environments for different models
./scripts/setup/setup_all_environments.sh
```

### 2. Dataset Preparation

```bash
# Download and prepare the PairTally dataset
cd dataset
python tools/download_dataset.py
python tools/prepare_annotations.py
```

### 3. Run Evaluations

```bash
# Run all model evaluations
./scripts/evaluation/run_all_evaluations.sh

# Or run individual models
./scripts/evaluation/run_countgd.sh
./scripts/evaluation/run_dave.sh
./scripts/evaluation/run_geco.sh
# ... etc
```

### 4. Generate Results

```bash
# Generate summary tables and figures
python scripts/analysis/generate_summary_tables.py
python scripts/visualization/create_comparison_plots.py
```

## Repository Structure

```
DICTA25-Can-AI-Models-Count-Release/
├── README.md                 # This file
├── LICENSE                   # License information
├── dataset/                  # Dataset files and tools
│   ├── annotations/          # Annotation files in various formats
│   ├── images/              # Image data (download separately)
│   └── tools/               # Dataset preparation and conversion tools
├── models/                   # Model-specific code and configurations
│   ├── countgd/             # CountGD model setup and evaluation
│   ├── dave/                # DAVE model setup and evaluation
│   ├── geco/                # GeCo model setup and evaluation
│   ├── learningtocount/     # LearningToCountEverything setup
│   ├── loca/                # LOCA model setup and evaluation
│   └── vlms/                # Vision-Language Models evaluation
├── evaluation/               # Evaluation scripts and results
│   ├── individual/          # Individual model evaluation scripts
│   ├── combined/            # Combined evaluation and comparison scripts
│   └── analysis/            # Analysis and metrics calculation
├── results/                  # Results and outputs
│   ├── figures/             # Generated plots and visualizations
│   ├── tables/              # Summary tables and metrics
│   └── raw_data/            # Raw evaluation outputs
├── scripts/                  # Utility scripts
│   ├── setup/               # Environment and dependency setup
│   ├── evaluation/          # Evaluation pipeline scripts
│   └── visualization/       # Plotting and visualization scripts
└── requirements/             # Dependencies and environment files
    ├── environments/        # Conda environment files
    └── models/              # Model-specific requirements
```

## Dataset Details

The **PairTally** dataset contains:
- **681 high-resolution images** across 5 super-categories (Food, Fun, Household, Office, Other)
- **54 object categories** with 98 subcategories total
- **97 subcategory pairs**: 47 INTRA-category + 50 INTER-category pairs
- **Fine-grained attributes**: Color (43.5%), Texture/Shape (42.5%), Size (14.1%)

### Super-Categories
- **Food**: pasta, lime, peppercorn, tomato, chili, peanut, bean, seeds, coffee candy, garlic, shallot
- **Fun**: checker pieces, mahjong tiles, lego pieces, chess pieces, puzzle pieces, poker chips, playing cards, marbles, dice
- **Household**: toothpicks, cotton buds, pills, batteries, hair clippers, bills, coins, bottle caps, shirt buttons, utensils
- **Office**: push pins, stickers, craft sticks, rubber bands, sticky notes, paper clips, pens, pencils, rhinestones, zip ties, safety pins
- **Other**: screws, bolts, nuts, washers, beads, clips, pegs, stones, novelty buttons

## Results Summary

### Overall Performance (MAE/RMSE - Lower is Better)

| Model | Overall MAE | Overall RMSE | Model Type |
|-------|-------------|--------------|------------|
| **GeCo** | **53.07** | **98.00** | Exemplar-based |
| **CountGD** | **57.33** | **108.93** | Language-prompted |
| **LOCA** | 62.78 | 136.76 | Exemplar-based |
| **DAVE** | 69.49 | 130.42 | Exemplar-based |
| **FamNet** | 88.30 | 148.42 | Exemplar-based |
| **LLaMA-3.2** | 97.56 | 175.80 | Vision-Language Model |
| **Qwen2.5-VL** | 99.88 | 174.93 | Vision-Language Model |
| **LLMDet** | 107.84 | 177.66 | Language-prompted |
| **Ovis2** | 111.56 | 174.16 | Vision-Language Model |
| **InternVL3** | 115.98 | 179.89 | Vision-Language Model |

### Key Findings

1. **Best Overall**: GeCo (MAE: 53.07) and CountGD (MAE: 57.33) achieve the best performance
2. **VLM Limitations**: All Vision-Language Models perform poorly (MAE > 97), indicating they struggle with precise enumeration
3. **INTER vs INTRA**: Models generally perform better on INTER-category (different objects) than INTRA-category (similar objects) counting
4. **Attribute Sensitivity**: Color differences are easiest to distinguish, while size and texture/shape are more challenging

*Detailed results and analysis available in the `results/` directory.*

## Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@inproceedings{your-name2025dicta25,
  title={Can Current AI Models Count What We Mean, Not What They See? A Benchmark and Systematic Evaluation},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the International Conference on Digital Image Computing: Techniques and Applications (DICTA)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the authors of the original model implementations:
- CountGD: [Amini-Naieni et al., NeurIPS 2024]
- DAVE: [Pelhan et al., CVPR 2024]
- GeCo: [Pelhan et al., NeurIPS 2024]
- LearningToCountEverything: [Ranjan et al., CVPR 2021]

## Contact

For questions or issues, please contact:
- [Your Name] - [your.email@institution.edu]
- [Co-author Name] - [coauthor.email@institution.edu]

## Updates

- **[Date]**: Initial release with dataset and evaluation code
- **[Date]**: Added VLM evaluation scripts
- **[Date]**: Updated with final paper results
