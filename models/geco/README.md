# GeCo - A Novel Unified Architecture for Low-Shot Counting

This directory contains the evaluation setup for GeCo (NeurIPS 2024) on the PairTally dataset.

## Original Paper
**A Novel Unified Architecture for Low-Shot Counting by Detection and Segmentation**  
Jer Pelhan, Alan Lukezic, Vitjan Zavrtanik, Matej Kristan  
NeurIPS 2024  
[[Paper]](https://arxiv.org/pdf/2409.18686) [[Code]](https://github.com/jerpelhan/GeCo)

## Setup Instructions

### 1. Clone Original Repository
```bash
# Clone the official GeCo repository
git clone https://github.com/jerpelhan/GeCo.git
cd GeCo
```

### 2. Environment Setup
```bash
# Create conda environment
conda create -n geco_test python=3.8
conda activate geco_test

# Install PyTorch and dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib tqdm pycocotools scipy

# Install detectron2 for evaluation
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 3. Download Pre-trained Weights
```bash
# Download pretrained weights from:
# https://drive.google.com/file/d/1wjOF9MWkrVJVo5uG3gVqZEW9pwRq_aIk/view?usp=sharing

# Place in pretrained_models/
mkdir -p pretrained_models
# Download and place: pretrained_models/model_weights.pth
```

## PairTally Evaluation

### Files in this Directory

**Evaluation Scripts:**
- `evaluate_pairtally_count_both_classes.py` - **Combined evaluation**: 2 positive + 1 negative exemplars, count both classes
- `evaluate_pairtally_count_one_class.py` - **Single-class evaluation**: Positive exemplars only, count 1 class at a time

**Run Scripts:**
- `run_count_both_classes.sh` - Shell script for combined evaluation
- `run_count_one_class.sh` - Shell script for single-class evaluation
- `README.md` - This file

### Running Evaluation

GeCo is already set up in this directory. The scripts automatically use the dataset at `../../dataset/pairtally_dataset/`.

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
./run_count_both_classes.sh
```

**Option 2: Single-Class Mode (One class at a time)**
```bash
./run_count_one_class.sh
```

**For Testing (limit to N images):**
```bash
./run_count_both_classes.sh --output_limit 10
```

### Evaluation Modes

**Combined Mode** (`run_count_both_classes.sh`):
- Provides **2 positive exemplars + 1 negative exemplar** per image
- Asks model to count **both object classes simultaneously**
- Tests ability to distinguish between different object types in the same scene
- More challenging as model must handle distractors

**Single-Class Mode** (`run_count_one_class.sh`):
- Provides **positive exemplars only** for the target class
- Asks model to count **one class at a time**
- Simpler task focusing on counting accuracy for a single object type
- Two separate runs per image (one for each object class)

### Evaluation Parameters

The evaluation uses the following key parameters:
- **Model**: GeCo pretrained weights (`pretrained_models/model_weights.pth`)
- **Image Size**: 1024x1024
- **Device**: CUDA (configurable via `CUDA_VISIBLE_DEVICES`)
- **Exemplar-based counting**: Visual exemplar boxes for object identification

### Output Structure

Results are saved to `../../results/` with the following structure:

**Single-Class Mode:**
```
../../results/GeCo-quantitative/pairtally_dataset/
├── GeCo_updated_quantitative_results.json
├── GeCo_updated_quantitative_results.pkl
└── GeCo_updated_summary.txt

../../results/GeCo-qualitative/pairtally_dataset/
├── positive_qualitative_data.json
├── negative_qualitative_data.json
└── complete_qualitative_data.json
```

**Combined Mode:**
```
../../results/GeCo-quantitative-combined/pairtally_dataset/
├── GeCo_updated_combined_quantitative_results.json
├── GeCo_updated_combined_quantitative_results.pkl
└── GeCo_updated_combined_summary.txt

../../results/GeCo-qualitative-combined/pairtally_dataset/
└── GeCo_updated_combined_detailed_results.json
```


### Troubleshooting

**Common Issues:**
1. **CUDA out of memory**: Reduce image size or batch processing
2. **Dataset not found**: Verify dataset is at `../../dataset/pairtally_dataset/`
3. **Model not found**: Ensure `pretrained_models/model_weights.pth` exists
4. **Environment errors**: Activate geco_test conda environment

**Performance Tips:**
- Use `--output_limit N` for testing on subset of images
- Check CUDA availability with `echo $CUDA_VISIBLE_DEVICES`
- Monitor GPU memory usage during evaluation
- Results are automatically saved to `../../results/`

### Citation

If you use GeCo in your research, please cite:

```bibtex
@article{pelhan2024novel,
  title={A Novel Unified Architecture for Low-Shot Counting by Detection and Segmentation},
  author={Pelhan, Jer and Lukezic, Alan and Zavrtanik, Vitjan and Kristan, Matej},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={66260--66282},
  year={2024}
}
```
