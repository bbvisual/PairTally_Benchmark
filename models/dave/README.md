# DAVE - A Detect-and-Verify Paradigm for Low-Shot Counting

This directory contains the evaluation setup for DAVE (CVPR 2024) on the PairTally dataset.

## Original Paper
**DAVE – A Detect-and-Verify Paradigm for Low-Shot Counting**  
Jer Pelhan, Alan Lukežič, Vitjan Zavrtanik, Matej Kristan  
CVPR 2024  
[[Paper]](https://arxiv.org/pdf/2404.16622) [[Code]](https://github.com/jerpelhan/DAVE)

## Setup Instructions

### 1. Clone Original Repository
```bash
# Clone the official DAVE repository
git clone https://github.com/jerpelhan/DAVE.git
cd DAVE
```

### 2. Environment Setup
```bash
# Create conda environment
conda create -n dave python==3.8
conda activate dave

# Install PyTorch and dependencies
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy scikit-image scikit-learn tqdm pycocotools

# For text-prompt-based counting (optional)
conda install transformers
```

### 3. Download Pre-trained Models
```bash
# Download pre-trained models from:
# https://drive.google.com/drive/folders/10O4SB3Y380hcKPIK8Dt8biniVbdQ4dH4?usp=sharing

# Place models in the material/ directory
mkdir -p material
# Download and place:
# - DAVE_0_shot.pth (zero-shot model)
# - DAVE_3_shot.pth (few-shot model)
```

### 4. Install Detectron2 (for evaluation)
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## PairTally Evaluation

### Files in this Directory

**Evaluation Scripts:**
- `evaluate_pairtally_count_both_classes.py` - **Combined evaluation**: 2 positive + 1 negative exemplars, count both classes
- `evaluate_pairtally_count_one_class.py` - **Single-class evaluation**: Positive exemplars only, count 1 class at a time

**Run Scripts:**
- `run_count_both_classes.sh` - Shell script for combined evaluation
- `run_count_one_class.sh` - Shell script for single-class evaluation
- `run_custom_eval.sh` - Additional evaluation script
- `README.md` - This file

### Running Evaluation

DAVE evaluation scripts are ready to run. The scripts automatically use the dataset at `../../dataset/pairtally_dataset/`.

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

**Option 3: Custom Evaluation**
```bash
./run_custom_eval.sh
```

**For Testing (limit to N images):**
```bash
./run_count_both_classes.sh --output_limit 10
```

### Evaluation Parameters

The evaluation uses the following key parameters:
- **Model**: DAVE_3_shot (few-shot counting)
- **Backbone**: ResNet-50 with SwAV pre-training
- **Number of exemplars**: 3 (few-shot mode)
- **Detection threshold**: Adaptive based on exemplars
- **Verification**: Enabled for false positive removal

### Model Architecture

DAVE uses a detect-and-verify paradigm:
1. **Detection Stage**: High-recall object detection using exemplars
2. **Verification Stage**: False positive removal through similarity verification
3. **Counting**: Final count based on verified detections

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

### Output Structure

Results are saved to `../../results/` with the following structure:

**Single-Class Mode:**
```
../../results/DAVE-quantitative/pairtally_dataset/
├── DAVE_quantitative_results.json
├── DAVE_quantitative_results.pkl
└── DAVE_summary.txt

../../results/DAVE-qualitative/pairtally_dataset/
├── positive_qualitative_data.json
├── negative_qualitative_data.json
└── complete_qualitative_data.json
```

**Combined Mode:**
```
../../results/DAVE-quantitative-combined/pairtally_dataset/
├── DAVE_combined_quantitative_results.json
├── DAVE_combined_quantitative_results.pkl
└── DAVE_combined_summary.txt

../../results/DAVE-qualitative-combined/pairtally_dataset/
└── DAVE_combined_detailed_results.json
```

### Troubleshooting

**Common Issues:**
1. **Missing detectron2**: Install with pip install 'git+https://github.com/facebookresearch/detectron2.git'
2. **Dataset not found**: Verify dataset is at `../../dataset/pairtally_dataset/`
3. **Model not found**: Ensure DAVE model weights are properly configured
4. **Environment errors**: Activate dave conda environment

**Performance Tips:**
- Use `--output_limit N` for testing on subset of images
- Check CUDA availability with `echo $CUDA_VISIBLE_DEVICES`
- Monitor GPU memory usage during evaluation
- Results are automatically saved to `../../results/`

### Citation

If you use DAVE in your research, please cite:

```bibtex
@InProceedings{Pelhan_2024_CVPR,
    author    = {Pelhan, Jer and Luke\v{z}ic, Alan and Zavrtanik, Vitjan and Kristan, Matej},
    title     = {DAVE - A Detect-and-Verify Paradigm for Low-Shot Counting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {23293-23302}
}
```
