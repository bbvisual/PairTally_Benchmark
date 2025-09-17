# LOCA - Low-shot Object Counting with Adversarial Learning

This directory contains the evaluation setup for LOCA (CVPR 2022) on the PairTally dataset.

## Original Paper
**Low-shot Object Counting with Adversarial Learning**  
Nikola Djukic, Alan Lukežič, Vitjan Zavrtanik, Matej Kristan  
CVPR 2022  
[[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Dukic_A_Low-Shot_Object_Counting_Network_With_Iterative_Prototype_Adaptation_ICCV_2023_paper.pdf) [[Code]](https://github.com/djukicn/LOCA)

## Setup Instructions

### 1. Clone Original Repository
```bash
# Clone the official LOCA repository
git clone https://github.com/djukicn/LOCA.git loca
cd loca
```

### 2. Environment Setup
```bash
# Create conda environment
conda create -n loca python=3.8
conda activate loca

# Install PyTorch and dependencies
pip install torch torchvision
pip install opencv-python pillow numpy matplotlib
pip install tqdm argparse
```

### 3. Download Pre-trained Weights
```bash
# Download pretrained weights from the LOCA repository
# Follow the instructions in the repository to get the model weights
# Place in loca/pretrained_models/
```

## PairTally Evaluation

### Files in this Directory

**Evaluation Scripts:**
- `evaluate_pairtally_count_both_classes.py` - **Combined evaluation**: 2 positive + 1 negative exemplars, count both classes
- `evaluate_pairtally_count_one_class.py` - **Single-class evaluation**: Positive exemplars only, count 1 class at a time

**Run Scripts:**
- `run_count_both_classes.sh` - Shell script for combined evaluation
- `run_count_one_class.sh` - Shell script for single-class evaluation
- `SETUP.md` - Detailed setup instructions
- `README.md` - This file

### Running Evaluation

LOCA evaluation scripts are ready to run. The scripts automatically use the dataset at `../../dataset/pairtally_dataset/`.

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
- **Model**: LOCA pretrained checkpoint
- **Architecture**: CNN backbone with adversarial learning components
- **Training Strategy**: Adversarial learning for robust feature representation
- **Device**: CUDA (configurable via `CUDA_VISIBLE_DEVICES`)
- **Exemplar-based counting**: Visual exemplar boxes for object identification

### Output Structure

Results are saved to `../../results/` with the following structure:

**Single-Class Mode:**
```
../../results/LOCA-quantitative/pairtally_dataset/
├── LOCA_quantitative_results.json
├── LOCA_quantitative_results.pkl
└── LOCA_summary.txt

../../results/LOCA-qualitative/pairtally_dataset/
├── positive_qualitative_data.json
├── negative_qualitative_data.json
└── complete_qualitative_data.json
```

**Combined Mode:**
```
../../results/LOCA-quantitative-combined/pairtally_dataset/
├── LOCA_combined_quantitative_results.json
├── LOCA_combined_quantitative_results.pkl
└── LOCA_combined_summary.txt

../../results/LOCA-qualitative-combined/pairtally_dataset/
└── LOCA_combined_detailed_results.json
```

### Troubleshooting

**Common Issues:**
1. **Model not found**: Ensure LOCA model weights are in `loca/pretrained_models/`
2. **Dataset not found**: Verify dataset is at `../../dataset/pairtally_dataset/`
3. **Import errors**: Install LOCA dependencies and ensure Python path is correct
4. **Environment errors**: Activate loca conda environment

**Performance Tips:**
- Use `--output_limit N` for testing on subset of images
- Check CUDA availability with `echo $CUDA_VISIBLE_DEVICES`
- Monitor GPU memory usage during evaluation
- Results are automatically saved to `../../results/`

### Citation

If you use LOCA in your research, please cite:

```bibtex
@InProceedings{Djukic_2022_CVPR,
    author    = {Djukic, Nikola and Luke\v{z}i\v{c}, Alan and Zavrtanik, Vitjan and Kristan, Matej},
    title     = {Low-Shot Object Counting With Adversarial Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {21600-21609}
}
```