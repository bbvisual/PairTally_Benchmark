# FamNet - Learning to Count Everything

This directory contains the evaluation setup for FamNet (CVPR 2021) on the PairTally dataset.

## Original Paper
**Learning To Count Everything**  
Viresh Ranjan, Udbhav Sharma, Thu Nguyen, Minh Hoai  
CVPR 2021  
[[Paper]](https://arxiv.org/abs/2104.08391) [[Code]](https://github.com/cvlab-stonybrook/LearningToCountEverything)

## Setup Instructions

### 1. Clone Original Repository
```bash
# Clone the official Learning To Count Everything repository
git clone https://github.com/cvlab-stonybrook/LearningToCountEverything.git
cd LearningToCountEverything
```

### 2. Environment Setup
```bash
# Create conda environment
conda create -n famnet python=3.8
conda activate famnet

# Install PyTorch and dependencies
pip install torch torchvision
pip install opencv-python pillow numpy matplotlib
pip install argparse tqdm
```

### 3. Download Pre-trained Weights
```bash
# Download pretrained weights from the FamNet repository
# Follow the instructions in the repository to get FSC147 pretrained weights
# Place in LearningToCountEverything/pretrained_models/
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

FamNet evaluation scripts are ready to run. The scripts automatically use the dataset at `../../dataset/pairtally_dataset/`.

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
- **Model**: FamNet pretrained on FSC147
- **Architecture**: ResNet-50 backbone with FamNet counting head
- **Input Size**: 384x384 (standard FamNet input size)
- **Device**: CUDA (configurable via `CUDA_VISIBLE_DEVICES`)
- **Exemplar-based counting**: Visual exemplar boxes for object identification

### Output Structure

Results are saved to `../../results/` with the following structure:

**Single-Class Mode:**
```
../../results/FamNet-quantitative/pairtally_dataset/
├── FamNet_quantitative_results.json
├── FamNet_quantitative_results.pkl
└── FamNet_summary.txt

../../results/FamNet-qualitative/pairtally_dataset/
├── positive_qualitative_data.json
├── negative_qualitative_data.json
└── complete_qualitative_data.json
```

**Combined Mode:**
```
../../results/FamNet-quantitative-combined/pairtally_dataset/
├── FamNet_combined_quantitative_results.json
├── FamNet_combined_quantitative_results.pkl
└── FamNet_combined_summary.txt

../../results/FamNet-qualitative-combined/pairtally_dataset/
└── FamNet_combined_detailed_results.json
```

### Troubleshooting

**Common Issues:**
1. **Model not found**: Ensure FamNet model weights are in `LearningToCountEverything/pretrained_models/`
2. **Dataset not found**: Verify dataset is at `../../dataset/pairtally_dataset/`
3. **Import errors**: Install FamNet dependencies and ensure Python path is correct
4. **Environment errors**: Activate famnet conda environment

**Performance Tips:**
- Use `--output_limit N` for testing on subset of images
- Check CUDA availability with `echo $CUDA_VISIBLE_DEVICES`
- Monitor GPU memory usage during evaluation
- Results are automatically saved to `../../results/`

### Citation

If you use FamNet in your research, please cite:

```bibtex
@InProceedings{Ranjan_2021_CVPR,
    author    = {Ranjan, Viresh and Sharma, Udbhav and Nguyen, Thu and Hoai, Minh},
    title     = {Learning To Count Everything},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3394-3403}
}
```