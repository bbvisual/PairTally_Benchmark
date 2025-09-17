# CountGD - Multi-Modal Open-World Counting

This directory contains the evaluation setup for CountGD (NeurIPS 2024) on the PairTally dataset.

## Original Paper
**CountGD: Multi-Modal Open-World Counting**  
Niki Amini-Naieni, Tengda Han, & Andrew Zisserman  
NeurIPS 2024  
[[Paper]](https://arxiv.org/abs/2407.04619) [[Project Page]](https://www.robots.ox.ac.uk/~vgg/research/countgd/) [[Code]](https://github.com/niki-amini-naieni/CountGD)

## Setup Instructions

### 1. Clone Original Repository
```bash
# Clone the official CountGD repository
git clone https://github.com/niki-amini-naieni/CountGD.git
cd CountGD
```

### 2. Environment Setup
```bash
# Create conda environment
conda create -n countgd python=3.9.19
conda activate countgd

# Install dependencies
pip install -r requirements.txt
export CC=/usr/bin/gcc-11

# Build GroundingDINO
cd models/GroundingDINO/ops
python setup.py build install
python test.py  # should result in 6 lines of * True

# Install Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git
cd ../../../
```

### 3. Download Pre-trained Weights
```bash
# Create checkpoints directory
mkdir checkpoints

# Download BERT weights
python download_bert.py

# Download GroundingDINO weights
wget -P checkpoints https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

# Download SAM weights
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Download CountGD model weights
# Download from: https://drive.google.com/file/d/1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI/view?usp=sharing
# Place as: checkpoints/checkpoint_fsc147_best.pth
gdown 1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI
```

## PairTally Evaluation

### Files in this Directory

**Evaluation Scripts:**
- `evaluate_pairtally_count_both_classes.py` - **Combined evaluation**: 2 positive + 1 negative exemplars, count both classes
- `evaluate_pairtally_count_one_class.py` - **Single-class evaluation**: Positive exemplars only, count 1 class at a time
- `evaluate_pairtally_count_both_classes_text_only.py` - Combined text-only evaluation (no exemplar boxes)
- `evaluate_pairtally_count_one_class_text_only.py` - Single-class text-only evaluation

**Run Scripts:**
- `run_count_both_classes.sh` - Shell script for combined evaluation
- `run_count_one_class.sh` - Shell script for single-class evaluation  
- `run_count_both_classes_text_only.sh` - Shell script for combined text-only evaluation
- `run_count_one_class_text_only.sh` - Shell script for single-class text-only evaluation
- `SETUP.md` - Detailed setup instructions
- `README.md` - This file

### Evaluation Modes

**Combined Mode** (`run_count_both_classes.sh`):
- Provides **2 positive exemplars + 1 negative exemplar** per image
- Uses combined text prompt: "positive_class and negative_class"
- Asks model to count **both object classes simultaneously**
- Tests ability to distinguish between different object types in the same scene
- More challenging as model must handle distractors

**Single-Class Mode** (`run_count_one_class.sh`):
- Provides **positive exemplars only** for the target class
- Uses separate text prompts for each class
- Asks model to count **one class at a time**
- Simpler task focusing on counting accuracy for a single object type
- Two separate runs per image (one for each object class)

**Text-Only Modes** (`run_count_*_text_only.sh`):
- Same as above modes but **without visual exemplar boxes**
- Uses only text descriptions for object identification
- Tests pure language-based counting capability

### Running Evaluation

All evaluation scripts are ready to run. The scripts automatically use the dataset at `../../dataset/pairtally_dataset/`.

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

**Option 3: Combined Text-Only Mode**
```bash
./run_count_both_classes_text_only.sh
```

**Option 4: Single-Class Text-Only Mode**
```bash
./run_count_one_class_text_only.sh
```

**For Testing (limit to N images):**
```bash
./run_count_both_classes.sh --output_limit 10
```

### Evaluation Parameters

The evaluation uses the following key parameters:
- **Model**: FSC147 pretrained checkpoint (`checkpoint_fsc147_best.pth`)
- **Configuration**: `cfg_fsc147_vit_b.py`
- **Confidence threshold**: 0.23 (single-class), 0.3 (combined)
- **Text + Visual mode**: Both text descriptions and visual exemplars (default)
- **Text-only mode**: Available via `*_text_only.sh` scripts
- **Device**: CUDA (configurable via `CUDA_VISIBLE_DEVICES`)

### Output Structure

Results are saved to `../../results/` with the following structure:

**Single-Class Mode:**
```
../../results/CountGD-quantitative/pairtally_dataset/
├── CountGD_quantitative_results.json
├── CountGD_quantitative_results.pkl
└── CountGD_summary.txt

../../results/CountGD-qualitative/pairtally_dataset/
├── positive_qualitative_data.json
├── negative_qualitative_data.json
└── complete_qualitative_data.json
```

**Combined Mode:**
```
../../results/CountGD-count-both-classes-quantitative/pairtally_dataset/
├── CountGD-Combined_quantitative_results.json
├── CountGD-Combined_quantitative_results.pkl
└── CountGD-Combined_summary.txt

../../results/CountGD-count-both-classes-qualitative/pairtally_dataset/
└── CountGD-Combined_detailed_results.json
```


### Troubleshooting

**Common Issues:**
1. **CUDA out of memory**: Reduce batch size or use smaller confidence threshold
2. **Dataset not found**: Verify dataset is at `../../dataset/pairtally_dataset/`
3. **Model not found**: Ensure `CountGD/checkpoints/checkpoint_fsc147_best.pth` exists
4. **Environment errors**: Activate countgd conda environment

**Performance Tips:**
- Use `--output_limit N` for testing on subset of images
- Check CUDA availability with `echo $CUDA_VISIBLE_DEVICES`
- Monitor GPU memory usage during evaluation
- Results are automatically saved to `../../results/`

### Citation

If you use CountGD in your research, please cite:

```bibtex
@InProceedings{AminiNaieni24,
  author = "Amini-Naieni, N. and Han, T. and Zisserman, A.",
  title = "CountGD: Multi-Modal Open-World Counting",
  booktitle = "Advances in Neural Information Processing Systems (NeurIPS)",
  year = "2024",
}
```
