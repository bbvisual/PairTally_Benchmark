# CountGD - Multi-Modal Open-World Counting

This directory contains the evaluation setup for CountGD (NeurIPS 2024) on the DICTA25 dataset.

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
```

## PairTally Evaluation

### Files in this Directory

**Evaluation Scripts:**
- `evaluate_DICTA25_combined.py` - **Combined evaluation**: 2 positive + 1 negative exemplars, count both classes
- `evaluate_DICTA25_custom.py` - **Custom evaluation**: Positive exemplars only, count 1 class at a time
- `evaluate_DICTA25_text_only.py` - Text-only evaluation (no exemplar boxes)
- `evaluate_DICTA25_combined_text_only.py` - Combined text-only evaluation

**Run Scripts:**
- `run_combined_eval.sh` - Shell script for combined evaluation
- `run_custom_eval.sh` - Shell script for custom evaluation  
- `run_custom_eval_text_only.sh` - Shell script for text-only custom evaluation
- `README.md` - This file

### Evaluation Modes

**Combined Mode** (`evaluate_DICTA25_combined.py`):
- Provides **2 positive exemplars + 1 negative exemplar** per image
- Asks model to count **both object classes simultaneously**
- Tests ability to distinguish between different object types in the same scene
- More challenging as model must handle distractors

** Custom Mode** (`evaluate_DICTA25_custom.py`):
- Provides **positive exemplars only** for the target class
- Asks model to count **one class at a time**
- Simpler task focusing on counting accuracy for a single object type
- Two separate runs per image (one for each object class)

### Running Evaluation

1. **Copy evaluation scripts to CountGD directory:**
```bash
cp evaluate_DICTA25_*.py /path/to/CountGD/
cp run_combined_eval.sh /path/to/CountGD/
```

2. **Update paths in the evaluation scripts:**
Edit the scripts to point to your PairTally dataset location:
```python
base_data_path = "/path/to/PairTally-Benchmark-Release/dataset"
# The script will use: base_data_path/pairtally_dataset/
```

**Important**: The evaluation scripts expect the FSC147-compatible format in `dataset/pairtally_dataset/`:
- Annotations: `dataset/pairtally_dataset/annotations/pairtally_annotations.json`
- Images: `dataset/pairtally_dataset/images/`

3. **Run evaluation:**

**Option A: Combined Mode (Recommended for paper results)**
```bash
cd /path/to/CountGD
conda activate countgd
./run_combined_eval.sh
```

**Option B: Custom Mode (Single-class counting)**
```bash
cd /path/to/CountGD
conda activate countgd
./run_custom_eval.sh
```

**Option C: Text-Only Custom Mode**
```bash
cd /path/to/CountGD
conda activate countgd
./run_custom_eval_text_only.sh
```

### Evaluation Parameters

The evaluation uses the following key parameters:
- **Text + Visual mode**: Both text descriptions and visual exemplars
- **SAM test-time normalization**: Enabled for better accuracy
- **Confidence threshold**: 0.3 (default)
- **Crop mode**: Enabled for better object localization
- **Remove bad exemplars**: Enabled

### Output Structure

Results are saved to:
```
/path/to/results/CountGD-PairTally-Results/
├── CountGD-quantitative/          # Quantitative metrics
│   └── annotations/
│       └── results.json
└── CountGD-qualitative/           # Qualitative results with visualizations
    └── annotations/
        ├── detections/            # Per-image detection results
        ├── positive_qualitative_data.json
        ├── negative_qualitative_data.json
        └── complete_qualitative_data.json
```

### Expected Performance

CountGD performance on PairTally:
- **Overall MAE**: X.XX
- **Overall RMSE**: X.XX
- **Best performing category**: FOO (Food)
- **Most challenging category**: OTR (Other)

### Key Features Evaluated

1. **Multi-modal counting**: Text + visual exemplars
2. **Open-world capability**: Counting novel object classes
3. **Fine-grained distinction**: INTRA-class attribute differences
4. **Robustness**: Performance across different object categories

### Troubleshooting

**Common Issues:**
1. **CUDA out of memory**: Reduce batch size in evaluation script
2. **GroundingDINO compilation errors**: Ensure GCC 11+ is installed
3. **Missing weights**: Verify all checkpoint files are downloaded
4. **Path errors**: Update all dataset paths in evaluation scripts

**Performance Tips:**
- Use `--sam_tt_norm` for better accuracy (slower)
- Increase batch size for faster evaluation (if memory allows)
- Use `--crop` for better object localization

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
