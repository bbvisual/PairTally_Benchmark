# CountGD Model Setup Instructions

## Prerequisites

You need to download the CountGD pretrained model and config files before running evaluations.

## 1. Download CountGD Model

Visit the [CountGD GitHub repository](https://github.com/Niki-Amini-Naieni/CountGD) and download the pretrained model:

- **Model**: FSC147 pretrained checkpoint
- **Download**: Follow the instructions in the CountGD repository to get `checkpoint_fsc147_best.pth`

## 2. Setup Directory Structure

The CountGD repository files should be extracted directly into this folder:

```
models/countgd/
├── checkpoints/                # CountGD model directory
│   └── checkpoint_fsc147_best.pth  # Download model weights here
├── config/                     # CountGD config files
│   └── cfg_fsc147_vit_b.py
├── pretrained_models/          # Symlink to checkpoints (for consistency)
├── [other CountGD files]       # All CountGD repository files at this level
├── requirements.txt            # CountGD requirements
├── evaluate_pairtally_count_both_classes.py  # Our evaluation scripts
├── evaluate_pairtally_count_one_class.py
├── evaluate_pairtally_count_both_classes_text_only.py
├── evaluate_pairtally_count_one_class_text_only.py
├── run_count_both_classes.sh
├── run_count_one_class.sh
├── run_count_one_class_text_only.sh
└── SETUP.md                    # This file
```

## 3. Setup Steps

1. **Clone and extract CountGD repository into this directory**:
   ```bash
   cd models/countgd/
   git clone https://github.com/Niki-Amini-Naieni/CountGD.git temp_countgd
   mv temp_countgd/* ./
   mv temp_countgd/.[^.]* ./ 2>/dev/null || true  # Move hidden files if any
   rm -rf temp_countgd
   ```

2. **Create symlink for pretrained_models**:
   ```bash
   ln -s checkpoints pretrained_models
   ```

3. **Download the CountGD model weights**:
   ```bash
   # Follow CountGD repository instructions to download checkpoint_fsc147_best.pth
   # Place it in checkpoints/
   # The exact download instructions are in the CountGD repository README
   ```

4. **Install CountGD dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 4. Install Dependencies

Make sure you have the CountGD dependencies installed:

```bash
# Install CountGD requirements
pip install -r /tmp/CountGD/requirements.txt

# Key dependencies include:
# - torch
# - torchvision  
# - transformers
# - opencv-python
# - pillow
```

## 5. Run Evaluation

Once setup is complete, you can run evaluations:

```bash
# Count both classes (recommended for paper results)
./run_count_both_classes.sh

# Count one class at a time
./run_count_one_class.sh

# Text-only evaluation (no exemplar boxes)
./run_count_one_class_text_only.sh
```

## Expected Model Performance

Based on the paper results, CountGD achieves:
- **MAE**: 57.33
- **RMSE**: 108.93

## Evaluation Modes

CountGD supports multiple evaluation modes:

1. **Count Both Classes** (`run_count_both_classes.sh`):
   - Uses 2 positive + 1 negative exemplars
   - Counts both object classes simultaneously
   - Main results reported in paper

2. **Count One Class** (`run_count_one_class.sh`):
   - Uses positive exemplars only
   - Counts one class at a time
   - Useful for per-class analysis

3. **Text-Only** (`run_count_one_class_text_only.sh`):
   - Uses only text descriptions (no bounding boxes)
   - Tests language-only counting capability

## Troubleshooting

- **Model not found**: Ensure `checkpoint_fsc147_best.pth` is in `checkpoints/`
- **Config errors**: Make sure `cfg_fsc147_vit_b.py` is in `config/`
- **CUDA errors**: Ensure GPU is available and CUDA is properly installed
- **Import errors**: Install CountGD dependencies

## Directory Structure After Setup

```
models/countgd/
├── checkpoints/                # ✅ CountGD model directory
│   └── checkpoint_fsc147_best.pth  # ✅ Downloaded model weights
├── config/                     # ✅ CountGD config files
│   └── cfg_fsc147_vit_b.py    # ✅ Config file
├── pretrained_models/          # ✅ Symlink to checkpoints
├── [other CountGD files]       # ✅ All CountGD repository files
├── requirements.txt            # ✅ CountGD requirements
├── outputs/                    # ✅ Created automatically
└── [evaluation scripts]        # ✅ Already present
```
