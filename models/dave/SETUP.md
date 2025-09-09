# DAVE Model Setup Instructions

## Prerequisites

You need to download the DAVE pretrained model before running evaluations.

## 1. Download DAVE Model

Visit the [DAVE GitHub repository](https://github.com/Jingkang50/DAVE) and download the pretrained model:

- **Model**: DAVE 3-shot model
- **Download**: Follow the instructions in the DAVE repository to get `DAVE_3_shot.pth`

## 2. Setup Directory Structure

The DAVE repository files should be extracted directly into this folder:

```
models/dave/
├── MODEL_folder/               # DAVE model directory
│   └── DAVE_3_shot.pth        # Download model weights here
├── config/                     # DAVE config files
├── pretrained_models/          # Symlink to MODEL_folder
├── [other DAVE files]          # All DAVE repository files at this level
├── requirements.txt            # DAVE requirements
├── evaluate_pairtally_count_both_classes.py  # Our evaluation scripts
├── evaluate_pairtally_count_one_class.py
├── run_count_both_classes.sh
├── run_count_one_class.sh
└── SETUP.md                    # This file
```

## 3. Setup Steps

1. **Clone and extract DAVE repository into this directory**:
   ```bash
   cd models/dave/
   git clone https://github.com/Jingkang50/DAVE.git temp_dave
   mv temp_dave/* ./
   mv temp_dave/.[^.]* ./ 2>/dev/null || true  # Move hidden files if any
   rm -rf temp_dave
   ```

2. **Create symlink for pretrained_models**:
   ```bash
   ln -s MODEL_folder pretrained_models
   ```

3. **Download the DAVE model weights**:
   ```bash
   # Follow DAVE repository instructions to download DAVE_3_shot.pth
   # Place it in MODEL_folder/
   # The exact download instructions are in the DAVE repository README
   ```

4. **Install DAVE dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 4. Install Dependencies

Make sure you have the DAVE dependencies installed:

```bash
# Install DAVE requirements
pip install -r /tmp/DAVE/requirements.txt

# Or create a conda environment following DAVE instructions
```

## 5. Run Evaluation

Once setup is complete, you can run evaluations:

```bash
# Count both classes (recommended)
./run_count_both_classes.sh

# Count one class at a time
./run_count_one_class.sh
```

## Expected Model Performance

Based on the paper results, DAVE achieves:
- **MAE**: 69.49
- **RMSE**: 130.42

## Troubleshooting

- **Model not found**: Ensure `DAVE_3_shot.pth` is in `pretrained_models/`
- **Config errors**: Make sure config files are copied from DAVE repository
- **Import errors**: Install DAVE dependencies and ensure Python path is correct

## Directory Structure After Setup

```
models/dave/
├── MODEL_folder/               # ✅ DAVE model directory
│   └── DAVE_3_shot.pth        # ✅ Downloaded model weights
├── config/                     # ✅ DAVE config files
├── pretrained_models/          # ✅ Symlink to MODEL_folder
├── [other DAVE files]          # ✅ All DAVE repository files
├── requirements.txt            # ✅ DAVE requirements
├── outputs/                    # ✅ Created automatically
└── [evaluation scripts]        # ✅ Already present
```
