# GeCo Model Setup Instructions

## Prerequisites

You need to download the GeCo pretrained model before running evaluations.

## 1. Download GeCo Model

Visit the [GeCo GitHub repository](https://github.com/JakubPelhan/GeCo) and download the pretrained model:

- **Model**: GeCo updated model
- **Download**: Follow the instructions in the GeCo repository to get `GeCo_updated.pth`

## 2. Setup Directory Structure

The GeCo repository should be cloned as a subdirectory:

```
models/geco/
├── GeCo/                       # Clone repository here
│   ├── pretrained_models/     # GeCo model directory
│   │   └── GeCo_updated.pth  # Download model weights here
│   ├── [other GeCo files]     # All GeCo repository files
│   └── requirements.txt       # GeCo requirements
├── pretrained_models/          # Symlink to GeCo/pretrained_models
├── evaluate_pairtally_count_both_classes.py  # Our evaluation scripts
├── evaluate_pairtally_count_one_class.py
├── run_count_both_classes.sh
├── run_count_one_class.sh
└── SETUP.md                    # This file
```

## 3. Setup Steps

1. **Clone the GeCo repository into this directory**:
   ```bash
   cd models/geco/
   git clone https://github.com/JakubPelhan/GeCo.git
   ```

2. **Create symlink for pretrained_models**:
   ```bash
   ln -s GeCo/pretrained_models pretrained_models
   ```

3. **Download the GeCo model weights**:
   ```bash
   # Follow GeCo repository instructions to download GeCo_updated.pth
   # Place it in GeCo/pretrained_models/
   # The exact download instructions are in the GeCo repository README
   ```

4. **Install GeCo dependencies**:
   ```bash
   cd GeCo
   pip install -r requirements.txt
   cd ..
   ```

## 4. Install Dependencies

Make sure you have the GeCo dependencies installed:

```bash
# Install GeCo requirements
pip install -r GeCo/requirements.txt

# Key dependencies include:
# - torch
# - torchvision
# - opencv-python
# - pillow
# - numpy
```

## 5. Run Evaluation

Once setup is complete, you can run evaluations:

```bash
# Count both classes (recommended for paper results)
./run_count_both_classes.sh

# Count one class at a time
./run_count_one_class.sh
```

## Expected Model Performance

Based on the paper results, GeCo achieves:
- **MAE**: 53.07 (Best overall performance)
- **RMSE**: 98.00 (Best overall performance)

## Evaluation Modes

1. **Count Both Classes** (`run_count_both_classes.sh`):
   - Uses 2 positive + 1 negative exemplars
   - Counts both object classes simultaneously
   - Main results reported in paper

2. **Count One Class** (`run_count_one_class.sh`):
   - Uses positive exemplars only
   - Counts one class at a time
   - Useful for per-class analysis

## Troubleshooting

- **Model not found**: Ensure `GeCo_updated.pth` is in `pretrained_models/`
- **CUDA errors**: Ensure GPU is available and CUDA is properly installed
- **Import errors**: Install GeCo dependencies and ensure Python path is correct
- **Memory errors**: GeCo may require significant GPU memory

## Directory Structure After Setup

```
models/geco/
├── GeCo/                           # Cloned repository
│   ├── pretrained_models/         # Original GeCo structure
│   │   └── GeCo_updated.pth      # Downloaded model weights
│   └── [other GeCo files]         # Full GeCo codebase
├── pretrained_models/             # Symlink to GeCo/pretrained_models
├── outputs/                       # Created automatically
└── [evaluation scripts]           # Already present
```
