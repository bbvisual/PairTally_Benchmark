# LOCA Model Setup Instructions

## Prerequisites

You need to download the LOCA pretrained model before running evaluations.

## 1. Download LOCA Model

Visit the [LOCA GitHub repository](https://github.com/djukicn/LOCA) and download the pretrained model:

- **Model**: LOCA pretrained checkpoint
- **Download**: Follow the instructions in the LOCA repository to get the model weights

## 2. Setup Directory Structure

The LOCA model repository should be cloned directly into this folder:

```
models/loca/
├── loca/                       # Clone repository here (lowercase name)
│   ├── pretrained_models/     # Original LOCA model directory
│   │   └── [LOCA model files] # Download model weights here
│   ├── [other LOCA files]
│   └── requirements.txt
├── pretrained_models/          # Symlink to loca/pretrained_models
├── evaluate_pairtally_count_both_classes.py
├── evaluate_pairtally_count_one_class.py
├── run_count_both_classes.sh
├── run_count_one_class.sh
└── SETUP.md                    # This file
```

## 3. Setup Steps

1. **Clone the LOCA repository into this directory**:
   ```bash
   cd models/loca/
   git clone https://github.com/djukicn/LOCA.git loca
   ```

2. **Create symlink for pretrained_models**:
   ```bash
   ln -s loca/pretrained_models pretrained_models
   ```

3. **Download the LOCA model weights**:
   ```bash
   # Follow LOCA repository instructions to download model weights
   # Place them in loca/pretrained_models/
   # The exact download instructions are in the LOCA repository README
   ```

4. **Install LOCA dependencies**:
   ```bash
   cd loca
   pip install -r requirements.txt
   cd ..
   ```

## 4. Install Dependencies

Make sure you have the LOCA dependencies installed:

```bash
# Install LOCA requirements
pip install -r loca/requirements.txt

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

Based on the paper results, LOCA achieves:
- **MAE**: 62.78
- **RMSE**: 136.76

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

- **Model not found**: Ensure model files are in `pretrained_models/`
- **CUDA errors**: Ensure GPU is available and CUDA is properly installed
- **Import errors**: Install LOCA dependencies and ensure Python path is correct

## Directory Structure After Setup

```
models/loca/
├── loca/                          # Cloned repository
│   ├── pretrained_models/        # Original LOCA structure
│   │   └── [LOCA model files]   # Downloaded model weights
│   └── [other LOCA files]        # Full LOCA codebase
├── pretrained_models/             # Symlink to loca/pretrained_models
├── outputs/                       # Created automatically
└── [evaluation scripts]           # Already present
```
