# FamNet (Learning To Count Everything) Model Setup Instructions

## Prerequisites

You need to download the FamNet pretrained model before running evaluations.

## 1. Download FamNet Model

Visit the [Learning To Count Everything GitHub repository](https://github.com/cvlab-stonybrook/LearningToCountEverything) and download the pretrained model:

- **Model**: FamNet pretrained checkpoint
- **Download**: Follow the instructions in the repository to get the FSC147 pretrained weights

## 2. Setup Directory Structure

The FamNet model repository should be cloned directly into this folder:

```
models/learningtocount/
├── LearningToCountEverything/       # Clone repository here
│   ├── pretrained_models/          # Original FamNet model directory
│   │   └── [FamNet model files]   # Download model weights here
│   ├── [other FamNet files]
│   └── requirements.txt
├── evaluate_pairtally_count_both_classes.py
├── evaluate_pairtally_count_one_class.py
├── run_count_both_classes.sh
├── run_count_one_class.sh
└── SETUP.md                        # This file
```

## 3. Setup Steps

1. **Clone the FamNet repository into this directory**:
   ```bash
   cd models/learningtocount/
   git clone https://github.com/cvlab-stonybrook/LearningToCountEverything.git
   ```

2. **Create symlink for pretrained_models**:
   ```bash
   ln -s LearningToCountEverything/pretrained_models pretrained_models
   ```

3. **Download the FamNet model weights**:
   ```bash
   # Follow repository instructions to download pretrained weights
   # Place them in LearningToCountEverything/pretrained_models/
   # The exact download instructions are in the repository README
   ```

4. **Install FamNet dependencies**:
   ```bash
   cd LearningToCountEverything
   pip install -r requirements.txt
   cd ..
   ```

## 4. Install Dependencies

Make sure you have the FamNet dependencies installed:

```bash
# Install requirements
pip install -r LearningToCountEverything/requirements.txt

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

Based on the paper results, FamNet achieves:
- **MAE**: 88.30
- **RMSE**: 148.42

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
- **Import errors**: Install dependencies and ensure Python path is correct

## Directory Structure After Setup

```
models/learningtocount/
├── LearningToCountEverything/      # Cloned repository
│   ├── pretrained_models/         # Original FamNet structure
│   │   └── [FamNet model files]  # Downloaded model weights
│   └── [other FamNet files]       # Full FamNet codebase
├── pretrained_models/             # Symlink to LearningToCountEverything/pretrained_models
├── outputs/                       # Created automatically
└── [evaluation scripts]           # Already present
```
