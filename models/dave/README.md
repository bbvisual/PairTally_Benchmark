# DAVE - A Detect-and-Verify Paradigm for Low-Shot Counting

This directory contains the evaluation setup for DAVE (CVPR 2024) on the DICTA25 dataset.

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

- `evaluate_DICTA25_combined.py` - Main evaluation script for PairTally dataset
- `evaluate_DICTA25_custom.py` - Custom evaluation with specific configurations
- `run_combined_eval.sh` - Shell script to run the evaluation
- `README.md` - This file

### Running Evaluation

1. **Copy evaluation scripts to DAVE directory:**
```bash
cp evaluate_DICTA25_*.py /path/to/DAVE/
cp run_combined_eval.sh /path/to/DAVE/
```

2. **Update paths in the evaluation scripts:**
Edit the scripts to point to your PairTally dataset location:
```python
base_data_path = "/path/to/PairTally-Benchmark-Release/dataset"
```

3. **Configure model paths:**
Update `utils/argparser.py` to point to your model and dataset paths.

4. **Run evaluation:**
```bash
cd /path/to/DAVE
conda activate dave
./run_combined_eval.sh
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

### Output Structure

Results are saved to:
```
/path/to/results/DAVE-PairTally-Results/
├── DAVE-quantitative/             # Quantitative metrics
│   └── annotations/
│       └── results.json
└── DAVE-qualitative/              # Qualitative results with visualizations
    └── annotations/
        ├── detections/            # Per-image detection results
        ├── positive_qualitative_data.json
        ├── negative_qualitative_data.json
        └── complete_qualitative_data.json
```

### Expected Performance

DAVE performance on PairTally:
- **Overall MAE**: X.XX
- **Overall RMSE**: X.XX  
- **Best performing category**: HOU (Household)
- **Most challenging category**: FUN (Fun/Games)

### Key Features Evaluated

1. **Detect-and-verify**: Two-stage counting approach
2. **Few-shot learning**: Learning from minimal exemplars
3. **False positive reduction**: Verification stage effectiveness
4. **Detection quality**: Bounding box accuracy and localization

### Demo Usage

Test DAVE on custom images:
```bash
# Few-shot demo
python demo.py --skip_train --model_name DAVE_3_shot --model_path material \
    --backbone resnet50 --swav_backbone --reduction 8 --num_enc_layers 3 \
    --num_dec_layers 3 --kernel_dim 3 --emb_dim 256 --num_objects 3 \
    --use_query_pos_emb --use_objectness --use_appearance --batch_size 1 --pre_norm

# Zero-shot demo  
python demo_zero.py --img_path <input-file> --show --zero_shot --two_passes \
    --skip_train --model_name DAVE_0_shot --model_path material \
    --backbone resnet50 --swav_backbone --use_objectness --use_appearance --pre_norm
```

### Troubleshooting

**Common Issues:**
1. **Missing detectron2**: Install with pip install 'git+https://github.com/facebookresearch/detectron2.git'
2. **CUDA compatibility**: Ensure PyTorch CUDA version matches your system
3. **Model weights**: Verify pre-trained models are in material/ directory
4. **Path errors**: Update dataset and model paths in configuration files

**Performance Tips:**
- Use `--use_objectness` for better detection quality
- Enable `--use_appearance` for appearance-based verification
- Adjust `--num_objects` based on expected object count

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
