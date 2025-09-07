# GeCo - A Novel Unified Architecture for Low-Shot Counting

This directory contains the evaluation setup for GeCo (NeurIPS 2024) on the DICTA25 dataset.

## Original Paper
**A Novel Unified Architecture for Low-Shot Counting by Detection and Segmentation**  
Jer Pelhan, Alan Lukezic, Vitjan Zavrtanik, Matej Kristan  
NeurIPS 2024  
[[Paper]](https://arxiv.org/pdf/2409.18686) [[Code]](https://github.com/jerpelhan/GeCo)

## Setup Instructions

### 1. Clone Original Repository
```bash
# Clone the official GeCo repository
git clone https://github.com/jerpelhan/GeCo.git
cd GeCo
```

### 2. Environment Setup
```bash
# Create conda environment
conda create -n geco_test python=3.8
conda activate geco_test

# Install PyTorch and dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib tqdm pycocotools scipy

# Install detectron2 for evaluation
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 3. Download Pre-trained Weights
```bash
# Download pretrained weights from:
# https://drive.google.com/file/d/1wjOF9MWkrVJVo5uG3gVqZEW9pwRq_aIk/view?usp=sharing

# Place in MODEL_folder/
mkdir -p MODEL_folder
# Download and place: MODEL_folder/model_weights.pth
```

### 4. Download FSC147 Dataset (for training data)
```bash
# Download FSC147 dataset from:
# https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing
# Place in DATA_folder/

# Download box annotations from:
# https://drive.google.com/drive/folders/1Jvr2Bu2cD_yn4W_DjKIW6YjdAiUsw_WA
# Place in DATA_folder/annotations/

# Generate density maps
python utils/data.py --data_path DATA_folder
```

## DICTA25 Evaluation

### Files in this Directory

- `evaluate_DICTA25_combined.py` - Main evaluation script for DICTA25 dataset
- `run_combined_eval.sh` - Shell script to run the evaluation
- `README.md` - This file

### Running Evaluation

1. **Copy evaluation scripts to GeCo directory:**
```bash
cp evaluate_DICTA25_combined.py /path/to/GeCo/
cp run_combined_eval.sh /path/to/GeCo/
```

2. **Update paths in the evaluation scripts:**
Edit the scripts to point to your DICTA25 dataset location:
```python
base_data_path = "/path/to/DICTA25-Can-AI-Models-Count-Release/dataset"
```

3. **Run evaluation:**
```bash
cd /path/to/GeCo
conda activate geco_test
./run_combined_eval.sh
```

### Model Architecture

GeCo features a unified architecture that combines:
1. **Dense Object Queries**: Robust prototype generalization across object appearances
2. **Detection Branch**: Object localization with bounding boxes
3. **Segmentation Branch**: Precise object segmentation masks
4. **Counting Loss**: Direct optimization of the detection task

### Quick Demo

Test GeCo on a single image:
```bash
# Run demo with mask output
python demo.py --image_path ./material/4.jpg --output_masks
```

### Evaluation on FSC147

For comparison with other methods:
```bash
# Run inference on FSC147
python evaluate.py --data_path DATA_folder --model_path MODEL_folder

# Evaluate bounding boxes
python evaluate_bboxes.py --data_path DATA_folder
```

### Output Structure

Results are saved to:
```
/path/to/results/GeCo-DICTA25-Results/
├── GeCo-quantitative/             # Quantitative metrics
│   └── annotations/
│       └── results.json
└── GeCo-qualitative/              # Qualitative results with visualizations
    └── annotations/
        ├── detections/            # Per-image detection results
        ├── masks/                 # Segmentation masks (if enabled)
        ├── positive_qualitative_data.json
        ├── negative_qualitative_data.json
        └── complete_qualitative_data.json
```

### Expected Performance

GeCo performance on DICTA25:
- **Overall MAE**: X.XX
- **Overall RMSE**: X.XX
- **Best performing category**: OFF (Office)
- **Most challenging category**: FOO (Food)

### Key Features Evaluated

1. **Unified architecture**: Joint detection, segmentation, and counting
2. **Dense object queries**: Robust prototype formulation
3. **Direct counting loss**: Optimized for detection task
4. **Segmentation capability**: Precise object boundaries
5. **Low-shot learning**: Few-shot and zero-shot counting

### Training (Optional)

To train GeCo from scratch:

1. **Download additional data:**
```bash
# Download train split box annotations
# https://drive.google.com/file/d/15_qpEZ7f0ZBrcTmgFnxx71lCdxAGtuTz/view?usp=sharing

# Download SAM-HQ weights
# https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?usp=sharing
```

2. **Generate density maps:**
```bash
python utils/data.py
```

3. **Run pretraining:**
```bash
sbatch pretrain.sh
```

4. **Run main training:**
```bash
sbatch train.sh
```

### Troubleshooting

**Common Issues:**
1. **Missing detectron2**: Install with the provided pip command
2. **CUDA compatibility**: Ensure PyTorch CUDA version matches system
3. **Missing weights**: Verify model weights are downloaded and placed correctly
4. **Data format**: Ensure FSC147 data is properly formatted and density maps generated

**Performance Tips:**
- Use `--output_masks` for segmentation visualization
- Adjust batch size based on GPU memory
- Use multiple GPUs for faster training (if available)

### Citation

If you use GeCo in your research, please cite:

```bibtex
@article{pelhan2024novel,
  title={A Novel Unified Architecture for Low-Shot Counting by Detection and Segmentation},
  author={Pelhan, Jer and Lukezic, Alan and Zavrtanik, Vitjan and Kristan, Matej},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={66260--66282},
  year={2024}
}
```
