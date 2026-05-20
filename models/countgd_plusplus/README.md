# CountGD++ - Generalized Prompting for Open-World Counting

This directory contains the evaluation setup for CountGD++ (CVPR 2026) on the PairTally dataset.

## Original Paper
**CountGD++: Generalized Prompting for Open-World Counting**  
Niki Amini-Naieni & Andrew Zisserman  
CVPR 2026  
[[Paper]](https://arxiv.org/abs/2512.23351) [[Code]](https://github.com/niki-amini-naieni/CountGDPlusPlus)

## Setup Instructions

### 1. Environment Setup
```bash
conda create -n countgdplusplus python=3.10
conda activate countgdplusplus
conda install -c conda-forge gxx_linux-64 compilers libstdcxx-ng # ensure to install required compilers
pip install -r requirements.txt
export CC=/usr/bin/gcc-11 # this ensures that gcc 11 is being used for compilation
cd models/GroundingDINO/ops
python setup.py build install
python test.py # should result in 6 lines of * True
cd ../../../
```

### 3. Download Pre-trained Weights
```bash
# Create checkpoints directory
mkdir checkpoints

# Download BERT weights
python download_bert.py

# Download CountGD++ model weights
# Download from: https://drive.google.com/file/d/1j6N22TtKu2NVcKpgfrf-sJHGeLDqs9hs/view?usp=sharing
# Place as: checkpoints/countgd_plusplus.pth
gdown 1j6N22TtKu2NVcKpgfrf-sJHGeLDqs9hs
```

## PairTally Evaluation

### Evaluation Modes

**Positive Prompts Only**
- Provides **positive text and 3 positive exemplars** per image
- Less accurate setting as only provides information about what should be counted
- Run the following command:
```bash
python evaluate_countgd_plusplus_pos_prompts.py
```
- Expected Output
```
==================================================
EVALUATION RESULTS
==================================================
Model: CountGDPlusPlus
Total Images: 681
MAE:  46.41
RMSE:  69.52
NAE: 
==================================================
```

**Positive and Negative Prompts**
- Provides **positive text, 3 positive exemplars, negative text, and 3 negative exemplars** per image
- Most accurate setting as provides information about what to count AND what NOT to count
- Run the following command:
```bash
python evaluate_countgd_plusplus_pos_neg_prompts.py
```
- Expected Output
```
==================================================
EVALUATION RESULTS
==================================================
Model: CountGDPlusPlus
Total Images: 681
MAE: 35.27
RMSE: 60.85
NAE: 0.455
==================================================
```

**Dataset Structure Expected:**
```
../../dataset/pairtally_dataset/
├── annotations/
│   └── pairtally_annotations_simple.json
└── images/
    └── [image files]
```

### Output Structure

Results are saved to `../../results/` 

### Troubleshooting

See [here](https://github.com/bbvisual/PairTally_Benchmark/tree/main/models/countgd#troubleshooting)

### Citation

If you use CountGD++ in your research, please cite:

```bibtex
@InProceedings{AminiNaieni26,
  title={CountGD++: Generalized Prompting for Open-World Counting},
  author={Amini-Naieni, N. and Zisserman, A.},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
