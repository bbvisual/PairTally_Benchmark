# LLMDet Model Environment Setup

This directory contains the environment setup and evaluation notebook for the LLMDet model in the PairTally Benchmark.

## Environment Setup

### Prerequisites
- Anaconda or Miniconda installed
- NVIDIA GPU with CUDA 12.1 support (for GPU acceleration)

### Installation Steps

1. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate fg_count
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Environment Details

The environment (`fg_count`) includes:
- Python 3.10.18
- PyTorch 2.3.0 with CUDA 12.1 support
- Essential ML libraries: transformers, datasets, accelerate, deepspeed
- Computer vision: opencv-python, pillow, torchvision
- Data science: pandas, numpy, matplotlib, seaborn
- Development tools: ipython, jupyter, debugpy

## Running the Evaluation

After setting up the environment, you can run the LLMDet evaluation:

1. **Ensure the environment is activated:**
   ```bash
   conda activate fg_count
   ```

2. **Start Jupyter and run the evaluation notebook:**
   ```bash
   jupyter notebook eval_on_llmdet.ipynb
   ```

3. **Execute all cells in the notebook** to perform the LLMDet model evaluation on the PairTally benchmark.

## Colab notebook
Alternatively, you can use this notebook here to run and visualise results https://colab.research.google.com/drive/1zDhdTx4LqV4swJppCzXPjIiFLcMccawY?usp=sharing.
