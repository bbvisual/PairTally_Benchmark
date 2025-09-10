# 🎯 PairTally: Can AI Count What We Mean, Not What They See?

<p align="center">
  <img src="assets/banner.png" alt="PairTally Dataset Examples" width="100%">
</p>

<p align="center">
  <strong>A Benchmark for Fine-Grained Visual Counting in the Real World</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx">📄 Paper</a> •
  <a href="#-quick-start">🚀 Quick Start</a> •
  <a href="#-dataset">📊 Dataset</a> •
  <a href="#-leaderboard">🏆 Leaderboard</a> •
  <a href="#-demo">💻 Demo</a> •
  <a href="#-citation">📝 Citation</a>
</p>

---

## 🤔 The Challenge

Imagine you're asked to count the **red poker chips** in a pile that also contains blue ones. Or distinguish between **long silver screws** and **short bronze screws** scattered on a table. Simple for humans, right? 

**But can AI do it?**

Current vision models excel at detecting objects, but struggle when asked to count specific items based on subtle differences in:
- 🎨 **Color** (black vs. white game pieces)
- 📏 **Size** (large vs. small marbles) 
- 🔷 **Shape/Texture** (spiral vs. penne pasta)

## 🌟 Introducing PairTally

PairTally is the first benchmark specifically designed to test whether AI can truly understand *what* humans want to count, not just detect what's visible. Each of our **681 high-resolution images** contains two types of objects that require fine-grained discrimination.

### Why PairTally Matters

- **🎯 Real-world scenarios**: From inventory management to medical diagnostics
- **🔬 Systematic evaluation**: Controlled pairs of visually similar objects
- **📈 Reveals critical gaps**: Current SOTA models achieve only 53.07 MAE (humans: ~5 MAE)
- **🚀 Drives innovation**: Pushes the boundaries of visual understanding

## 📊 Dataset at a Glance

<table>
<tr>
<td width="50%">

### Key Statistics
- **681** high-resolution images
- **54** object categories
- **98** fine-grained subcategories
- **2** object types per image
- **~200** average objects per image

</td>
<td width="50%">

### Attribute Distribution
- 🎨 **43.5%** Color variations
- 🔷 **42.5%** Shape/Texture differences  
- 📏 **14.0%** Size distinctions

</td>
</tr>
</table>

### Dataset Structure

```
PairTally/
├── 📸 Images (681 files)
├── 📋 Annotations
│   ├── INTER-category (350 images) - Different object types
│   └── INTRA-category (331 images) - Same object, different attributes
└── 📓 Evaluation Tools
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/bbvisual/PairTally_Benchmark.git
cd PairTally_Benchmark

# Download the dataset
python download_dataset.py

# Verify installation
python verify_dataset.py
```

### Your First Count

```python
from pairtally import PairTallyDataset

# Load dataset
dataset = PairTallyDataset()

# Get a random image
image, annotation = dataset.get_random_sample()

# Display the challenge
dataset.visualize(image, annotation)
# Shows: "Count the red poker chips" vs "Count the blue poker chips"
```

## 💻 Demo

Try our interactive demo notebook to explore the dataset:

```bash
jupyter notebook PairTally_Demo_Notebook.ipynb
```

The notebook includes:
- 🎲 Random sample visualization
- 📊 Dataset statistics and distributions
- 🔍 Fine-grained attribute analysis
- 🏃 Model evaluation pipeline

## 🏆 Leaderboard

| Model | Type | Overall MAE ↓ | INTER MAE | INTRA MAE | Color NAE | Size NAE | Texture NAE |
|-------|------|--------------|-----------|-----------|-----------|----------|-------------|
| 🥇 **GeCo** | Exemplar | **53.07** | 45.05 | 54.80 | 0.791 | 1.345 | 0.946 |
| 🥈 CountGD | Hybrid | 57.33 | 39.78 | 56.54 | 0.856 | 1.402 | 0.793 |
| 🥉 LoCA | Exemplar | 62.78 | 71.89 | 57.45 | 0.799 | 1.244 | 1.007 |
| DAVE | Exemplar | 69.49 | 46.27 | 46.75 | 0.738 | 1.293 | 0.693 |
| FamNet | Exemplar | 88.30 | 66.97 | 74.75 | 1.296 | 1.859 | 1.448 |
| Qwen2.5-VL | VLM | 99.88 | 46.35 | 67.86 | - | - | - |
| Human* | - | ~5 | ~3 | ~7 | ~0.1 | ~0.2 | ~0.15 |

*Estimated human performance

## 🔍 Key Findings

Our benchmark reveals that current AI models:

1. **Struggle with fine-grained discrimination** - Often count all objects regardless of specified attributes
2. **Color is easiest, size is hardest** - Models achieve best performance on color variations (NAE: 0.74-0.79)
3. **VLMs underperform** - Large vision-language models lag behind specialized counting methods
4. **Room for improvement** - Best model (GeCo) still 10x worse than human performance

## 🛠️ Evaluation

### Evaluate Your Model

```python
from pairtally import evaluate_model

# Load your model
model = load_your_model()

# Run evaluation
results = evaluate_model(
    model=model,
    dataset=dataset,
    subset='all'  # or 'inter', 'intra'
)

print(f"MAE: {results['mae']:.2f}")
print(f"RMSE: {results['rmse']:.2f}")
print(f"NAE: {results['nae']:.2f}")
```

### Supported Model Types

- 📦 **Exemplar-based**: FamNet, DAVE, GeCo, LoCA
- 💬 **Language-prompted**: CountGD, LLMDet
- 🤖 **Vision-Language Models**: GPT-4V, Qwen-VL, LLaMA-Vision

## 📚 Resources

- 📄 **Paper**: [arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)
- 💾 **Dataset Download**: [Google Drive](https://drive.google.com/file/d/1TnenXS4yFicjo81NnmClfzgc8ltmmeBv/view)
- 📓 **Demo Notebook**: [PairTally_Demo.ipynb](./PairTally_Demo_Notebook.ipynb)
- 🏆 **Challenge**: [Submit your results](https://forms.gle/xxxxx)

## 📝 Citation

If you use PairTally in your research, please cite:

```bibtex
@inproceedings{nguyen2025pairtally,
  title={Can Current AI Models Count What We Mean, Not What They See? A Benchmark and Systematic Evaluation},
  author={Nguyen, Gia Khanh and Huang, Yifeng and Hoai, Minh},
  booktitle={Digital Image Computing: Techniques and Applications (DICTA)},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📧 Contact

- **Gia Khanh Nguyen**: [giakhanh.nguyen@adelaide.edu.au](mailto:giakhanh.nguyen@adelaide.edu.au)
- **Lab Website**: [Australian Institute for Machine Learning](https://www.adelaide.edu.au/aiml/)

## 🙏 Acknowledgments

We thank the Australian Institute for Machine Learning and Stony Brook University for their support in creating this benchmark.

---

<p align="center">
  <strong>🎯 Help us push the boundaries of visual counting!</strong><br>
  <a href="https://github.com/bbvisual/PairTally_Benchmark">⭐ Star</a> • 
  <a href="https://github.com/bbvisual/PairTally_Benchmark/fork">🍴 Fork</a> • 
  <a href="https://github.com/bbvisual/PairTally_Benchmark/issues">🐛 Issues</a>
</p>