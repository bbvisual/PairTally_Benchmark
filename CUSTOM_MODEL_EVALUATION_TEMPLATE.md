# Custom Model Evaluation Template

This template shows how to quickly set up your own counting model for evaluation on the PairTally dataset.

## Quick Setup Guide

### 1. Model Interface Template

Create a wrapper for your model that follows this interface:

```python
class YourCountingModel:
    def __init__(self, model_path, device='cuda'):
        """Initialize your model"""
        self.device = device
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load your pre-trained model"""
        # Your model loading code here
        pass
    
    def predict(self, image, exemplars, prompt=None):
        """
        Predict counts for given image and exemplars
        
        Args:
            image: PIL Image or numpy array
            exemplars: List of bounding boxes [[x1,y1,x2,y2], ...]
            prompt: Optional text prompt
            
        Returns:
            count: Integer count prediction
        """
        # Your inference code here
        pass
```

### 2. Evaluation Script Template

Use this template to create your evaluation script:

```python
import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# Import your model
from your_model import YourCountingModel

def load_pairtally_data(annotation_file, image_dir):
    """Load PairTally dataset annotations"""
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def calculate_metrics(predictions, ground_truths):
    """Calculate MAE, RMSE, NAE metrics"""
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    mae = np.mean(np.abs(predictions - ground_truths))
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    nae = np.mean(np.abs(predictions - ground_truths) / (ground_truths + 1e-8))
    
    return {'MAE': mae, 'RMSE': rmse, 'NAE': nae}

def evaluate_model():
    # Configuration
    model_path = "path/to/your/model.pth"
    data_dir = "dataset/pairtally_dataset"
    annotation_file = f"{data_dir}/annotations/pairtally_annotations_simple.json"
    image_dir = f"{data_dir}/images"
    
    # Initialize model
    model = YourCountingModel(model_path)
    
    # Load dataset
    annotations = load_pairtally_data(annotation_file, image_dir)
    
    predictions = []
    ground_truths = []
    
    print("Running evaluation...")
    for image_name, annotation in tqdm(annotations.items()):
        # Load image
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)
        
        # Get exemplars (first 3 positive examples)
        exemplars = annotation['box_examples_coordinates'][:3]
        
        # Get ground truth count
        gt_count = len(annotation['points'])
        
        # Predict count
        pred_count = model.predict(image, exemplars)
        
        predictions.append(pred_count)
        ground_truths.append(gt_count)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truths)
    
    print("\nResults:")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"NAE: {metrics['NAE']:.3f}")
    
    return metrics

if __name__ == "__main__":
    evaluate_model()
```

### 3. Example: GeCo Model Implementation

Here's how the template looks implemented for GeCo:

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

class GeCoModel:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        # Load GeCo model architecture
        from geco_architecture import GeCo
        model = GeCo(backbone='resnet50')
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def predict(self, image, exemplars, prompt=None):
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Preprocess exemplars
        exemplar_boxes = torch.tensor(exemplars).float().to(self.device)
        
        with torch.no_grad():
            # GeCo forward pass
            outputs = self.model(image_tensor, exemplar_boxes)
            count = outputs['count'].item()
        
        return int(round(count))
```

## Step-by-Step Setup Instructions

### 1. Prepare Your Model
- Ensure your model can take an image and exemplar bounding boxes as input
- Implement the model interface shown above
- Test your model on a few sample images

### 2. Set Up Environment
```bash
# Create conda environment
conda create -n your_model python=3.8
conda activate your_model

# Install basic requirements
pip install torch torchvision pillow tqdm numpy

# Install your model-specific requirements
pip install your_additional_requirements
```

### 3. Download PairTally Dataset
```bash
# Download images from Google Drive
# Link: https://drive.google.com/file/d/1TnenXS4yFicjo81NnmClfzgc8ltmmeBv/view

# Extract and setup
unzip PairTally-Images-Only.zip
mv PairTally-Images-Only/* dataset/pairtally_dataset/images/

# Verify setup
cd dataset
python verify_dataset.py
```

### 4. Run Evaluation
```bash
python your_evaluation_script.py
```

## Dataset Format

### Annotation Structure
Each image annotation contains:
```json
{
  "image_name.jpg": {
    "points": [[x, y], ...],                          // Ground truth points
    "box_examples_coordinates": [                      // Exemplar bounding boxes
      [[x1,y1], [x2,y1], [x2,y2], [x1,y2]], ...
    ],
    "positive_prompt": "Description of target objects",
    "negative_prompt": "Description of distractor objects"
  }
}
```

### Input Formats
- **Image**: PIL Image or numpy array
- **Exemplars**: List of bounding boxes in format `[[x1,y1,x2,y2], ...]`
- **Ground Truth**: Number of target objects (length of `points` array)

## Evaluation Metrics

The benchmark uses three standard counting metrics:
- **MAE** (Mean Absolute Error): Average absolute difference between predictions and ground truth
- **RMSE** (Root Mean Square Error): Square root of mean squared differences
- **NAE** (Normalized Absolute Error): MAE normalized by ground truth counts
