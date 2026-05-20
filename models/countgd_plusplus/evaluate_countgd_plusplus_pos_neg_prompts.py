"""
CountGD++ Evaluation on PairTally
"""

import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import argparse, glob, logging
from typing import Dict, List, Any, Tuple
from util.slconfig import SLConfig, DictAction
from util.misc import nested_tensor_from_tensor_list
import datasets.transforms_app as T
import scipy.ndimage as ndimage
import io
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


CONF_THRESH = 0.23
NUM_SHOTS = 3 # 3 positive and 3 negative exemplars provided

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Script to test CountGD++ on PairTally")
    
    # Model parameters
    p.add_argument(
        "--device", default="cuda", help="device to use for inference"
    )
    p.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )

    # dataset parameters
    p.add_argument("--remove_difficult", action="store_true")
    p.add_argument("--fix_size", action="store_true")

    # training parameters
    p.add_argument("--note", default="", help="add some notes to the experiment")
    p.add_argument("--resume", default="", help="resume from checkpoint")
    p.add_argument("--finetune_ignore", type=str, nargs="+")
    p.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    p.add_argument("--eval", action="store_false")
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--test", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--find_unused_params", action="store_true")
    p.add_argument("--save_results", action="store_true")
    p.add_argument("--save_log", action="store_true")

    # distributed training parameters
    p.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    p.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    p.add_argument(
        "--rank", default=0, type=int, help="number of distributed processes"
    )
    p.add_argument(
        "--local_rank", type=int, help="local rank for DistributedDataParallel"
    )
    p.add_argument(
        "--local-rank", type=int, help="local rank for DistributedDataParallel"
    )
    p.add_argument("--amp", action="store_true", help="Train with mixed precision")
    return p

def get_boxes_from_prediction(model_output):
    input_ids = model_output["input_ids"][0]
    logits = model_output["pred_logits"].sigmoid()[0][:, :]
    boxes = model_output["pred_boxes"][0]

    # [pos_neg_split_idx] is the index of the first occurence of the "." separating token.
    for idx in range(len(input_ids)):
        token = input_ids[idx]
        if token == 1012:
            pos_neg_split_idx = idx
            break
    
    pos_logits = logits[:, :(pos_neg_split_idx + 1)]
    neg_logits = logits[:, (pos_neg_split_idx + 1):]

    # Stage 1 filtering:
    box_mask = pos_logits.max(dim=-1).values > CONF_THRESH
    boxes = boxes[box_mask, :]
    logits = logits[box_mask, :]

    # Stage 2 filtering:
    pos_logits = pos_logits[box_mask, :]
    neg_logits = neg_logits[box_mask, :]
    box_mask = pos_logits.max(dim=-1).values > neg_logits.max(dim=-1).values
    boxes = boxes[box_mask, :].cpu().numpy()
    logits = logits[box_mask, :].cpu().numpy().max(axis=-1)

    return boxes, logits

class CountGDPlusPlus:
    """
    CountGD++ model class implementation for PairTally
    """
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        
        # CountGD++ image preprocessing
        normalize = T.Compose(
            [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                normalize,
            ]
        )
    
    def load_model(self, model_path):
        """Load CountGD++ model"""

        p = build_arg_parser()
        args = p.parse_args()
        args.pretrain_model_path = model_path
        cfg = SLConfig.fromfile("cfg_app.py")
        cfg.merge_from_dict({"text_encoder_type": "checkpoints/bert-base-uncased"})
        cfg_dict = cfg._cfg_dict.to_dict()
        args_vars = vars(args)
        for k, v in cfg_dict.items():
            if k not in args_vars:
                setattr(args, k, v)
            else:
                raise ValueError("Key {} can used by args only".format(k))

        # fix the seed for reproducibility
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # we use register to maintain models from catdet6 on.
        from models.GroundingDINO import groundingdino_app

        build_func = groundingdino_app.build_groundingdino
        model, _, _ = build_func(args)

        checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
        model.load_state_dict(checkpoint, strict=False)

        model.eval().to(self.device)

        return model

    def preprocess_inputs(self, image, exemplars):
        """
        Preprocess image and exemplars for CountGD++
        """
        # exemplars come in format [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
        # reformatted to [[x1, y1, x2, y2], [x1, y1, x2, y2], [x1, y1, x2, y2], [x1, y1, x2, y2]]
        reformatted_exemplars = [[exemp[0][0], exemp[0][1], exemp[2][0], exemp[2][1]] for exemp in exemplars]
        image_tensor, exemplar_boxes = self.transform(image, {"exemplars": torch.tensor(reformatted_exemplars, dtype=torch.float)})
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        exemplar_boxes = [exemplar_boxes["exemplars"].to(self.device)]
        return image_tensor, exemplar_boxes
    
    def predict(self, image, pos_exemplars, neg_exemplars, pos_prompt, neg_prompt):
        """
        Predict count for given image and positive and negative text and exemplar prompts
        
        Args:
            image: PIL Image
            pos_exemplars: List of bounding boxes corresponding to the object to count
            neg_exemplars: List of bounding boxes corresponding to the object to NOT count 
            pos_prompt: Text prompt corresponding to the object to count
            neg_prompt: Text prompt corresponding to the object to NOT count
            
        Returns:
            count: Integer count prediction
        """
        # Preprocess inputs
        image_tensor, pos_exemplar_boxes = self.preprocess_inputs(image, pos_exemplars)
        neg_image_tensor, neg_exemplar_boxes = self.preprocess_inputs(image, neg_exemplars)

        caption = pos_prompt + " . " + neg_prompt + " ."
        
        with torch.no_grad():
            model_output = self.model(
                nested_tensor_from_tensor_list(image_tensor),
                nested_tensor_from_tensor_list(image_tensor),
                pos_exemplar_boxes,
                [nested_tensor_from_tensor_list(neg_image_tensor)],
                [neg_exemplar_boxes],
                captions=[caption],
            )

        pred_boxes, pred_logits = get_boxes_from_prediction(model_output)
        count = pred_boxes.shape[0]
        
        return count, pred_boxes


def load_pairtally_data(annotation_file, image_dir):
    """Load PairTally dataset annotations"""
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Verify image directory
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    return annotations


def calculate_metrics(predictions, ground_truths):
    """Calculate MAE, RMSE, NAE metrics"""
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    mae = np.mean(np.abs(predictions - ground_truths))
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    nae = np.mean(np.abs(predictions - ground_truths) / (ground_truths + 1e-8))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'NAE': nae,
        'predictions': predictions.tolist(),
        'ground_truths': ground_truths.tolist()
    }

def get_box_coords_from_boxes(image, boxes):
    """
    Get box coordinates in the format (x, y, box_w, box_h) such that (x, y) is the top left of the box with width [box_w] and height [box_h] with all coordinates in the image coordinate system
    """
    (w, h) = image.size
    center_x = w * boxes[:, 0]
    center_y = h * boxes[:, 1]
    box_w = w * boxes[:, 2]
    box_h = h * boxes[:, 3]
    (x, y) = center_x - box_w/2, center_y - box_h/2
    return (x, y, box_w, box_h)

def draw_exemplars(image, pos_exemplars=None, neg_exemplars=None):
    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    plt.axis('off')

    # Reformat exemplars
    pos_exemplar_boxes = []
    if pos_exemplars is not None:
        for exemp in pos_exemplars:
            x1, y1, x2, y2 = exemp[0][0], exemp[0][1], exemp[2][0], exemp[2][1]
            reformatted_exemp = [x1, y1, (x2 - x1), (y2 - y1)]
            pos_exemplar_boxes.append(reformatted_exemp)

    neg_exemplar_boxes = []
    if neg_exemplars is not None:
        for exemp in neg_exemplars:
            x1, y1, x2, y2 = exemp[0][0], exemp[0][1], exemp[2][0], exemp[2][1]
            reformatted_exemp = [x1, y1, (x2 - x1), (y2 - y1)]
            neg_exemplar_boxes.append(reformatted_exemp)

    # Plot exemplars
    for box_ind in range(len(pos_exemplar_boxes)):
        (x_i, y_i, box_w_i, box_h_i) = pos_exemplar_boxes[box_ind][0], pos_exemplar_boxes[box_ind][1], pos_exemplar_boxes[box_ind][2], pos_exemplar_boxes[box_ind][3]
        
        rect_border = patches.Rectangle(
            (x_i, y_i), box_w_i, box_h_i,
            linewidth=1, edgecolor='green', facecolor='none'
        )
        ax.add_patch(rect_border)
             
    for box_ind in range(len(neg_exemplar_boxes)):
        (x_i, y_i, box_w_i, box_h_i) = neg_exemplar_boxes[box_ind][0], neg_exemplar_boxes[box_ind][1], neg_exemplar_boxes[box_ind][2], neg_exemplar_boxes[box_ind][3]
        
        rect_border = patches.Rectangle(
            (x_i, y_i), box_w_i, box_h_i,
            linewidth=1, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect_border)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    plt.close()

    output_img = Image.open(img_buf)
    return output_img

def draw_boxes(image, boxes, scores=None):
    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    plt.axis('off')

    # Plot bounding boxes
    (x0, y0, box_w, box_h) = get_box_coords_from_boxes(image, boxes)

    for box_ind in range(boxes.shape[0]):
        (x_i, y_i, box_w_i, box_h_i) = (x0[box_ind], y0[box_ind], box_w[box_ind], box_h[box_ind])
        
        rect_border = patches.Rectangle(
            (x_i, y_i), box_w_i, box_h_i,
            linewidth=1, edgecolor='cyan', facecolor='none'
        )
        ax.add_patch(rect_border)
    
        # label (score)
        scores = None
        if scores is not None:
            s = float(scores[box_ind])
            ax.text(
                x_i, y_i, f"{s:.2f}",              # text position and content
                fontsize=12, color="white",
                bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"),
                ha="left", va="top",               # top-left corner of box
                zorder=5, clip_on=True
            )

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    plt.close()

    output_img = Image.open(img_buf)
    return output_img

def evaluate_model(config):
    """Main evaluation function"""
    print("Initializing model...")
    model = CountGDPlusPlus(config['model_path'], config['device'])
    
    print("Loading dataset...")
    annotations = load_pairtally_data(config['annotation_file'], config['image_dir'])
    
    predictions = []
    ground_truths = []
    results_per_image = []
    
    print(f"Evaluating on {len(annotations)} images...")

    for image_name, annotation in tqdm(annotations.items()):
        try:
            # Load image
            image_path = os.path.join(config['image_dir'], image_name)
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
                
            image = Image.open(image_path).convert('RGB')
            
            # Get exemplars (use first 3 positive examples)
            pos_exemplars = annotation['box_examples_coordinates'][:NUM_SHOTS]

            neg_exemplars = annotation['negative_box_exemples_coordinates'][:NUM_SHOTS]
            
            # Get ground truth count
            gt_count = len(annotation['points'])
            
            # Get text prompt if available
            pos_prompt = annotation['positive_prompt']

            neg_prompt = annotation['negative_prompt']
            
            # Predict count
            pred_count, boxes = model.predict(image, pos_exemplars, neg_exemplars, pos_prompt, neg_prompt)
            
            # Print results for monitoring. Comment below lines out for no printing.
            print(image_name)
            print("pos prompt: " + pos_prompt)
            print("neg prompt: " + neg_prompt)
            print("GT: " + str(gt_count) + ", Pred: " + str(pred_count))

            # Save results.
            predictions.append(pred_count)
            ground_truths.append(gt_count)
            
            # Store per-image results
            results_per_image.append({
                'image_name': image_name,
                'predicted_count': pred_count,
                'ground_truth_count': gt_count,
                'absolute_error': abs(pred_count - gt_count),
                'positive_prompt': pos_prompt
            })
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            continue
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truths)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {config.get('model_name', 'Custom Model')}")
    print(f"Total Images: {len(predictions)}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"NAE: {metrics['NAE']:.3f}")
    print("="*50)
    
    # Save detailed results
    if config.get('save_results', True):
        output_dir = config.get('output_dir', 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'model_name': config.get('model_name', 'Custom Model'),
            'metrics': metrics,
            'per_image_results': results_per_image,
            'config': config
        }
        
        output_file = os.path.join(output_dir, 'evaluation_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Detailed results saved to: {output_file}")
    
    return metrics


def main():
    """Main function with configuration"""
    
    # TODO: Modify these paths for your setup
    config = {
        # Model configuration
        'model_name': 'CountGDPlusPlus',
        'model_path': 'checkpoints/countgd_plusplus.pth',  
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Dataset configuration
        'data_dir': '../../dataset/pairtally_dataset',
        'annotation_file': '../../dataset/pairtally_dataset/annotations/pairtally_annotations_simple.json',
        'image_dir': '../../dataset/pairtally_dataset/images',
        
        # Output configuration
        'save_results': True,
        'output_dir': '../../results/countgd_plusplus',
    }
    
    # Verify paths exist
    if not os.path.exists(config['annotation_file']):
        raise FileNotFoundError(f"Annotation file not found: {config['annotation_file']}")
    
    if not os.path.exists(config['image_dir']):
        raise FileNotFoundError(f"Image directory not found: {config['image_dir']}")
    
    # Run evaluation
    metrics = evaluate_model(config)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    print("CountGD++ PairTally evaluation (given positive text, 3 positive exemplars, negative text, and 3 negative exemplars)")
    
    # Run evaluation
    try:
        metrics = main()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease check:")
        print("1. Your model implementation in YourCountingModel class")
        print("2. Dataset paths in config")
        print("3. Model file exists and is accessible")
