import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'CountGD'))

import glob
import torch
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
import argparse
import json
import os
import pickle
# Visualization imports removed - only saving results dictionaries
import scipy.ndimage as ndimage
from torch.utils.data import Dataset, DataLoader
import time
import cv2

from CountGD.util.slconfig import SLConfig, DictAction
from CountGD.util.misc import nested_tensor_from_tensor_list
import CountGD.datasets_inference.transforms as T


# Custom Dataset class for PairTally data (count both classes evaluation)
class PairTallyCountBothClassesDataset(Dataset):
    def __init__(self, annotations_file, images_folder, transform=None):
        self.annotations_file = annotations_file
        self.images_folder = images_folder
        self.transform = transform
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_files = list(self.annotations.keys())
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        annotation = self.annotations[image_name]
        
        # Load image
        image_path = os.path.join(self.images_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        
        # Get combined ground truth counts (positive + negative)
        pos_gt_count = len(annotation['points'])
        neg_gt_count = len(annotation['negative_points'])
        combined_gt_count = pos_gt_count + neg_gt_count
        
        # Get exemplar bounding boxes (2 positive + 1 negative)
        # Note: The format is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] - convert to [x1, y1, x2, y2]
        positive_box_coords = annotation['box_examples_coordinates'][:2]  # Take first 2 positive
        negative_box_coords = annotation['negative_box_exemples_coordinates'][:1]  # Take first 1 negative
        
        # Convert positive exemplars from FSC147 format to standard box format
        positive_exemplars = []
        for box_coords in positive_box_coords:
            x1, y1 = box_coords[0]
            x2, y2 = box_coords[2]
            positive_exemplars.append([x1, y1, x2, y2])
        
        # Convert negative exemplars from FSC147 format to standard box format
        negative_exemplars = []
        for box_coords in negative_box_coords:
            x1, y1 = box_coords[0]
            x2, y2 = box_coords[2]
            negative_exemplars.append([x1, y1, x2, y2])
        
        # Extract class names from filename for combined prompt
        # Format: {positive_class}_{negative_class}_{...}.jpg
        filename_parts = image_name.split('_')
        pos_class = filename_parts[0].replace('-', ' ')  # Convert "5-cents-coin" to "5 cents coin"
        neg_class = filename_parts[1].replace('-', ' ')  # Convert "10-cents-coin" to "10 cents coin"
        
        # Create combined prompt
        combined_prompt = f"{pos_class} and {neg_class}"
        
        return {
            'image': image,
            'image_name': image_name,
            'pos_gt_count': pos_gt_count,
            'neg_gt_count': neg_gt_count,
            'combined_gt_count': combined_gt_count,
            'positive_exemplars': positive_exemplars,
            'negative_exemplars': negative_exemplars,
            'combined_prompt': combined_prompt,
            'annotation': annotation
        }


def get_args_parser():
    parser = argparse.ArgumentParser("PairTally CountGD Count Both Classes Evaluation", add_help=False)
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )

    # dataset parameters
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")

    # training parameters
    parser.add_argument("--note", default="", help="add some notes to the experiment")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--pretrain_model_path",
        help="load from other checkpoint",
        default="./checkpoints/checkpoint_fsc147_best.pth",
    )
    parser.add_argument(
        "--config",
        help="config file",
        default="./config/cfg_fsc147_vit_b.py",
    )
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--modelname", default="groundingdino", help="model name")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_path", default="./outputs")
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--confidence_thresh", default=0.3, type=float)
    # Removed --save_visualizations argument - only saving results dictionaries
    parser.add_argument("--output_limit", type=int, default=None, help="Limit processing to N images (for testing)")

    return parser


def build_dataset(image_set, args):
    if image_set == "val":
        dataset = PairTallyCountBothClassesDataset(
            annotations_file=args.annotation_file,
            images_folder=args.image_folder,
            transform=make_transforms(image_set)
        )
    return dataset


def make_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def load_model(args):
    # Get config
    cfg = SLConfig.fromfile(args.config)
    cfg_dict = cfg._cfg_dict.copy()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build the data transform
    transforms = make_transforms("val")

    from CountGD.models.registry import MODULE_BUILD_FUNCS

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, _, _ = build_func(args)

    model.to(device)

    checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
    model.load_state_dict(checkpoint, strict=False)

    model.eval()

    return model, transforms


def run_combined_inference(model, image, positive_exemplars, negative_exemplars, combined_prompt, transform, args):
    """Run inference with 2 positive + 1 negative exemplars and combined prompt"""
    
    # Prepare exemplars (2 positive + 1 negative)
    exemplar_boxes = []
    
    # Add positive exemplars
    for box in positive_exemplars:
        x1, y1, x2, y2 = box
        exemplar_boxes.append([x1, y1, x2, y2])
    
    # Add negative exemplars  
    for box in negative_exemplars:
        x1, y1, x2, y2 = box
        exemplar_boxes.append([x1, y1, x2, y2])
    
    # Transform image and exemplars (following reference implementation)
    input_image, target = transform(image, {"exemplars": torch.tensor(exemplar_boxes, dtype=torch.float32)})
    input_image = input_image.cuda()
    input_exemplar = target["exemplars"].cuda()
    
    print(f"    Combined inference with {len(positive_exemplars)} positive + {len(negative_exemplars)} negative exemplars")
    print(f"    Using combined prompt: '{combined_prompt}'")
    print(f"    Exemplars shape: {input_exemplar.shape}, using {len(exemplar_boxes)} exemplars")
    
    with torch.no_grad():
        model_output = model(
            input_image.unsqueeze(0),
            [input_exemplar],
            [torch.tensor([0]).cuda()],
            captions=[combined_prompt + " ."],
        )
    
    logits = model_output["pred_logits"][0].sigmoid()
    boxes = model_output["pred_boxes"][0]
    
    # Apply confidence threshold
    box_mask = logits.max(dim=-1).values > args.confidence_thresh
    logits = logits[box_mask, :]
    boxes = boxes[box_mask, :]
    
    return boxes, logits


# Visualization function removed - only saving results dictionaries


def main(args):
    print("Loading model...")
    model, data_transform = load_model(args)
    
    print(f"Loading dataset from: {args.annotation_file}")
    dataset = PairTallyCountBothClassesDataset(args.annotation_file, args.image_folder, data_transform)
    
    # Create output directories following PairTally-Results pattern
    dataset_folder_name = args.dataset_name
    
    # Use PairTally-Results structure with "combined" identifier to avoid overwriting
    quant_output_dir = "../../results/CountGD-count-both-classes-quantitative"
    dataset_quant_dir = os.path.join(quant_output_dir, dataset_folder_name)
    
    qual_output_dir = "../../results/CountGD-count-both-classes-qualitative"
    dataset_pred_dir = os.path.join(qual_output_dir, dataset_folder_name)
    
    # Visualization directory creation removed - only saving results dictionaries
    
    os.makedirs(dataset_pred_dir, exist_ok=True)
    os.makedirs(dataset_quant_dir, exist_ok=True)
    
    print(f"Saving combined qualitative data to: {dataset_pred_dir}")
    print(f"Saving combined quantitative data to: {dataset_quant_dir}")
    
    model_name = "CountGD-Combined"
    
    print(f"Running combined evaluation on {len(dataset)} images...")
    print(f"Model: {model_name}")
    print(f"Confidence threshold: {args.confidence_thresh}")
    
    all_results = {
        'model_name': model_name,
        'dataset_name': dataset_folder_name,
        'evaluation_info': {
            'confidence_thresh': args.confidence_thresh,
            'total_images': len(dataset),
            'exemplars_used': '2_positive_1_negative',
            'prompt_type': 'combined_positive_and_negative'
        },
        'image_results': {}
    }
    
    mae_sum = 0
    se_sum = 0
    total_count = 0
    
    for i, sample in enumerate(dataset):
        # Check output limit
        if args.output_limit is not None and i >= args.output_limit:
            print(f"Stopping after {i} images (output_limit={args.output_limit})...")
            break
            
        image = sample['image']
        image_name = sample['image_name']
        combined_gt_count = sample['combined_gt_count']
        pos_exemplars = sample['positive_exemplars']
        neg_exemplars = sample['negative_exemplars']
        combined_prompt = sample['combined_prompt']
        annotation = sample['annotation']
        
        print(f"\nProcessing {i+1}/{len(dataset)}: {image_name}")
        print(f"  Combined GT count: {combined_gt_count} (pos: {sample['pos_gt_count']}, neg: {sample['neg_gt_count']})")
        print(f"  Combined prompt: '{combined_prompt}'")
        
        # Run combined inference
        pred_boxes, pred_logits = run_combined_inference(
            model, image, pos_exemplars, neg_exemplars, combined_prompt, data_transform, args
        )
        
        pred_count = len(pred_boxes)
        mae = abs(pred_count - combined_gt_count)
        se = (pred_count - combined_gt_count) ** 2
        
        mae_sum += mae
        se_sum += se
        total_count += 1
        
        print(f"  Predicted: {pred_count}, MAE: {mae:.2f}")
        
        # Store results
        all_results['image_results'][image_name] = {
            'predicted_count': int(pred_count),
            'combined_gt_count': int(combined_gt_count),
            'pos_gt_count': int(sample['pos_gt_count']),
            'neg_gt_count': int(sample['neg_gt_count']),
            'mae': float(mae),
            'squared_error': float(se),
            'combined_prompt': combined_prompt,
            'exemplars_info': {
                'positive_exemplars': len(pos_exemplars),
                'negative_exemplars': len(neg_exemplars)
            }
        }
        
        # Visualization saving removed - only saving results dictionaries
    
    # Calculate overall metrics
    overall_mae = mae_sum / total_count if total_count > 0 else 0
    overall_rmse = np.sqrt(se_sum / total_count) if total_count > 0 else 0
    
    # Save detailed results
    pred_file = os.path.join(dataset_pred_dir, f'{model_name}_detailed_results.json')
    with open(pred_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save as pickle too
    pred_pickle_file = os.path.join(dataset_pred_dir, f'{model_name}_detailed_results.pkl')
    with open(pred_pickle_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Prepare quantitative results
    quantitative_results = {
        'model_name': model_name,
        'dataset_name': dataset_folder_name,
        'evaluation_type': 'combined_2pos_1neg',
        'combined_results': {
            'total_images': total_count,
            'mae': float(overall_mae),
            'rmse': float(overall_rmse),
            'total_absolute_error': float(mae_sum),
            'total_squared_error': float(se_sum)
        }
    }
    
    print(f"\nCombined Evaluation Results:")
    print(f"  MAE: {overall_mae:.2f}")
    print(f"  RMSE: {overall_rmse:.2f}")
    print(f"  Total images: {total_count}")
    
    # Save quantitative results
    quant_file = os.path.join(dataset_quant_dir, f'{model_name}_quantitative_results.json')
    with open(quant_file, 'w') as f:
        json.dump(quantitative_results, f, indent=2)
    
    # Also save as pickle
    quant_pickle_file = os.path.join(dataset_quant_dir, f'{model_name}_quantitative_results.pkl')
    with open(quant_pickle_file, 'wb') as f:
        pickle.dump(quantitative_results, f)
    
    # Save a simple summary file
    summary_file = os.path.join(dataset_quant_dir, f'{model_name}_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"CountGD Combined Evaluation Results Summary\n")
        f.write(f"==========================================\n\n")
        f.write(f"Dataset: {dataset_folder_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Evaluation Type: Combined (2 positive + 1 negative exemplars + combined prompt)\n")
        f.write(f"Prompt Format: 'positive_class and negative_class'\n")
        f.write(f"Confidence Threshold: {args.confidence_thresh}\n\n")
        f.write(f"Combined Results (Positive + Negative Objects):\n")
        f.write(f"  MAE:  {quantitative_results['combined_results']['mae']:.2f}\n")
        f.write(f"  RMSE: {quantitative_results['combined_results']['rmse']:.2f}\n")
        f.write(f"  Total Images: {quantitative_results['combined_results']['total_images']}\n")
    
    print(f"\nQuantitative results saved to: {dataset_quant_dir}")
    print(f"Detailed results saved to: {dataset_pred_dir}")
    # Visualization output message removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PairTally CountGD Count Both Classes Evaluation", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
