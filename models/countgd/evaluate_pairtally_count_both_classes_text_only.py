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


# Custom Dataset class for PairTally data (text-only combined)
class PairTallyCombinedTextDataset(Dataset):
    def __init__(self, annotations_file, images_folder, transform=None):
        self.annotations_file = annotations_file
        self.images_folder = images_folder
        self.transform = transform
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_files = list(self.annotations.keys())
        
        # Class code to text description mapping
        self.class_mapping = {
            "FOO": {
                "PAS": ["pasta", "spiral pasta", "penne pasta"],
                "RIC": ["rice grain", "jasmine rice grain", "brown rice grain"],
                "LIM": ["citrus fruit", "lime", "calamansi"],
                "PEP": ["peppercorn", "black peppercorn", "white peppercorn"],
                "TOM": ["tomato", "normal tomato", "baby tomato"],
                "CHI": ["chili", "long chili", "short chili"],
                "PNT": ["peanut", "peanut with skin", "peanut without skin"],
                "BEA": ["bean", "black bean", "soy bean"],
                "SED": ["seed", "pumpkin seed", "sunflower seed"],
                "CFC": ["coffee candy", "brown coffee candy", "black coffee candy"],
                "ONI": ["shallot"],
                "CAN": ["candy"],
                "GAR": ["garlic"]
            },
            "FUN": {
                "CHK": ["checker piece", "black checker piece", "white checker piece"],
                "MAH": ["mahjong tile", "bamboo mahjong tile", "character mahjong tile"],
                "LEG": ["lego piece", "green lego piece", "light pink lego piece"],
                "CHS": ["chess piece", "black chess piece", "white chess piece"],
                "PZP": ["puzzle piece", "edge puzzle piece", "center puzzle piece"],
                "PUZ": ["puzzle piece", "edge puzzle piece", "center puzzle piece"],
                "PKC": ["poker chip", "blue poker chip", "white poker chip"],
                "PLC": ["playing card", "red playing card", "black playing card"],
                "MAR": ["marble", "big marble", "small marble"],
                "DIC": ["dice", "green dice", "white dice"],
                "CSC": ["chinese slim card", "chinese slim card without red marks", "chinese slim card with red marks"]
            },
            "HOU": {
                "TPK": ["toothpick", "straight plastic toothpick", "dental floss"],
                "CTB": ["cotton bud", "wooden cotton bud", "plastic cotton bud"],
                "PIL": ["pill", "white pill", "yellow pill"],
                "BAT": ["battery", "small AAA battery", "big AA battery"],
                "HCP": ["hair clipper", "black hair clipper", "brown hair clipper"],
                "MNY": ["money bill", "1000 vietnamese dong bill", "5000 vietnamese dong bill"],
                "COI": ["coin", "5 Australian cents coin", "10 Australian cents coin"],
                "BOT": ["bottle cap", "beer bottle cap", "plastic bottle cap"],
                "BBT": ["button", "button with 4 holes", "button with 2 holes"],
                "ULT": ["plastic utensil", "plastic spoon", "plastic fork"]
            },
            "OFF": {
                "PPN": ["push pin", "normal push pin", "round push pin"],
                "HST": ["heart sticker", "big heart sticker", "small heart sticker"],
                "CRS": ["craft stick", "red or orange craft stick", "blue or purple craft stick"],
                "RUB": ["rubber band", "yellow rubber band", "blue rubber band"],
                "STN": ["sticky note", "dark green sticky note", "light green sticky note"],
                "PPC": ["paper clip", "colored paper clip", "silver paper clip"],
                "PEN": ["pen", "pen with cap", "pen without cap"],
                "PNC": ["pencil"],
                "RHS": ["rhinestone", "round rhinestone", "star rhinestone"],
                "ZPT": ["zip tie", "short zip tie", "long zip tie"],
                "SFP": ["safety pin", "big safety pin", "small safety pin"],
                "LPP": ["lapel pin"],
                "WWO": ["wall wire organizer"]
            },
            "OTR": {
                "SCR": ["screw", "long silver concrete screw", "short bronze screw"],
                "BOL": ["bolt", "hex head bolt", "mushroom head bolt"],
                "NUT": ["nut", "hex nut", "square nut"],
                "WAS": ["washer", "metal washer", "nylon washer"],
                "BUT": ["button", "Beige button", "Clear button"],
                "NAI": ["nail", "common nail", "concrete nail"],
                "BEA": ["bead", "Blue and purple bead", "Orange and pink bead"],
                "IKC": ["ikea clip", "green ikea clip", "red ikea clip"],
                "IKE": ["ikea clip", "green ikea clip", "red ikea clip"],
                "PEG": ["peg", "grey peg", "white peg"],
                "STO": ["stone", "red stone", "yellowstone"]
            }
        }
        
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
        
        # Extract class names from filename for prompts
        # Format: {category}_{inter_intra}_{pos_class_code}_{neg_class_code}_{...}.jpg
        # Example: FOO_INTER_BEA1_SED2_058_038_9d20c4.jpg
        filename_parts = image_name.split('_')
        category = filename_parts[0]  # FOO, FUN, HOU, OFF, OTR
        pos_class_code = filename_parts[2]  # e.g., BEA1
        neg_class_code = filename_parts[3]  # e.g., SED2
        
        # Extract base class code (remove the number)
        pos_base_code = ''.join([c for c in pos_class_code if not c.isdigit()])  # BEA1 -> BEA
        neg_base_code = ''.join([c for c in neg_class_code if not c.isdigit()])  # SED2 -> SED
        
        # Extract variant number
        pos_variant = int(''.join([c for c in pos_class_code if c.isdigit()])) if any(c.isdigit() for c in pos_class_code) else 1  # BEA1 -> 1
        neg_variant = int(''.join([c for c in neg_class_code if c.isdigit()])) if any(c.isdigit() for c in neg_class_code) else 1  # SED2 -> 2
        
        # Get the text descriptions
        if category in self.class_mapping and pos_base_code in self.class_mapping[category]:
            pos_descriptions = self.class_mapping[category][pos_base_code]
            # Use the variant number to select the specific description (1-indexed)
            pos_prompt = pos_descriptions[min(pos_variant - 1, len(pos_descriptions) - 1)]
        else:
            pos_prompt = pos_class_code.lower()  # fallback
            
        if category in self.class_mapping and neg_base_code in self.class_mapping[category]:
            neg_descriptions = self.class_mapping[category][neg_base_code]
            # Use the variant number to select the specific description (1-indexed)
            neg_prompt = neg_descriptions[min(neg_variant - 1, len(neg_descriptions) - 1)]
        else:
            neg_prompt = neg_class_code.lower()  # fallback
        
        return {
            'image': image,
            'image_name': image_name,
            'pos_gt_count': pos_gt_count,
            'neg_gt_count': neg_gt_count,
            'combined_gt_count': combined_gt_count,
            'pos_prompt': pos_prompt,
            'neg_prompt': neg_prompt,
            'annotation': annotation
        }


def get_args_parser():
    parser = argparse.ArgumentParser("PairTally CountGD Combined Text-Only Evaluation", add_help=False)
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
        dataset = PairTallyCombinedTextDataset(
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


def run_text_only_inference(model, image, prompt, transform, args):
    """Run inference on a single image with only text prompt (no exemplars)"""
    # Use empty exemplars (like the original single_image_inference.py)
    input_image, target = transform(image, {"exemplars": torch.tensor([])})
    input_image = input_image.cuda()
    input_exemplar = target["exemplars"].cuda()
    
    print(f"    Text-only inference with prompt: '{prompt}' (no exemplars)")
    
    with torch.no_grad():
        model_output = model(
            input_image.unsqueeze(0),
            [input_exemplar],
            [torch.tensor([0]).cuda()],
            captions=[prompt + " ."],
        )
    
    logits = model_output["pred_logits"][0].sigmoid()
    boxes = model_output["pred_boxes"][0]
    
    # Apply confidence threshold
    box_mask = logits.max(dim=-1).values > args.confidence_thresh
    logits = logits[box_mask, :]
    boxes = boxes[box_mask, :]
    
    return boxes, logits


def run_combined_text_inference(model, image, pos_prompt, neg_prompt, transform, args):
    """Run combined inference using a single prompt that combines positive and negative descriptions"""
    
    # Create a single combined prompt
    combined_prompt = f"{pos_prompt} and {neg_prompt}"
    
    # Run inference with the combined prompt
    combined_boxes, combined_logits = run_text_only_inference(model, image, combined_prompt, transform, args)
    
    # For visualization purposes, we'll treat all detections as combined
    # Since we can't distinguish which boxes are positive vs negative from a single inference
    pos_boxes = torch.empty(0, 4)  # Empty for visualization
    neg_boxes = torch.empty(0, 4)  # Empty for visualization
    
    return combined_boxes, combined_logits, pos_boxes, neg_boxes


# Visualization function removed - only saving results dictionaries


def main(args):
    print("Loading model...")
    model, data_transform = load_model(args)
    
    print(f"Loading dataset from: {args.annotation_file}")
    dataset = PairTallyCombinedTextDataset(args.annotation_file, args.image_folder, data_transform)
    
    # Create output directories following PairTally-RESULTS pattern
    dataset_folder_name = args.dataset_name
    
    # Use PairTally-RESULTS structure with "combined-text-only" identifier to avoid overwriting
    quant_output_dir = "../../results/CountGD-combined-text-only-quantitative"
    dataset_quant_dir = os.path.join(quant_output_dir, dataset_folder_name)
    
    qual_output_dir = "../../results/CountGD-combined-text-only-qualitative"
    dataset_pred_dir = os.path.join(qual_output_dir, dataset_folder_name)
    
    # Visualization directory creation removed - only saving results dictionaries
    
    os.makedirs(dataset_pred_dir, exist_ok=True)
    os.makedirs(dataset_quant_dir, exist_ok=True)
    
    print(f"Saving combined text-only qualitative data to: {dataset_pred_dir}")
    print(f"Saving combined text-only quantitative data to: {dataset_quant_dir}")
    
    model_name = "CountGD-Combined-TextOnly"
    
    print(f"Running combined text-only evaluation on {len(dataset)} images...")
    print(f"Model: {model_name}")
    print(f"Confidence threshold: {args.confidence_thresh}")
    
    all_results = {
        'model_name': model_name,
        'dataset_name': dataset_folder_name,
        'evaluation_info': {
            'confidence_thresh': args.confidence_thresh,
            'total_images': len(dataset),
            'inference_type': 'text_only_combined_prompts'
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
        pos_prompt = sample['pos_prompt']
        neg_prompt = sample['neg_prompt']
        annotation = sample['annotation']
        
        print(f"\nProcessing {i+1}/{len(dataset)}: {image_name}")
        print(f"  Combined GT count: {combined_gt_count} (pos: {sample['pos_gt_count']}, neg: {sample['neg_gt_count']})")
        print(f"  Positive prompt: '{pos_prompt}'")
        print(f"  Negative prompt: '{neg_prompt}'")
        
        # Run combined text-only inference
        combined_boxes, combined_logits, pos_boxes, neg_boxes = run_combined_text_inference(
            model, image, pos_prompt, neg_prompt, data_transform, args
        )
        
        combined_pred_count = len(combined_boxes)
        pos_pred_count = len(pos_boxes)
        neg_pred_count = len(neg_boxes)
        
        mae = abs(combined_pred_count - combined_gt_count)
        se = (combined_pred_count - combined_gt_count) ** 2
        
        mae_sum += mae
        se_sum += se
        total_count += 1
        
        print(f"  Predicted: {combined_pred_count} total, MAE: {mae:.2f}")
        
        # Store results
        all_results['image_results'][image_name] = {
            'combined_predicted_count': int(combined_pred_count),
            'pos_predicted_count': int(pos_pred_count),
            'neg_predicted_count': int(neg_pred_count),
            'combined_gt_count': int(combined_gt_count),
            'pos_gt_count': int(sample['pos_gt_count']),
            'neg_gt_count': int(sample['neg_gt_count']),
            'mae': float(mae),
            'squared_error': float(se),
            'prompts_used': {
                'positive_prompt': pos_prompt,
                'negative_prompt': neg_prompt
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
        'evaluation_type': 'combined_text_only',
        'combined_results': {
            'total_images': total_count,
            'mae': float(overall_mae),
            'rmse': float(overall_rmse),
            'total_absolute_error': float(mae_sum),
            'total_squared_error': float(se_sum)
        }
    }
    
    print(f"\nCombined Text-Only Evaluation Results:")
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
        f.write(f"CountGD Combined Text-Only Evaluation Results Summary\n")
        f.write(f"===================================================\n\n")
        f.write(f"Dataset: {dataset_folder_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Evaluation Type: Combined Text-Only (positive + negative prompts)\n")
        f.write(f"Confidence Threshold: {args.confidence_thresh}\n\n")
        f.write(f"Combined Results (Positive + Negative Objects):\n")
        f.write(f"  MAE:  {quantitative_results['combined_results']['mae']:.2f}\n")
        f.write(f"  RMSE: {quantitative_results['combined_results']['rmse']:.2f}\n")
        f.write(f"  Total Images: {quantitative_results['combined_results']['total_images']}\n")
    
    print(f"\nQuantitative results saved to: {dataset_quant_dir}")
    print(f"Detailed results saved to: {dataset_pred_dir}")
    # Visualization output message removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PairTally CountGD Combined Text-Only Evaluation", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
