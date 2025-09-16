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


# Custom Dataset class for PairTally data (text-only)
class PairTallyDataset(Dataset):
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
        
        # Get ground truth counts from points (not exemplar boxes!)
        pos_gt_count = len(annotation['points'])
        neg_gt_count = len(annotation['negative_points'])
        
        # Extract prompts from filename
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
            pos_class = pos_descriptions[min(pos_variant - 1, len(pos_descriptions) - 1)]
        else:
            pos_class = pos_class_code.lower()  # fallback
            
        if category in self.class_mapping and neg_base_code in self.class_mapping[category]:
            neg_descriptions = self.class_mapping[category][neg_base_code]
            # Use the variant number to select the specific description (1-indexed)
            neg_class = neg_descriptions[min(neg_variant - 1, len(neg_descriptions) - 1)]
        else:
            neg_class = neg_class_code.lower()  # fallback
        
        # Create specific prompts for this image
        pos_prompt = pos_class
        neg_prompt = neg_class
        
        return {
            'image': image,
            'image_name': image_name,
            'pos_gt_count': pos_gt_count,
            'neg_gt_count': neg_gt_count,
            'pos_prompt': pos_prompt,
            'neg_prompt': neg_prompt,
            'annotation': annotation
        }


def get_args_parser():
    parser = argparse.ArgumentParser("PairTally CountGD Text-Only Evaluation", add_help=False)
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
    parser.add_argument(
        "--annotations_file",
        help="path to PairTally annotations file",
        default="../test_bbx_frames/annotations/annotation_FSC147_384.json",
    )
    parser.add_argument(
        "--images_folder",
        help="path to PairTally images folder",
        default="../test_bbx_frames/images_384_VarV2",
    )
    parser.add_argument(
        "--output_dir",
        help="output directory for results",
        default="./CountGD_PairTally_TextOnly_Results",
    )
    parser.add_argument(
        "--base_data_path",
        help="base path to PairTally data",
        default=".",
    )
    parser.add_argument(
        "--dataset_name",
        help="name of the dataset folder",
        default="test_bbx_frames",
    )
    parser.add_argument(
        "--single_dataset",
        help="run evaluation on a single specific dataset",
        default=None,
    )
    parser.add_argument(
        "--confidence_thresh", help="confidence threshold for model", default=0.23, type=float
    )
    parser.add_argument(
        "--output_limit", help="limit number of output images for testing", default=None, type=int
    )
    parser.add_argument("--finetune_ignore", type=str, nargs="+")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_false")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--local_rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument(
        "--local-rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")
    return parser


def build_model_and_transforms(args):
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    data_transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            normalize,
        ]
    )
    cfg = SLConfig.fromfile(args.config)
    cfg.merge_from_dict({"text_encoder_type": "CountGD/checkpoints/bert-base-uncased"})
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # we use register to maintain models from catdet6 on.
    from CountGD.models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, _, _ = build_func(args)

    model.to(device)

    checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
    model.load_state_dict(checkpoint, strict=False)

    model.eval()

    return model, data_transform


def run_inference_text_only(model, image, prompt, transform, args):
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


# Visualization function removed - only saving results dictionaries


def save_qualitative_data(img_tensor, pred_bboxes, pred_scores, img_id, img_name, 
                         num_gt, num_pred, mae_error, class_type, output_dir, prompt):
    """Save qualitative data to structured format matching CountGD pattern"""
    
    # Convert tensors to lists for JSON serialization
    pred_bboxes_list = pred_bboxes.cpu().numpy().tolist() if hasattr(pred_bboxes, 'cpu') else pred_bboxes
    pred_scores_list = pred_scores.cpu().numpy().tolist() if hasattr(pred_scores, 'cpu') else pred_scores
    
    # Create data structure
    data_entry = {
        'image_id': int(img_id),
        'image_name': img_name,
        'class_type': class_type,  # 'positive' or 'negative'
        'gt_count': int(num_gt),
        'pred_count': int(num_pred),
        'mae_error': float(mae_error),
        'predicted_boxes': pred_bboxes_list,     # Absolute coordinates
        'prediction_scores': pred_scores_list,
        'prompt': prompt,
        'inference_mode': 'text_only',
        'coordinate_format': 'absolute_xyxy',    # Format specification
        'notes': {
            'total_predicted_boxes': len(pred_bboxes_list),
            'no_exemplar_boxes_used': True
        }
    }
    
    return data_entry

def save_quantitative_results(all_results, dataset_folder_name, model_name):
    """Save quantitative metrics (MAE, RMSE) to separate folder matching CountGD pattern"""
    
    # Create quantitative results directory
    quant_output_dir = "../../results/CountGD-TextOnly-quantitative"
    dataset_quant_dir = os.path.join(quant_output_dir, dataset_folder_name)
    os.makedirs(dataset_quant_dir, exist_ok=True)
    
    # Extract quantitative metrics
    quantitative_results = {
        'dataset': dataset_folder_name,
        'model_name': model_name,
        'evaluation_info': all_results['evaluation_info'],
        'class_results': {}
    }
    
    overall_mae_sum = 0
    overall_se_sum = 0
    overall_count = 0
    
    for class_type in ['positive', 'negative']:
        if class_type in all_results['class_results']:
            class_images = all_results['class_results'][class_type]['images']
            class_count = len(class_images)
            
            if class_count > 0:
                # Calculate MAE and RMSE for this class
                mae_sum = sum(img['mae_error'] for img in class_images)
                se_sum = sum(img['mae_error'] ** 2 for img in class_images)
                
                class_mae = mae_sum / class_count
                class_rmse = np.sqrt(se_sum / class_count)
                
                # Store class metrics
                quantitative_results['class_results'][class_type] = {
                    'total_images': class_count,
                    'mae': float(class_mae),
                    'rmse': float(class_rmse),
                    'total_absolute_error': float(mae_sum),
                    'total_squared_error': float(se_sum)
                }
                
                # Accumulate for overall metrics
                overall_mae_sum += mae_sum
                overall_se_sum += se_sum
                overall_count += class_count
                
                print(f"{class_type.capitalize()} Class Results:")
                print(f"  MAE: {class_mae:.2f}")
                print(f"  RMSE: {class_rmse:.2f}")
                print(f"  Total images: {class_count}")
    
    # Calculate overall metrics
    if overall_count > 0:
        overall_mae = overall_mae_sum / overall_count
        overall_rmse = np.sqrt(overall_se_sum / overall_count)
        
        quantitative_results['overall'] = {
            'total_images': overall_count,
            'mae': float(overall_mae),
            'rmse': float(overall_rmse),
            'total_absolute_error': float(overall_mae_sum),
            'total_squared_error': float(overall_se_sum)
        }
        
        print(f"\nOverall Quantitative Results:")
        print(f"  MAE: {overall_mae:.2f}")
        print(f"  RMSE: {overall_rmse:.2f}")
        print(f"  Total images: {overall_count}")
    
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
        f.write(f"CountGD Text-Only Quantitative Results Summary\n")
        f.write(f"===========================================\n\n")
        f.write(f"Dataset: {dataset_folder_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Inference Mode: Text-Only (no exemplar boxes)\n")
        f.write(f"Confidence Threshold: {all_results['evaluation_info']['confidence_thresh']}\n\n")
        
        for class_type, metrics in quantitative_results['class_results'].items():
            f.write(f"{class_type.capitalize()} Class:\n")
            f.write(f"  MAE:  {metrics['mae']:.2f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"  Images: {metrics['total_images']}\n\n")
        
        if 'overall' in quantitative_results:
            f.write(f"Overall Results:\n")
            f.write(f"  MAE:  {quantitative_results['overall']['mae']:.2f}\n")
            f.write(f"  RMSE: {quantitative_results['overall']['rmse']:.2f}\n")
            f.write(f"  Total Images: {quantitative_results['overall']['total_images']}\n")
    
    print(f"\nQuantitative results saved to: {dataset_quant_dir}")
    print(f"Files created:")
    print(f"  - {model_name}_quantitative_results.json")
    print(f"  - {model_name}_quantitative_results.pkl")
    print(f"  - {model_name}_summary.txt")
    
    return quant_file

def main():
    parser = argparse.ArgumentParser("PairTally CountGD Text-Only Evaluation", parents=[get_args_parser()])
    args = parser.parse_args()
    
    # If single_dataset is provided, use it; otherwise use dataset_name
    dataset_to_use = args.single_dataset if args.single_dataset else args.dataset_name
    
    # Only update paths if they weren't explicitly provided
    # Check if paths are still at their default values
    default_annotations = "./test_bbx_frames/annotations/annotation_FSC147_384.json"
    default_images = "../test_bbx_frames/images_384_VarV2"
    
    if args.base_data_path and dataset_to_use:
        # Only override if using default paths, not explicit ones
        if args.annotations_file == default_annotations or not os.path.exists(args.annotations_file):
            args.annotations_file = os.path.join(args.base_data_path, dataset_to_use, "annotations", "annotation_FSC147_384.json")
        
        if args.images_folder == default_images or not os.path.exists(args.images_folder):
            args.images_folder = os.path.join(args.base_data_path, dataset_to_use, "images_384_VarV2")
    
    # Get dataset folder name for output organization
    if args.single_dataset:
        dataset_folder_name = args.single_dataset
    elif args.dataset_name:
        dataset_folder_name = args.dataset_name
    else:
        # Try to extract dataset name from annotations file path
        annotations_dir = os.path.dirname(args.annotations_file)
        parent_dir = os.path.dirname(annotations_dir)
        dataset_folder_name = os.path.basename(parent_dir)
        
        # If we still can't determine it, try from images folder
        if not dataset_folder_name or dataset_folder_name == "annotations":
            images_parent = os.path.dirname(args.images_folder)
            dataset_folder_name = os.path.basename(images_parent)
        
        # Final fallback
        if not dataset_folder_name:
            dataset_folder_name = "unknown_dataset"
    
    print("CountGD PairTally Text-Only Evaluation")
    print("===================================")
    print(f"Dataset: {dataset_folder_name}")
    print(f"Annotations file: {args.annotations_file}")
    print(f"Images folder: {args.images_folder}")
    print(f"Confidence threshold: {args.confidence_thresh}")
    print(f"Output limit: {args.output_limit}")
    
    print("Building model and loading weights...")
    model, transform = build_model_and_transforms(args)
    print("Model loaded successfully!")
    
    # Create output directory structure matching CountGD pattern
    base_output_dir = "../../results/CountGD-TextOnly-qualitative"
    dataset_output_dir = os.path.join(base_output_dir, dataset_folder_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    print(f"Saving qualitative data to: {dataset_output_dir}")
    
    # Load dataset
    print("Loading PairTally dataset...")
    dataset = PairTallyDataset(args.annotations_file, args.images_folder, transform=None)
    print(f"Loaded {len(dataset)} images")
    
    # Store all results in CountGD format
    all_results = {
        'dataset': dataset_folder_name,
        'model_name': 'CountGD-TextOnly',
        'model_path': args.pretrain_model_path,
        'evaluation_info': {
            'confidence_thresh': args.confidence_thresh,
            'annotation_file': args.annotations_file,
            'image_dir': args.images_folder,
            'config_file': args.config,
            'inference_mode': 'text_only'
        },
        'class_results': {
            'positive': {'images': []},
            'negative': {'images': []}
        }
    }
    
    total_ae_pos = 0.0
    total_ae_neg = 0.0
    
    # Determine processing limit
    limit = len(dataset) if args.output_limit is None else min(args.output_limit, len(dataset))
    print(f"Processing {limit} images...")

    for i in range(limit):
        data = dataset[i]
        image = data['image']
        image_name = data['image_name']
        pos_prompt = data['pos_prompt']
        neg_prompt = data['neg_prompt']
        
        print(f"\nProcessing {i+1}/{limit}: {image_name}")
        
        # Ground truth counts from point annotations (not exemplar boxes!)
        gt_pos_count = data['pos_gt_count']
        gt_neg_count = data['neg_gt_count']
        
        # ===== POSITIVE CLASS INFERENCE =====
        print(f"  Running positive inference with text-only prompt: '{pos_prompt}'")
        pred_pos_boxes, pred_pos_logits = run_inference_text_only(model, image, pos_prompt, transform, args)
        pred_pos_count = len(pred_pos_boxes)
        
        # Calculate error for positive class
        pos_mae = abs(pred_pos_count - gt_pos_count)
        total_ae_pos += pos_mae
        
        # Convert boxes to absolute coordinates for saving
        w, h = image.size
        pred_pos_boxes_abs = []
        if len(pred_pos_boxes) > 0:
            for cx, cy, box_w, box_h in pred_pos_boxes.cpu().numpy():
                x1 = (cx - box_w/2) * w
                y1 = (cy - box_h/2) * h
                x2 = (cx + box_w/2) * w
                y2 = (cy + box_h/2) * h
                pred_pos_boxes_abs.append([float(x1), float(y1), float(x2), float(y2)])
        
        # Save positive class data entry
        data_entry_pos = save_qualitative_data(
            image, pred_pos_boxes_abs, 
            pred_pos_logits.cpu().numpy().tolist() if len(pred_pos_logits) > 0 else [],
            i, image_name, gt_pos_count, pred_pos_count, pos_mae, 
            'positive', dataset_output_dir, pos_prompt
        )
        all_results['class_results']['positive']['images'].append(data_entry_pos)
        
        print(f"  Positive: GT={gt_pos_count}, Pred={pred_pos_count}, MAE={pos_mae}")
        
        # ===== NEGATIVE CLASS INFERENCE =====
        print(f"  Running negative inference with text-only prompt: '{neg_prompt}'")
        pred_neg_boxes, pred_neg_logits = run_inference_text_only(model, image, neg_prompt, transform, args)
        pred_neg_count = len(pred_neg_boxes)
        
        # Calculate error for negative class
        neg_mae = abs(pred_neg_count - gt_neg_count)
        total_ae_neg += neg_mae
        
        # Convert boxes to absolute coordinates for saving
        pred_neg_boxes_abs = []
        if len(pred_neg_boxes) > 0:
            for cx, cy, box_w, box_h in pred_neg_boxes.cpu().numpy():
                x1 = (cx - box_w/2) * w
                y1 = (cy - box_h/2) * h
                x2 = (cx + box_w/2) * w
                y2 = (cy + box_h/2) * h
                pred_neg_boxes_abs.append([float(x1), float(y1), float(x2), float(y2)])
        
        # Save negative class data entry
        data_entry_neg = save_qualitative_data(
            image, pred_neg_boxes_abs,
            pred_neg_logits.cpu().numpy().tolist() if len(pred_neg_logits) > 0 else [],
            i, image_name, gt_neg_count, pred_neg_count, neg_mae,
            'negative', dataset_output_dir, neg_prompt
        )
        all_results['class_results']['negative']['images'].append(data_entry_neg)
        
        print(f"  Negative: GT={gt_neg_count}, Pred={pred_neg_count}, MAE={neg_mae}")
    
    # Calculate summary statistics
    pos_results = all_results['class_results']['positive']['images']
    neg_results = all_results['class_results']['negative']['images']
    
    if len(pos_results) > 0:
        avg_mae_pos = total_ae_pos / len(pos_results)
        all_results['class_results']['positive']['summary'] = {
            'total_images': len(pos_results),
            'average_mae': float(avg_mae_pos),
            'total_absolute_error': float(total_ae_pos)
        }
        print(f"\nPositive Class Summary:")
        print(f"Average MAE: {avg_mae_pos:.2f}")
        print(f"Processed {len(pos_results)} images")
    
    if len(neg_results) > 0:
        avg_mae_neg = total_ae_neg / len(neg_results)
        all_results['class_results']['negative']['summary'] = {
            'total_images': len(neg_results),
            'average_mae': float(avg_mae_neg),
            'total_absolute_error': float(total_ae_neg)
        }
        print(f"\nNegative Class Summary:")
        print(f"Average MAE: {avg_mae_neg:.2f}")
        print(f"Processed {len(neg_results)} images")
    
    # Save individual class files matching CountGD pattern
    for class_type in ['positive', 'negative']:
        class_file = os.path.join(dataset_output_dir, f'{class_type}_qualitative_data.json')
        with open(class_file, 'w') as f:
            json.dump({
                'class_type': class_type,
                'summary': all_results['class_results'][class_type].get('summary', {}),
                'images': all_results['class_results'][class_type]['images']
            }, f, indent=2)
        print(f"{class_type.capitalize()} data saved to: {class_file}")
    
    # Save combined qualitative results
    combined_file = os.path.join(dataset_output_dir, 'complete_qualitative_data.json')
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Also save as pickle for easier Python loading
    pickle_file = os.path.join(dataset_output_dir, 'complete_qualitative_data.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Save quantitative results (MAE, RMSE) matching CountGD pattern
    save_quantitative_results(all_results, dataset_folder_name, 'CountGD-TextOnly')
    
    print(f"\n=== Final Summary ===")
    print(f"Dataset folder: {dataset_folder_name}")
    print(f"Model: CountGD-TextOnly")
    print(f"Qualitative results saved to: {dataset_output_dir}")
    print(f"Quantitative results saved to: ../../results/CountGD-TextOnly-quantitative/{dataset_folder_name}/")
    print(f"\nQualitative files created:")
    print(f"  - positive_qualitative_data.json")
    print(f"  - negative_qualitative_data.json") 
    print(f"  - complete_qualitative_data.json")
    print(f"  - complete_qualitative_data.pkl")
    print("\nCountGD Text-Only evaluation completed successfully!")


if __name__ == "__main__":
    main() 