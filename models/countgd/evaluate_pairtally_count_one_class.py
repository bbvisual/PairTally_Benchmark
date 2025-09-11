import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'CountGD'))

import glob
import random
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


# CountOneClass Dataset class for PairTally data
class PairTallyDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None):
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.transform = transform
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_files = list(self.annotations.keys())
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        annotation = self.annotations[image_name]
        
        # Load image
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        
        # Extract exemplar boxes from FSC147 format
        pos_exemplar_coords = annotation['box_examples_coordinates'][:3]  # Take first 3
        neg_exemplar_coords = annotation['negative_box_exemples_coordinates'][:3]  # Take first 3
        
        # Convert from FSC147 format [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] to [x1, y1, x2, y2]
        pos_boxes = []
        for box_coords in pos_exemplar_coords:
            x1, y1 = box_coords[0]
            x2, y2 = box_coords[2]
            pos_boxes.append([x1, y1, x2, y2])
        
        neg_boxes = []
        for box_coords in neg_exemplar_coords:
            x1, y1 = box_coords[0]
            x2, y2 = box_coords[2]
            neg_boxes.append([x1, y1, x2, y2])
        
        # Get ground truth counts from points (not exemplar boxes!)
        pos_gt_count = len(annotation['points'])
        neg_gt_count = len(annotation['negative_points'])
        
        # Extract prompts from filename
        # Format: {positive_class}_{negative_class}_{...}.jpg
        filename_parts = image_name.split('_')
        pos_class = filename_parts[0].replace('-', ' ')  # Convert "5-cents-coin" to "5 cents coin"
        neg_class = filename_parts[1].replace('-', ' ')  # Convert "10-cents-coin" to "10 cents coin"
        
        # Create specific prompts for this image
        pos_prompt = pos_class
        neg_prompt = neg_class
        
        return {
            'image': image,
            'image_name': image_name,
            'pos_boxes': pos_boxes,  # Exemplar boxes for inference
            'neg_boxes': neg_boxes,  # Exemplar boxes for inference
            'pos_gt_count': pos_gt_count,  # Actual ground truth count
            'neg_gt_count': neg_gt_count,  # Actual ground truth count
            'pos_prompt': pos_prompt,
            'neg_prompt': neg_prompt,
            'annotation': annotation
        }


def get_args_parser():
    parser = argparse.ArgumentParser("PairTally CountGD Evaluation", add_help=False)
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
    # Updated argument names to match other models
    parser.add_argument(
        "--annotation_file",
        help="path to PairTally annotations file",
        default="../test_bbx_frames/annotations/annotation_pairtally_dataset.json",
    )
    parser.add_argument(
        "--image_dir",
        help="path to PairTally images folder",
        default="../test_bbx_frames/images",
    )
    parser.add_argument(
        "--model_name",
        help="model name for saving results",
        default="CountGD",
    )
    parser.add_argument(
        "--dataset_name",
        help="dataset name for output directory",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory for results",
        default="./CountGD_PairTally_Results",
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
    random.seed(seed)

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


def get_inds_from_tokens_and_keyphrases(tokenizer, tokens, keyphrases):
    """Get token indices that correspond to the keyphrases"""
    inds = []
    for keyphrase in keyphrases:
        tokenized_phrase = tokenizer([keyphrase], padding="longest", return_tensors="pt")["input_ids"][0][1:-1]  # remove CLS and SEP tokens
        tokenized_phrase = tokenizer.convert_ids_to_tokens(tokenized_phrase)
        
        for ind in range(len(tokens)):
            if tokens[ind: (ind + len(tokenized_phrase))] == tokenized_phrase:
                for sub_ind in range(len(tokenized_phrase)):
                    inds.append(ind + sub_ind)
                break
    return inds


def run_inference(model, image, prompt, exemplar_boxes, transform, args):
    """Run inference on a single image with given prompt and exemplar boxes"""
    # Convert exemplar boxes to the format expected by CountGD
    # PairTally format: [x1, y1, x2, y2] in absolute coordinates
    # CountGD expects: [x1, y1, x2, y2] in absolute coordinates (same format)
    exemplars = []
    if len(exemplar_boxes) > 0:
        # Use up to 3 exemplars (following FSC-147 convention)
        for box in exemplar_boxes[:3]:
            x1, y1, x2, y2 = box
            exemplars.append([x1, y1, x2, y2])
    
    # Transform image and exemplars
    input_image, target = transform(image, {"exemplars": torch.tensor(exemplars, dtype=torch.float32)})
    input_image = input_image.cuda()
    input_exemplar = target["exemplars"].cuda()
    
    print(f"    Exemplars shape: {input_exemplar.shape}, using {len(exemplars)} exemplars")
    
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


# Visualization function removed - only saving results dictionaries


def save_qualitative_data(img_tensor, exemplar_bboxes, pred_bboxes, pred_scores, img_id, img_name, 
                         num_gt, num_pred, mae_error, class_type, output_dir):
    """Save qualitative data to structured format matching GeCo pattern"""
    
    # Convert tensors to lists for JSON serialization
    exemplar_bboxes_list = exemplar_bboxes.tolist() if hasattr(exemplar_bboxes, 'tolist') else exemplar_bboxes
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
        'exemplar_boxes': exemplar_bboxes_list,  # Absolute coordinates
        'predicted_boxes': pred_bboxes_list,     # Absolute coordinates
        'prediction_scores': pred_scores_list,
        'coordinate_format': 'absolute_xyxy',    # Format specification
        'notes': {
            'exemplar_boxes_shown': min(3, len(exemplar_bboxes_list)),
            'total_exemplar_boxes': len(exemplar_bboxes_list),
            'total_predicted_boxes': len(pred_bboxes_list)
        }
    }
    
    return data_entry

def save_quantitative_results(all_results, dataset_folder_name, model_name):
    """Save quantitative metrics (MAE, RMSE) to separate folder matching GeCo pattern"""
    
    # Create quantitative results directory
    quant_output_dir = "../../results/CountGD-quantitative"
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
        f.write(f"CountGD Quantitative Results Summary\n")
        f.write(f"===================================\n\n")
        f.write(f"Dataset: {dataset_folder_name}\n")
        f.write(f"Model: {model_name}\n")
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
    args = get_args_parser().parse_args()
    
    print("CountGD PairTally CountOneClass Evaluation")
    print("=================================")
    print(f"Model: {args.model_name}")
    print(f"Annotation file: {args.annotation_file}")
    print(f"Image directory: {args.image_dir}")
    print(f"Confidence threshold: {args.confidence_thresh}")
    print(f"Output limit: {args.output_limit}")

    # Build model and transforms
    print("Building model...")
    model, transform = build_model_and_transforms(args)
    print("Model loaded successfully!")

    # Load dataset
    print("Loading PairTally dataset...")
    dataset = PairTallyDataset(args.annotation_file, args.image_dir, transform=None)
    print(f"Loaded {len(dataset)} images")
    
    # Get dataset folder name - extract from annotation file path
    if args.dataset_name:
        dataset_folder_name = args.dataset_name
    else:
        dataset_folder_name = os.path.basename(os.path.dirname(os.path.dirname(args.annotation_file)))
    
    # Create output directory structure matching GeCo pattern
    base_output_dir = "../../results/CountGD-qualitative"
    dataset_output_dir = os.path.join(base_output_dir, dataset_folder_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    print(f"Saving qualitative data to: {dataset_output_dir}")
    
    # Store all results in GeCo format
    all_results = {
        'dataset': dataset_folder_name,
        'model_name': args.model_name,
        'model_path': args.pretrain_model_path,
        'evaluation_info': {
            'confidence_thresh': args.confidence_thresh,
            'annotation_file': args.annotation_file,
            'image_dir': args.image_dir,
            'config_file': args.config
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
        pos_boxes = data['pos_boxes']  # Exemplar boxes for positive class
        neg_boxes = data['neg_boxes']  # Exemplar boxes for negative class
        pos_prompt = data['pos_prompt']
        neg_prompt = data['neg_prompt']
        
        print(f"\nProcessing {i+1}/{limit}: {image_name}")
        
        # Ground truth counts from point annotations (not exemplar boxes!)
        gt_pos_count = data['pos_gt_count']
        gt_neg_count = data['neg_gt_count']
        
        # ===== POSITIVE CLASS INFERENCE =====
        if len(pos_boxes) >= 3:
            print(f"  Running positive inference with prompt: '{pos_prompt}' and {len(pos_boxes)} exemplars")
            pred_pos_boxes, pred_pos_logits = run_inference(model, image, pos_prompt, pos_boxes, transform, args)
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
            
            # Convert exemplar boxes to absolute coordinates (they might already be)
            pos_exemplars_abs = pos_boxes[:3]  # Take first 3 exemplars
            
            # Save positive class data entry
            data_entry_pos = save_qualitative_data(
                image, pos_exemplars_abs, pred_pos_boxes_abs, 
                pred_pos_logits.cpu().numpy().tolist() if len(pred_pos_logits) > 0 else [],
                i, image_name, gt_pos_count, pred_pos_count, pos_mae, 
                'positive', dataset_output_dir
            )
            all_results['class_results']['positive']['images'].append(data_entry_pos)
            
            print(f"  Positive: GT={gt_pos_count}, Pred={pred_pos_count}, MAE={pos_mae}")
        else:
            print(f"  Skipping positive class - only {len(pos_boxes)} exemplars (need 3)")
        
        # ===== NEGATIVE CLASS INFERENCE =====
        if len(neg_boxes) >= 3:
            print(f"  Running negative inference with prompt: '{neg_prompt}' and {len(neg_boxes)} exemplars")
            pred_neg_boxes, pred_neg_logits = run_inference(model, image, neg_prompt, neg_boxes, transform, args)
            pred_neg_count = len(pred_neg_boxes)
            
            # Calculate error for negative class
            neg_mae = abs(pred_neg_count - gt_neg_count)
            total_ae_neg += neg_mae
            
            # Convert boxes to absolute coordinates for saving
            w, h = image.size
            pred_neg_boxes_abs = []
            if len(pred_neg_boxes) > 0:
                for cx, cy, box_w, box_h in pred_neg_boxes.cpu().numpy():
                    x1 = (cx - box_w/2) * w
                    y1 = (cy - box_h/2) * h
                    x2 = (cx + box_w/2) * w
                    y2 = (cy + box_h/2) * h
                    pred_neg_boxes_abs.append([float(x1), float(y1), float(x2), float(y2)])
            
            # Convert exemplar boxes to absolute coordinates (they might already be)
            neg_exemplars_abs = neg_boxes[:3]  # Take first 3 exemplars
            
            # Save negative class data entry
            data_entry_neg = save_qualitative_data(
                image, neg_exemplars_abs, pred_neg_boxes_abs,
                pred_neg_logits.cpu().numpy().tolist() if len(pred_neg_logits) > 0 else [],
                i, image_name, gt_neg_count, pred_neg_count, neg_mae,
                'negative', dataset_output_dir
            )
            all_results['class_results']['negative']['images'].append(data_entry_neg)
            
            print(f"  Negative: GT={gt_neg_count}, Pred={pred_neg_count}, MAE={neg_mae}")
        else:
            print(f"  Skipping negative class - only {len(neg_boxes)} exemplars (need 3)")
    
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
    
    # Save individual class files matching GeCo pattern
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
    
    # Save quantitative results (MAE, RMSE) matching GeCo pattern
    save_quantitative_results(all_results, dataset_folder_name, args.model_name)
    
    print(f"\n=== Final Summary ===")
    print(f"Dataset folder: {dataset_folder_name}")
    print(f"Model: {args.model_name}")
    print(f"Qualitative results saved to: {dataset_output_dir}")
    print(f"Quantitative results saved to: ../../results/CountGD-quantitative/{dataset_folder_name}/")
    print(f"\nQualitative files created:")
    print(f"  - positive_qualitative_data.json")
    print(f"  - negative_qualitative_data.json") 
    print(f"  - complete_qualitative_data.json")
    print(f"  - complete_qualitative_data.pkl")
    print("\nCountGD evaluation completed successfully!")


if __name__ == "__main__":
    main() 