#!/usr/bin/env python3
"""
GeCo combined evaluation script for DICTA25
Uses 2 positive + 1 negative exemplars to count all objects in a single inference pass.
"""

import json
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn import DataParallel
from torchvision import ops
from tqdm import tqdm
import pickle
from PIL import Image

from models.geco_infer import build_model
from utils.data import resize_and_pad, xywh_to_x1y1x2y2
from utils.arg_parser import get_argparser
from torchvision import transforms as T
from torchvision.transforms import functional as TVF

class CombinedFSC147Dataset(Dataset):
    """Dataset that combines positive and negative exemplars for unified inference"""
    
    def __init__(self, annotation_file, image_dir, evaluation=True):
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.evaluation = evaluation
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_names = list(self.annotations.keys())
        
        # Initialize same settings as GeCo dataset
        self.img_size = 1024
        self.num_objects = 3
        self.zero_shot = False
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        annotation = self.annotations[img_name]
        
        # Load image exactly like GeCo dataset
        img = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        img = T.ToTensor()(img)
        
        # Process bboxes exactly like GeCo dataset
        # Convert FSC147 format [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] to [x1, y1, x2, y2]
        positive_exemplars = torch.tensor(
            annotation['box_examples_coordinates'],
            dtype=torch.float32
        )[:3, [0, 2], :].reshape(-1, 4)[:self.num_objects, ...]
        
        negative_exemplars = torch.tensor(
            annotation['negative_box_exemples_coordinates'],
            dtype=torch.float32
        )[:3, [0, 2], :].reshape(-1, 4)[:self.num_objects, ...]
        
        # ===== COMBINED EXEMPLARS: 2 POSITIVE + 1 NEGATIVE =====
        # Take first 2 positive and first 1 negative
        if len(positive_exemplars) >= 2 and len(negative_exemplars) >= 1:
            selected_pos = positive_exemplars[:2]  # [2, 4]
            selected_neg = negative_exemplars[:1]  # [1, 4]
            combined_exemplars = torch.cat([selected_pos, selected_neg], dim=0)  # [3, 4]
        else:
            # If not enough exemplars, skip this image or use what's available
            combined_exemplars = torch.zeros(3, 4)  # Dummy exemplars
        
        # Apply GeCo evaluation preprocessing
        dummy_density_map = torch.zeros(1, 1024, 1024)
        dummy_gt_bboxes = torch.zeros(0, 4)
        
        result = resize_and_pad(
            img, combined_exemplars, 
            density_map=dummy_density_map, 
            gt_bboxes=dummy_gt_bboxes, 
            full_stretch=False if not self.zero_shot else True, 
            size=1024.0
        )
        
        # Unpack results (with density_map and gt_bboxes, returns 6 values)
        img_processed, combined_bboxes, _, _, scaling_factor, padwh = result
        
        # Check if bboxes are too small and need larger image size (like in GeCo evaluation)
        if (combined_bboxes[:, 2] - combined_bboxes[:, 0]).min() < 25 and (combined_bboxes[:, 3] - combined_bboxes[:, 1]).min() < 25 and not self.zero_shot:
            result = resize_and_pad(
                img, combined_exemplars,
                density_map=dummy_density_map,
                gt_bboxes=dummy_gt_bboxes,
                full_stretch=False,
                size=1536.0
            )
            img_processed, combined_bboxes, _, _, scaling_factor, padwh = result
        
        # Apply normalization like GeCo
        img_processed = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_processed)
        
        # Get ground truth counts from points
        pos_count = len(annotation['points'])  # Positive points
        neg_count = len(annotation['negative_points'])  # Negative points
        total_count = pos_count + neg_count  # Combined ground truth
        
        return {
            'image': img_processed,
            'image_name': img_name,
            'image_id': idx,
            'combined_exemplars': combined_bboxes,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'total_count': total_count,
            'padwh': padwh,
            'scaling_factor': scaling_factor,
            'has_enough_exemplars': len(positive_exemplars) >= 2 and len(negative_exemplars) >= 1
        }

def postprocess_for_data_extraction(img, bboxes, outputs, padwh, device, s_f=8):
    """Modified postprocess function that returns data for saving - matches GeCo postprocessing"""
    nms_bboxes = []
    nms_scores = []
    bs, c, h, w = img.shape
    
    for idx in range(img.shape[0]):
        if len(outputs[idx]['pred_boxes']) == 0:
            nms_bboxes.append(torch.zeros((0, 4)))
            nms_scores.append(torch.zeros((0)))
        else:
            # Apply NMS to predictions (same as GeCo evaluation)
            keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / s_f],
                          outputs[idx]['box_v'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / s_f], 0.5)
            
            boxes = (outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / s_f])[keep]
            boxes = torch.clamp(boxes, 0, 1)
            scores = (outputs[idx]['scores'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / s_f])[keep]
            
            # Filter out boxes in padded areas (same logic as GeCo evaluation)
            maxw = torch.tensor(img.shape[-1] - padwh[0], device=device)
            maxh = torch.tensor(img.shape[-2] - padwh[1], device=device)
            valid_mask = (boxes[:, 0] * h < maxw) & (boxes[:, 1] * w < maxh) & (boxes[:, 2] * h < maxw) & (boxes[:, 3] * w < maxh)
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            
            nms_bboxes.append(boxes)
            nms_scores.append(scores)
    
    return nms_bboxes, nms_scores

def save_qualitative_data(img_tensor, combined_bboxes, pred_bboxes, pred_scores, img_id, img_name, 
                         pos_count, neg_count, total_count, pred_count, mae_error, output_dir):
    """Save qualitative data for combined inference"""
    
    # Convert tensors to lists for JSON serialization
    combined_bboxes_list = combined_bboxes.cpu().numpy().tolist() if len(combined_bboxes) > 0 else []
    pred_bboxes_list = pred_bboxes.cpu().numpy().tolist() if len(pred_bboxes) > 0 else []
    pred_scores_list = pred_scores.cpu().numpy().tolist() if len(pred_scores) > 0 else []
    
    # Create data structure
    data_entry = {
        'image_id': int(img_id),
        'image_name': img_name,
        'gt_pos_count': int(pos_count),
        'gt_neg_count': int(neg_count),
        'gt_total_count': int(total_count),
        'pred_total_count': int(pred_count),
        'mae_error': float(mae_error),
        'combined_exemplar_boxes': combined_bboxes_list,  # 2 positive + 1 negative, normalized [0,1]
        'predicted_boxes': pred_bboxes_list,             # Normalized coordinates [0,1]
        'prediction_scores': pred_scores_list,
        'image_shape': list(img_tensor.shape),           # [C, H, W]
        'coordinate_format': 'normalized_xyxy',          # Format specification
        'notes': {
            'inference_type': 'combined_2pos_1neg',
            'exemplar_boxes_shown': len(combined_bboxes_list),
            'total_predicted_boxes': len(pred_bboxes_list)
        }
    }
    
    return data_entry

def save_quantitative_results(all_results, dataset_folder_name, model_name):
    """Save quantitative metrics (MAE, RMSE) to separate folder"""
    
    # Create quantitative results directory
    quant_output_dir = "../../results/GeCo-quantitative-combined"
    dataset_quant_dir = os.path.join(quant_output_dir, dataset_folder_name)
    os.makedirs(dataset_quant_dir, exist_ok=True)
    
    # Extract quantitative metrics
    images = all_results['results']['images']
    total_images = len(images)
    
    if total_images > 0:
        # Calculate MAE and RMSE
        mae_sum = sum(img['mae_error'] for img in images)
        se_sum = sum(img['mae_error'] ** 2 for img in images)
        
        overall_mae = mae_sum / total_images
        overall_rmse = np.sqrt(se_sum / total_images)
        
        quantitative_results = {
            'dataset': dataset_folder_name,
            'model_name': model_name,
            'inference_type': 'combined_2pos_1neg',
            'evaluation_info': all_results['evaluation_info'],
            'overall': {
                'mae': float(overall_mae),
                'rmse': float(overall_rmse),
                'total_images': total_images,
                'total_mae_sum': float(mae_sum),
                'total_se_sum': float(se_sum)
            }
        }
        
        # Save quantitative results
        quant_file = os.path.join(dataset_quant_dir, f'{model_name}_combined_quantitative_results.json')
        with open(quant_file, 'w') as f:
            json.dump(quantitative_results, f, indent=2)
        
        pickle_quant_file = os.path.join(dataset_quant_dir, f'{model_name}_combined_quantitative_results.pkl')
        with open(pickle_quant_file, 'wb') as f:
            pickle.dump(quantitative_results, f)
        
        # Save summary
        summary_file = os.path.join(dataset_quant_dir, f'{model_name}_combined_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"GeCo Combined Inference Results Summary\n")
            f.write(f"======================================\n\n")
            f.write(f"Dataset: {dataset_folder_name}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Inference Type: 2 Positive + 1 Negative Exemplars\n")
            f.write(f"Image Size: {all_results['evaluation_info']['image_size']}\n\n")
            f.write(f"Overall Results:\n")
            f.write(f"  MAE:  {overall_mae:.2f}\n")
            f.write(f"  RMSE: {overall_rmse:.2f}\n")
            f.write(f"  Total Images: {total_images}\n")
        
        print(f"\nQuantitative results saved to: {dataset_quant_dir}")
        return quant_file
    
    return None

@torch.no_grad()
def evaluate_combined(args):
    """Main evaluation function for combined inference"""
    
    # Setup
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)
    
    # Load model
    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pth'))['model']
    state_dict = {k if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Use command line arguments for annotation file and image directory
    annotation_file = args.annotation_file
    image_dir = args.image_dir
    
    # Create dataset
    dataset = CombinedFSC147Dataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        evaluation=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one image at a time
        drop_last=False,
        num_workers=args.num_workers,
    )
    
    # Get dataset folder name from the annotation file path or use provided dataset name
    if hasattr(args, 'dataset_name') and args.dataset_name:
        dataset_folder_name = args.dataset_name
    else:
        # Extract from annotation file path: /path/to/dataset_name/annotations/...
        dataset_folder_name = os.path.basename(os.path.dirname(os.path.dirname(annotation_file)))
    
    # Create output directory structure
    base_output_dir = "../../results/GeCo-qualitative-combined"
    dataset_output_dir = os.path.join(base_output_dir, dataset_folder_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    print(f"Saving combined inference data to: {dataset_output_dir}")
    
    # Store all results
    all_results = {
        'dataset': dataset_folder_name,
        'model_name': args.model_name,
        'model_path': args.model_path,
        'inference_type': 'combined_2pos_1neg',
        'evaluation_info': {
            'image_size': 1024,  # GeCo always uses 1024x1024 images
            'annotation_file': annotation_file,
            'image_dir': image_dir
        },
        'results': {
            'images': []
        }
    }
    
    total_ae = 0.0
    valid_images = 0
    
    print(f"\nProcessing {len(dataset)} images...")
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing images")):
        # Extract batch data
        img = batch_data['image'].to(device)
        img_name = batch_data['image_name'][0]
        img_id = batch_data['image_id'].item()
        combined_exemplars = batch_data['combined_exemplars'][0].to(device)
        pos_count = batch_data['pos_count'].item()
        neg_count = batch_data['neg_count'].item()
        total_count = batch_data['total_count'].item()
        has_enough_exemplars = batch_data['has_enough_exemplars'].item()
        
        # Get padding info
        padwh_raw = batch_data['padwh'][0]
        if isinstance(padwh_raw, torch.Tensor):
            padwh = (padwh_raw[0].item(), padwh_raw[1].item()) if len(padwh_raw) >= 2 else (padwh_raw[0].item(), 0)
        else:
            padwh = padwh_raw
        
        # ===== COMBINED INFERENCE: 2 POSITIVE + 1 NEGATIVE =====
        if has_enough_exemplars:
            # Run inference with combined exemplars (imgs already have batch dimension from dataloader)
            combined_bboxes = combined_exemplars.unsqueeze(0)  # [1, 3, 4]
            outputs, ref_points, centerness, outputs_coord, masks = model(img, combined_bboxes)
            
            # Post-process predictions
            nms_bboxes, nms_scores = postprocess_for_data_extraction(
                img, combined_bboxes, outputs, padwh, device
            )
            
            # Calculate count and error against total ground truth
            pred_count = len(nms_bboxes[0])
            mae_error = abs(total_count - pred_count)
            total_ae += mae_error
            valid_images += 1
            
            # Prepare data for saving
            pred_bboxes = nms_bboxes[0].cpu() if len(nms_bboxes[0]) > 0 else torch.zeros((0, 4))
            pred_scores = nms_scores[0].cpu() if len(nms_scores[0]) > 0 else torch.zeros((0,))
            
            # Save combined inference data entry (remove batch dimension for save function)
            normalized_combined_bboxes = combined_bboxes[0].cpu() / 1024.0
            data_entry = save_qualitative_data(
                img[0], normalized_combined_bboxes, pred_bboxes, pred_scores,
                img_id, img_name, pos_count, neg_count, total_count, 
                pred_count, mae_error, dataset_output_dir
            )
            all_results['results']['images'].append(data_entry)
            
        else:
            print(f"Skipping {img_name}: insufficient exemplars (need 2+ positive, 1+ negative)")
        
        # Optional: limit number of processed images
        if args.output_limit is not None and batch_idx >= args.output_limit:
            print(f"Stopping after {batch_idx + 1} images (output_limit={args.output_limit})...")
            break
    
    # Calculate summary statistics
    if valid_images > 0:
        avg_mae = total_ae / valid_images
        all_results['results']['summary'] = {
            'total_valid_images': valid_images,
            'total_processed_images': batch_idx + 1,
            'average_mae': float(avg_mae),
            'total_absolute_error': float(total_ae)
        }
        print(f"\nCombined Inference Summary:")
        print(f"Valid images processed: {valid_images}")
        print(f"Average MAE: {avg_mae:.2f}")
    else:
        print(f"\nNo valid images processed (need 2+ positive and 1+ negative exemplars)")
        return
    
    # Save combined results
    combined_file = os.path.join(dataset_output_dir, 'combined_inference_data.json')
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Also save as pickle for easier Python loading
    pickle_file = os.path.join(dataset_output_dir, 'combined_inference_data.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Save quantitative results (MAE, RMSE)
    save_quantitative_results(all_results, dataset_folder_name, args.model_name)
    
    print(f"\n=== Final Summary ===")
    print(f"Dataset folder: {dataset_folder_name}")
    print(f"Model: {args.model_name}")
    print(f"Inference type: 2 Positive + 1 Negative Exemplars")
    print(f"Combined inference results saved to: {dataset_output_dir}")
    print(f"Quantitative results saved to: ../../results/GeCo-quantitative-combined/{dataset_folder_name}/")
    print(f"\nFiles created:")
    print(f"  - combined_inference_data.json")
    print(f"  - combined_inference_data.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('GeCo Combined Inference', parents=[get_argparser()])
    parser.add_argument('--output_limit', type=int, default=None, 
                       help='Limit number of processed images (for testing)')
    parser.add_argument('--annotation_file', type=str, required=True,
                       help='Path to annotation JSON file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to images directory')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Dataset name for output directory (will be inferred if not provided)')
    args = parser.parse_args()
    
    print("Starting GeCo combined inference (2 positive + 1 negative exemplars)...")
    print(f"Model: {args.model_name}")
    print(f"Annotation file: {args.annotation_file}")
    print(f"Image directory: {args.image_dir}")
    
    evaluate_combined(args)
    print("\nGeCo combined inference completed!")
