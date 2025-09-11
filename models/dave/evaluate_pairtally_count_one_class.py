#!/usr/bin/env python3
"""
DAVE evaluation script for DICTA25 - Based on GeCo/LOCA learnings
Handles positive and negative exemplars separately, saves both density maps and bounding boxes.
"""

import json
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
from PIL import Image

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'DAVE'))

from models.dave import build_model
from utils.arg_parser import get_argparser
from torchvision import transforms as T

class CustomFSC147Dataset(Dataset):
    """Custom dataset that loads from annotation file and handles positive/negative exemplars"""
    
    def __init__(self, annotation_file, image_dir, image_size=512, evaluation=True):
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.image_size = image_size
        self.evaluation = evaluation
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_names = list(self.annotations.keys())
        
        # Image transforms
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.image_names)
    
    def load_image(self, img_name, image_size=512):
        """Load and preprocess image, return scaling factors for bbox transformation"""
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Get original dimensions
        original_width, original_height = image.size
        
        # Convert to tensor first
        image = T.ToTensor()(image)
        
        # Calculate scaling factors for width and height
        # Since we're using simple resize to image_size x image_size (not maintaining aspect ratio)
        target_size = float(image_size)
        x_scale = target_size / original_width
        y_scale = target_size / original_height
        
        # DAVE model expects image_size x image_size images (typically 512x512)
        image = T.Resize((image_size, image_size), antialias=True)(image)
        
        # Apply normalization
        image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        
        return image, (x_scale, y_scale)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        annotation = self.annotations[img_name]
        
        # Load image and get scaling factors
        image, (x_scale, y_scale) = self.load_image(img_name, self.image_size)
        
        # Extract data from annotation
        # Process bboxes the same way as original FSC147Dataset
        positive_exemplars = torch.tensor(annotation['box_examples_coordinates'], dtype=torch.float32)[:3, [0, 2], :].reshape(-1, 4)
        negative_exemplars = torch.tensor(annotation['negative_box_exemples_coordinates'], dtype=torch.float32)[:3, [0, 2], :].reshape(-1, 4)
        
        # CRITICAL: Scale bboxes to match the resized image coordinate system
        # Format is [x1, y1, x2, y2], so scale with [x_scale, y_scale, x_scale, y_scale]
        bbox_scale_factor = torch.tensor([x_scale, y_scale, x_scale, y_scale], dtype=torch.float32)
        positive_exemplars = positive_exemplars * bbox_scale_factor
        negative_exemplars = negative_exemplars * bbox_scale_factor
        
        positive_points = torch.tensor(annotation['points'], dtype=torch.float32)
        negative_points = torch.tensor(annotation['negative_points'], dtype=torch.float32)
        
        # Scale points as well 
        positive_points = positive_points * torch.tensor([x_scale, y_scale], dtype=torch.float32)
        negative_points = negative_points * torch.tensor([x_scale, y_scale], dtype=torch.float32)
        
        # Get counts
        pos_count = len(positive_points)
        neg_count = len(negative_points)
        
        return {
            'image': image,
            'image_name': img_name,
            'image_id': idx,
            'positive_exemplars': positive_exemplars,
            'negative_exemplars': negative_exemplars,
            'positive_points': positive_points,
            'negative_points': negative_points,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'scaling_factors': (x_scale, y_scale)
        }

def save_density_map_files(density_map, img_name, class_type, output_dir):
    """Save density map as .npy file and create visualization image"""
    
    # Create subdirectories
    npy_dir = os.path.join(output_dir, f'density_maps_{class_type}')
    img_dir = os.path.join(output_dir, f'density_maps_{class_type}_images')
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    # Save as .npy file
    npy_path = os.path.join(npy_dir, f"{img_name}.npy")
    np.save(npy_path, density_map.cpu().numpy())
    
    # Create visualization image
    img_path = os.path.join(img_dir, f"{img_name}.png")
    density_np = density_map.cpu().numpy()
    
    if density_np.max() > 0:
        density_normalized = (density_np / density_np.max() * 255).astype(np.uint8)
    else:
        density_normalized = density_np.astype(np.uint8)
    
    # Save as grayscale image
    density_img = Image.fromarray(density_normalized, mode='L')
    density_img.save(img_path)
    
    return npy_path, img_path

def save_original_image(img_tensor, img_name, output_dir):
    """Save denormalized original image"""
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Denormalize image - ensure tensors are on same device
    device = img_tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(3, 1, 1)
    img_denorm = img_tensor * std + mean
    img_denorm = torch.clamp(img_denorm, 0, 1)
    
    img_pil = T.ToPILImage()(img_denorm)
    img_path = os.path.join(images_dir, f"{img_name}.png")
    img_pil.save(img_path)
    
    return img_path

def save_qualitative_data(img_tensor, exemplar_bboxes, density_map_pred, pred_bboxes, pred_scores, 
                         img_id, img_name, num_gt, num_pred, mae_error, class_type, output_dir):
    """Save qualitative data for DAVE model (both density maps and bounding boxes)"""
    
    # Save density map files
    density_npy_path, density_img_path = save_density_map_files(
        density_map_pred, img_name, class_type, output_dir
    )
    
    # Save original image (only once per image, not per class)
    original_img_path = save_original_image(img_tensor, img_name, output_dir)
    
    # Convert tensors to lists for JSON serialization
    exemplar_bboxes_list = exemplar_bboxes.cpu().numpy().tolist() if len(exemplar_bboxes) > 0 else []
    pred_bboxes_list = pred_bboxes.cpu().numpy().tolist() if len(pred_bboxes) > 0 else []
    pred_scores_list = pred_scores.cpu().numpy().tolist() if len(pred_scores) > 0 else []
    
    # Create data structure
    data_entry = {
        'image_id': int(img_id),
        'image_name': img_name,
        'class_type': class_type,  # 'positive' or 'negative'
        'gt_count': int(num_gt),
        'pred_count': int(num_pred),
        'mae_error': float(mae_error),
        'exemplar_boxes': exemplar_bboxes_list,  # Normalized coordinates [0,1]
        'predicted_boxes': pred_bboxes_list,     # Normalized coordinates [0,1]  
        'prediction_scores': pred_scores_list,
        'density_map_sum': float(density_map_pred.sum().item()),
        'image_shape': list(img_tensor.shape),   # [C, H, W]
        'coordinate_format': 'normalized_xyxy',  # Format specification for boxes
        'file_paths': {
            'original_image': original_img_path,
            'density_map_npy': density_npy_path,
            'density_map_image': density_img_path
        },
        'notes': {
            'exemplar_boxes_shown': min(3, len(exemplar_bboxes_list)),
            'total_exemplar_boxes': len(exemplar_bboxes_list),
            'total_predicted_boxes': len(pred_bboxes_list),
            'density_map_format': 'numpy_array_single_channel',
            'model_outputs': 'density_maps_and_bounding_boxes'
        }
    }
    
    return data_entry

def save_quantitative_results(all_results, dataset_folder_name, model_name):
    """Save quantitative metrics (MAE, RMSE) to separate folder"""
    
    # Create quantitative results directory
    quant_output_dir = "../../results/DAVE-quantitative"
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
                    'mae': float(class_mae),
                    'rmse': float(class_rmse),
                    'total_images': class_count,
                    'total_mae_sum': float(mae_sum),
                    'total_se_sum': float(se_sum)
                }
                
                # Add to overall metrics
                overall_mae_sum += mae_sum
                overall_se_sum += se_sum
                overall_count += class_count
    
    # Calculate overall metrics
    if overall_count > 0:
        overall_mae = overall_mae_sum / overall_count
        overall_rmse = np.sqrt(overall_se_sum / overall_count)
        
        quantitative_results['overall'] = {
            'mae': float(overall_mae),
            'rmse': float(overall_rmse),
            'total_images': overall_count
        }
    
    # Save quantitative results
    quant_file = os.path.join(dataset_quant_dir, f'{model_name}_quantitative_results.json')
    with open(quant_file, 'w') as f:
        json.dump(quantitative_results, f, indent=2)
    
    pickle_quant_file = os.path.join(dataset_quant_dir, f'{model_name}_quantitative_results.pkl')
    with open(pickle_quant_file, 'wb') as f:
        pickle.dump(quantitative_results, f)
    
    # Save summary
    summary_file = os.path.join(dataset_quant_dir, f'{model_name}_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"DAVE Quantitative Results Summary\n")
        f.write(f"=================================\n\n")
        f.write(f"Dataset: {dataset_folder_name}\n")
        f.write(f"Model: {model_name}\n\n")
        
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
    return quant_file

@torch.no_grad()
def evaluate_qualitative(args):
    """Main evaluation function for saving qualitative data"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build and load model
    print("Loading DAVE model...")
    model = build_model(args).to(device)
    
    # Wrap in DataParallel like in the demo
    from torch.nn import DataParallel
    gpu_id = 0 if device.type == 'cuda' else None
    if gpu_id is not None:
        model = DataParallel(model, device_ids=[gpu_id], output_device=gpu_id)
    else:
        model = DataParallel(model)
    
    # Load state dict - DAVE requires two checkpoints
    state_dict_path = os.path.join(args.model_path, f'{args.model_name}.pth')
    verification_path = os.path.join(args.model_path, 'verification.pth')
    print(f"Loading main weights from: {state_dict_path}")
    print(f"Loading feat_comp weights from: {verification_path}")
    
    # Load main model checkpoint
    checkpoint = torch.load(state_dict_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Load main model weights (strict=False to allow missing feat_comp)
    model.load_state_dict(state_dict, strict=False)
    
    # Load feat_comp module separately from verification checkpoint
    if os.path.exists(verification_path):
        verification_checkpoint = torch.load(verification_path, map_location=device)
        verification_state_dict = verification_checkpoint['model'] if 'model' in verification_checkpoint else verification_checkpoint
        
        # Extract feat_comp weights
        feat_comp_dict = {}
        for k, v in verification_state_dict.items():
            if 'feat_comp' in k:
                # Remove 'feat_comp.' prefix to get the actual module key
                feat_comp_key = k.split('feat_comp.')[1]
                feat_comp_dict[feat_comp_key] = v
        
        # Load feat_comp weights
        if feat_comp_dict:
            model.module.feat_comp.load_state_dict(feat_comp_dict)
            print("feat_comp module loaded successfully!")
        else:
            print("Warning: No feat_comp weights found in verification checkpoint")
    else:
        print(f"Warning: Verification checkpoint not found at {verification_path}")
    
    model.eval()
    print("Model loaded successfully!")
    
    # Setup paths from command line arguments
    annotation_file = args.annotation_file
    image_dir = args.image_dir
    
    # Create dataset - Use 384 for data but model will expect args.image_size (512 by default)
    dataset = CustomFSC147Dataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        image_size=args.image_size,  # Match available data size
        evaluation=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one image at a time
        drop_last=False,
        num_workers=getattr(args, 'num_workers', 0),
    )
    
    # Get dataset folder name - extract from annotation file path
    dataset_folder_name = os.path.basename(os.path.dirname(os.path.dirname(annotation_file)))
    
    # Create output directory structure
    base_output_dir = "../../results/DAVE-qualitative"
    dataset_output_dir = os.path.join(base_output_dir, dataset_folder_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    print(f"Saving qualitative data to: {dataset_output_dir}")
    
    # Store all results
    all_results = {
        'dataset': dataset_folder_name,
        'model_name': args.model_name,
        'model_path': args.model_path,
        'evaluation_info': {
            'image_size': getattr(args, 'image_size', 512),  # DAVE image size (typically 512x512)
            'annotation_file': annotation_file,
            'image_dir': image_dir,
            'backbone': args.backbone,
            'emb_dim': args.emb_dim,
            'num_heads': args.num_heads,
            'kernel_dim': args.kernel_dim,
            'reduction': args.reduction,
            'zero_shot': getattr(args, 'zero_shot', False),
            'prompt_shot': getattr(args, 'prompt_shot', False)
        },
        'class_results': {
            'positive': {'images': []},
            'negative': {'images': []}
        }
    }
    
    total_ae_pos = 0.0
    total_ae_neg = 0.0
    
    print(f"\nProcessing {len(dataset)} images...")
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing images")):
        # Extract batch data
        img = batch_data['image'].to(device)
        img_name = batch_data['image_name'][0]
        img_id = batch_data['image_id'].item()
        positive_exemplars = batch_data['positive_exemplars'][0]
        negative_exemplars = batch_data['negative_exemplars'][0]
        pos_count = batch_data['pos_count'].item()
        neg_count = batch_data['neg_count'].item()
        
        # ===== POSITIVE CLASS INFERENCE =====
        if len(positive_exemplars) >= 3:
            # positive_exemplars already has the right shape [3, 4] from processing
            pos_bboxes = positive_exemplars.unsqueeze(0).to(device)  # [1, 3, 4]
            # The coordinates are scaled to match the resized image
            
            # Run inference for positive class - DAVE returns (density_map, aux_outputs, tblr, bboxes)
            outputs_R, aux_outputs, tblr, pred_bboxes_pos = model(img, pos_bboxes)
            
            # Extract density map and predicted bboxes
            density_map_pred_pos = outputs_R
            
            # Extract bounding boxes and scores from DAVE's BoxList format
            if hasattr(pred_bboxes_pos, 'box'):
                pred_boxes_tensor = pred_bboxes_pos.box
                pred_scores_tensor = pred_bboxes_pos.fields.get('scores', torch.zeros(len(pred_boxes_tensor)))
            else:
                pred_boxes_tensor = torch.zeros((0, 4))
                pred_scores_tensor = torch.zeros((0,))
            
            # Calculate count and error for positive class
            pred_count_pos = len(pred_boxes_tensor) if len(pred_boxes_tensor) > 0 else float(density_map_pred_pos.sum().item())
            mae_error_pos = abs(pos_count - pred_count_pos)
            total_ae_pos += mae_error_pos
            
            # Prepare data for saving
            img_for_data = img[0]  # Remove batch dimension
            density_map_for_data = density_map_pred_pos[0, 0]  # Remove batch and channel dimensions
            
            # Normalize bboxes to [0,1] coordinates
            image_size = getattr(args, 'image_size', 512)
            normalized_pos_bboxes = pos_bboxes[0].cpu() / float(image_size)
            normalized_pred_bboxes = pred_boxes_tensor.cpu() / float(image_size) if len(pred_boxes_tensor) > 0 else torch.zeros((0, 4))
            
            # Save positive class data entry
            data_entry_pos = save_qualitative_data(
                img_for_data, normalized_pos_bboxes, density_map_for_data,
                normalized_pred_bboxes, pred_scores_tensor.cpu(),
                img_id, img_name, pos_count, int(pred_count_pos), mae_error_pos, 
                'positive', dataset_output_dir
            )
            all_results['class_results']['positive']['images'].append(data_entry_pos)
        
        # ===== NEGATIVE CLASS INFERENCE =====
        if len(negative_exemplars) >= 3:
            # negative_exemplars already has the right shape [3, 4] from processing
            neg_bboxes = negative_exemplars.unsqueeze(0).to(device)  # [1, 3, 4]
            # The coordinates are scaled to match the resized image
            
            # Run inference for negative class
            outputs_R, aux_outputs, tblr, pred_bboxes_neg = model(img, neg_bboxes)
            
            # Extract density map and predicted bboxes
            density_map_pred_neg = outputs_R
            
            # Extract bounding boxes and scores from DAVE's BoxList format
            if hasattr(pred_bboxes_neg, 'box'):
                pred_boxes_tensor = pred_bboxes_neg.box
                pred_scores_tensor = pred_bboxes_neg.fields.get('scores', torch.zeros(len(pred_boxes_tensor)))
            else:
                pred_boxes_tensor = torch.zeros((0, 4))
                pred_scores_tensor = torch.zeros((0,))
            
            # Calculate count and error for negative class
            pred_count_neg = len(pred_boxes_tensor) if len(pred_boxes_tensor) > 0 else float(density_map_pred_neg.sum().item())
            mae_error_neg = abs(neg_count - pred_count_neg)
            total_ae_neg += mae_error_neg
            
            # Prepare data for saving
            img_for_data = img[0]  # Remove batch dimension
            density_map_for_data = density_map_pred_neg[0, 0]  # Remove batch and channel dimensions
            
            # Normalize bboxes to [0,1] coordinates
            normalized_neg_bboxes = neg_bboxes[0].cpu() / float(image_size)
            normalized_pred_bboxes = pred_boxes_tensor.cpu() / float(image_size) if len(pred_boxes_tensor) > 0 else torch.zeros((0, 4))
            
            # Save negative class data entry
            data_entry_neg = save_qualitative_data(
                img_for_data, normalized_neg_bboxes, density_map_for_data,
                normalized_pred_bboxes, pred_scores_tensor.cpu(),
                img_id, img_name, neg_count, int(pred_count_neg), mae_error_neg, 
                'negative', dataset_output_dir
            )
            all_results['class_results']['negative']['images'].append(data_entry_neg)
        
        # Optional: limit number of processed images
        if hasattr(args, 'output_limit') and args.output_limit is not None and batch_idx >= args.output_limit:
            print(f"Stopping after {batch_idx + 1} images (output_limit={args.output_limit})...")
            break
    
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
    
    # Save individual class files
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
    
    # Save quantitative results (MAE, RMSE)
    save_quantitative_results(all_results, dataset_folder_name, args.model_name)
    
    print(f"\n=== Final Summary ===")
    print(f"Dataset folder: {dataset_folder_name}")
    print(f"Model: {args.model_name}")
    print(f"Qualitative results saved to: {dataset_output_dir}")
    print(f"Quantitative results saved to: ../../results/DAVE-quantitative/{dataset_folder_name}/")
    print(f"\nQualitative files created:")
    print(f"  - positive_qualitative_data.json")
    print(f"  - negative_qualitative_data.json") 
    print(f"  - complete_qualitative_data.json")
    print(f"  - complete_qualitative_data.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DAVE Qualitative Data Extraction', parents=[get_argparser()])
    parser.add_argument('--output_limit', type=int, default=None, 
                       help='Limit number of processed images (for testing)')
    parser.add_argument('--annotation_file', type=str, required=True,
                       help='Path to annotation JSON file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to images directory')
    args = parser.parse_args()
    
    print("Starting DAVE qualitative data extraction...")
    print(f"Model: {args.model_name}")
    print(f"Annotation file: {args.annotation_file}")
    print(f"Image directory: {args.image_dir}")
    
    evaluate_qualitative(args)
    print("\nDAVE qualitative data extraction completed!") 