#!/usr/bin/env python3
"""
LOCA evaluation script for DICTA25 - Combined positive/negative inference
Uses 2 positive + 1 negative exemplars to count all objects in a single inference pass.
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

from models.loca import build_model
from utils.arg_parser import get_argparser
from torchvision import transforms as T

class CombinedFSC147Dataset(Dataset):
    """Dataset that combines positive and negative exemplars for unified inference"""
    
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
        
        # LOCA model expects image_size x image_size images (typically 512x512)
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
        total_count = pos_count + neg_count  # Combined ground truth
        
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
            'total_count': total_count,
            'scaling_factors': (x_scale, y_scale)
        }

def save_density_map_files(density_map, img_name, output_dir):
    """Save density map as .npy file and create visualization image"""
    
    # Create subdirectories
    npy_dir = os.path.join(output_dir, 'density_maps_combined')
    img_dir = os.path.join(output_dir, 'density_maps_combined_images')
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

def save_qualitative_data(img_tensor, combined_bboxes, density_map_pred, img_id, img_name, 
                         pos_count, neg_count, total_count, pred_count, mae_error, output_dir):
    """Save qualitative data for combined inference"""
    
    # Save density map files
    density_npy_path, density_img_path = save_density_map_files(
        density_map_pred, img_name, output_dir
    )
    
    # Save original image
    original_img_path = save_original_image(img_tensor, img_name, output_dir)
    
    # Convert tensors to lists for JSON serialization
    combined_bboxes_list = combined_bboxes.cpu().numpy().tolist() if len(combined_bboxes) > 0 else []
    
    # Create data structure
    data_entry = {
        'image_id': int(img_id),
        'image_name': img_name,
        'gt_pos_count': int(pos_count),
        'gt_neg_count': int(neg_count),
        'gt_total_count': int(total_count),
        'pred_total_count': int(pred_count),
        'mae_error': float(mae_error),
        'combined_exemplar_boxes': combined_bboxes_list,  # 2 positive + 1 negative
        'density_map_sum': float(density_map_pred.sum().item()),
        'image_shape': list(img_tensor.shape),   # [C, H, W]
        'coordinate_format': 'normalized_xyxy',  # Format specification for exemplar boxes
        'file_paths': {
            'original_image': original_img_path,
            'density_map_npy': density_npy_path,
            'density_map_image': density_img_path
        },
        'notes': {
            'inference_type': 'combined_2pos_1neg',
            'exemplar_boxes_shown': len(combined_bboxes_list),
            'density_map_format': 'numpy_array_single_channel'
        }
    }
    
    return data_entry

def save_quantitative_results(all_results, dataset_folder_name, model_name):
    """Save quantitative metrics (MAE, RMSE) to separate folder"""
    
    # Create quantitative results directory
    quant_output_dir = "/home/khanhnguyen/DICTA25-RESULTS/LOCA-quantitative-combined"
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
            f.write(f"LOCA Combined Inference Results Summary\n")
            f.write(f"======================================\n\n")
            f.write(f"Dataset: {dataset_folder_name}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Inference Type: 2 Positive + 1 Negative Exemplars\n\n")
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
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build and load model
    print("Loading LOCA model...")
    model = build_model(args).to(device)
    
    # Load state dict
    state_dict_path = os.path.join(args.model_path, f'{args.model_name}.pt')
    print(f"Loading weights from: {state_dict_path}")
    
    checkpoint = torch.load(state_dict_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Remove 'module.' prefix if present (from DistributedDataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded successfully!")
    
    # Setup paths from command line arguments
    annotation_file = args.annotation_file
    image_dir = args.image_dir
    
    # Create dataset
    dataset = CombinedFSC147Dataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        image_size=getattr(args, 'image_size', 512),
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
    base_output_dir = "/home/khanhnguyen/DICTA25-RESULTS/LOCA-qualitative-combined"
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
            'image_size': getattr(args, 'image_size', 512),
            'annotation_file': annotation_file,
            'image_dir': image_dir,
            'backbone': args.backbone,
            'emb_dim': args.emb_dim,
            'num_heads': args.num_heads,
            'kernel_dim': args.kernel_dim,
            'reduction': args.reduction
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
        positive_exemplars = batch_data['positive_exemplars'][0]
        negative_exemplars = batch_data['negative_exemplars'][0]
        pos_count = batch_data['pos_count'].item()
        neg_count = batch_data['neg_count'].item()
        total_count = batch_data['total_count'].item()
        
        # ===== COMBINED INFERENCE: 2 POSITIVE + 1 NEGATIVE =====
        # Check if we have enough exemplars
        if len(positive_exemplars) >= 2 and len(negative_exemplars) >= 1:
            # Take first 2 positive and first 1 negative exemplars
            selected_pos = positive_exemplars[:2]  # [2, 4]
            selected_neg = negative_exemplars[:1]  # [1, 4]
            
            # Combine into single tensor: [2pos + 1neg, 4] = [3, 4]
            combined_exemplars = torch.cat([selected_pos, selected_neg], dim=0)  # [3, 4]
            combined_bboxes = combined_exemplars.unsqueeze(0).to(device)  # [1, 3, 4]
            
            # Run inference with combined exemplars
            density_map_pred, aux_outputs = model(img, combined_bboxes)
            
            # Calculate count and error against total ground truth
            pred_count = float(density_map_pred.sum().item())
            mae_error = abs(total_count - pred_count)
            total_ae += mae_error
            valid_images += 1
            
            # Prepare data for saving
            img_for_data = img[0]  # Remove batch dimension
            density_map_for_data = density_map_pred[0, 0]  # Remove batch and channel dimensions
            
            # Save combined inference data entry
            image_size = getattr(args, 'image_size', 512)
            normalized_combined_bboxes = combined_bboxes[0].cpu() / float(image_size)
            data_entry = save_qualitative_data(
                img_for_data, normalized_combined_bboxes, density_map_for_data,
                img_id, img_name, pos_count, neg_count, total_count, 
                int(pred_count), mae_error, dataset_output_dir
            )
            all_results['results']['images'].append(data_entry)
            
        else:
            print(f"Skipping {img_name}: insufficient exemplars (need 2+ positive, 1+ negative)")
        
        # Optional: limit number of processed images
        if hasattr(args, 'output_limit') and args.output_limit is not None and batch_idx >= args.output_limit:
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
    print(f"Quantitative results saved to: /home/khanhnguyen/DICTA25-RESULTS/LOCA-quantitative-combined/{dataset_folder_name}/")
    print(f"\nFiles created:")
    print(f"  - combined_inference_data.json")
    print(f"  - combined_inference_data.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LOCA Combined Inference', parents=[get_argparser()])
    parser.add_argument('--output_limit', type=int, default=None, 
                       help='Limit number of processed images (for testing)')
    parser.add_argument('--annotation_file', type=str, required=True,
                       help='Path to annotation JSON file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to images directory')
    args = parser.parse_args()
    
    print("Starting LOCA combined inference (2 positive + 1 negative exemplars)...")
    print(f"Model: {args.model_name}")
    print(f"Annotation file: {args.annotation_file}")
    print(f"Image directory: {args.image_dir}")
    
    evaluate_combined(args)
    print("\nLOCA combined inference completed!")
