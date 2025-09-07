#!/usr/bin/env python3
"""
LearningToCountEverything combined evaluation script for DICTA25
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
import copy
import torch.optim as optim

from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features
from utils import MincountLoss, PerturbationLoss
from torchvision import transforms as T

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
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        annotation = self.annotations[img_name]
        
        # Extract data from annotation
        positive_exemplars = annotation['box_examples_coordinates']
        negative_exemplars = annotation['negative_box_exemples_coordinates']
        
        # Convert to the format expected by LearningToCountEverything [y1, x1, y2, x2]
        pos_rects = []
        for bbox in positive_exemplars:
            x1, y1 = bbox[0][0], bbox[0][1]
            x2, y2 = bbox[2][0], bbox[2][1]
            pos_rects.append([y1, x1, y2, x2])
        
        neg_rects = []
        for bbox in negative_exemplars:
            x1, y1 = bbox[0][0], bbox[0][1]
            x2, y2 = bbox[2][0], bbox[2][1]
            neg_rects.append([y1, x1, y2, x2])
        
        # Get ground truth points
        positive_points = np.array(annotation['points'])
        negative_points = np.array(annotation['negative_points'])
        
        # Get counts
        pos_count = len(positive_points)
        neg_count = len(negative_points)
        total_count = pos_count + neg_count  # Combined ground truth
        
        # Return image path instead of PIL Image to avoid collate issues
        img_path = os.path.join(self.image_dir, img_name)
        
        return {
            'image_path': img_path,
            'image_name': img_name,
            'image_id': idx,
            'positive_exemplars': pos_rects,
            'negative_exemplars': neg_rects,
            'positive_points': positive_points,
            'negative_points': negative_points,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'total_count': total_count
        }

def custom_collate_fn(batch):
    """Custom collate function to handle nested lists for exemplars"""
    # Extract individual elements from batch
    image_paths = [item['image_path'] for item in batch]
    image_names = [item['image_name'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    positive_exemplars = [item['positive_exemplars'] for item in batch]
    negative_exemplars = [item['negative_exemplars'] for item in batch]
    positive_points = [item['positive_points'] for item in batch]
    negative_points = [item['negative_points'] for item in batch]
    pos_counts = [item['pos_count'] for item in batch]
    neg_counts = [item['neg_count'] for item in batch]
    total_counts = [item['total_count'] for item in batch]
    
    # Return as a batch dictionary without converting lists to tensors
    return {
        'image_path': image_paths,
        'image_name': image_names,
        'image_id': torch.tensor(image_ids),
        'positive_exemplars': positive_exemplars,
        'negative_exemplars': negative_exemplars,
        'positive_points': positive_points,
        'negative_points': negative_points,
        'pos_count': torch.tensor(pos_counts),
        'neg_count': torch.tensor(neg_counts),
        'total_count': torch.tensor(total_counts)
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

def save_qualitative_data(img_tensor, combined_boxes, density_map_pred, img_id, img_name, 
                         pos_count, neg_count, total_count, pred_count, mae_error, output_dir):
    """Save qualitative data for combined inference"""
    
    # Save density map files
    density_npy_path, density_img_path = save_density_map_files(
        density_map_pred, img_name, output_dir
    )
    
    # Save original image
    original_img_path = save_original_image(img_tensor, img_name, output_dir)
    
    # Convert boxes to lists for JSON serialization
    combined_boxes_list = combined_boxes if isinstance(combined_boxes, list) else combined_boxes.tolist()
    
    # Create data structure
    data_entry = {
        'image_id': int(img_id),
        'image_name': img_name,
        'gt_pos_count': int(pos_count),
        'gt_neg_count': int(neg_count),
        'gt_total_count': int(total_count),
        'pred_total_count': int(pred_count),
        'mae_error': float(mae_error),
        'combined_exemplar_boxes': combined_boxes_list,  # 2 positive + 1 negative, format: [y1, x1, y2, x2]
        'density_map_sum': float(density_map_pred.sum().item()),
        'image_shape': list(img_tensor.shape),   # [C, H, W]
        'coordinate_format': 'y1_x1_y2_x2',  # Format specification for boxes
        'file_paths': {
            'original_image': original_img_path,
            'density_map_npy': density_npy_path,
            'density_map_image': density_img_path
        },
        'notes': {
            'inference_type': 'combined_2pos_1neg',
            'exemplar_boxes_shown': len(combined_boxes_list),
            'density_map_format': 'numpy_array_single_channel',
            'model_outputs': 'density_maps_only'
        }
    }
    
    return data_entry

def save_quantitative_results(all_results, dataset_folder_name, model_name):
    """Save quantitative metrics (MAE, RMSE) to separate folder"""
    
    # Create quantitative results directory
    quant_output_dir = "/home/khanhnguyen/DICTA25-RESULTS/LearningToCountEverything-quantitative-combined"
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
            f.write(f"LearningToCountEverything Combined Inference Results Summary\n")
            f.write(f"==========================================================\n\n")
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
    print("Loading LearningToCountEverything model...")
    
    # Load feature extractor
    resnet50_conv = Resnet50FPN()
    if device.type == 'cuda':
        resnet50_conv.cuda()
    resnet50_conv.eval()
    
    # Load count regressor
    regressor = CountRegressor(6, pool='mean')
    regressor.load_state_dict(torch.load(args.model_path, map_location=device))
    if device.type == 'cuda':
        regressor.cuda()
    regressor.eval()
    
    print("Model loaded successfully!")
    
    # Setup paths
    annotation_file = args.annotation_file
    image_dir = args.image_dir
    
    # Create dataset
    dataset = CombinedFSC147Dataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        evaluation=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,  # LearningToCountEverything uses single-threaded processing
        collate_fn=custom_collate_fn  # Use the custom collate function
    )
    
    # Get dataset folder name - go up two levels to get actual dataset name (not "annotations")
    dataset_folder_name = os.path.basename(os.path.dirname(os.path.dirname(annotation_file)))
    
    # Create output directory structure
    base_output_dir = "/home/khanhnguyen/DICTA25-RESULTS/LearningToCountEverything-qualitative-combined"
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
            'annotation_file': annotation_file,
            'image_dir': image_dir,
            'adaptation_enabled': args.adapt,
            'gradient_steps': args.gradient_steps if args.adapt else 0,
            'learning_rate': args.learning_rate if args.adapt else 0,
            'weight_mincount': args.weight_mincount if args.adapt else 0,
            'weight_perturbation': args.weight_perturbation if args.adapt else 0
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
        img_name = batch_data['image_name'][0]
        img_id = batch_data['image_id'].item()
        positive_exemplars = batch_data['positive_exemplars'][0]
        negative_exemplars = batch_data['negative_exemplars'][0]
        pos_count = batch_data['pos_count'].item()
        neg_count = batch_data['neg_count'].item()
        total_count = batch_data['total_count'].item()
        
        # Load image from path
        img_path = batch_data['image_path'][0]
        image = Image.open(img_path).convert('RGB')
        image.load()
        
        # ===== COMBINED INFERENCE: 2 POSITIVE + 1 NEGATIVE =====
        # Check if we have enough exemplars
        if len(positive_exemplars) >= 2 and len(negative_exemplars) >= 1:
            print(f"Processing combined inference for {img_name}")
            
            # Combine exemplars: 2 positive + 1 negative
            selected_pos = positive_exemplars[:2]  # First 2 positive
            selected_neg = negative_exemplars[:1]   # First 1 negative
            combined_exemplars = selected_pos + selected_neg  # List concatenation
            
            # Prepare sample
            sample = {'image': image, 'lines_boxes': combined_exemplars}
            try:
                sample = Transform(sample)
                image_transformed, boxes_transformed = sample['image'], sample['boxes']
                
                if device.type == 'cuda':
                    image_transformed = image_transformed.cuda()
                    boxes_transformed = boxes_transformed.cuda()
                
                # Extract features
                features = extract_features(resnet50_conv, image_transformed.unsqueeze(0), boxes_transformed.unsqueeze(0), MAPS, Scales)
                
                if not args.adapt:
                    # Standard inference
                    output = regressor(features)
                else:
                    # Test-time adaptation
                    features.requires_grad = True
                    adapted_regressor = copy.deepcopy(regressor)
                    adapted_regressor.train()
                    optimizer = optim.Adam(adapted_regressor.parameters(), lr=args.learning_rate)
                    
                    for step in range(args.gradient_steps):
                        optimizer.zero_grad()
                        output = adapted_regressor(features)
                        lCount = args.weight_mincount * MincountLoss(output, boxes_transformed, device.type == 'cuda')
                        lPerturbation = args.weight_perturbation * PerturbationLoss(output, boxes_transformed, sigma=8, use_gpu=device.type == 'cuda')
                        Loss = lCount + lPerturbation
                        
                        # Perform gradient descent only for non-zero cases
                        if torch.is_tensor(Loss):
                            Loss.backward()
                            optimizer.step()
                    
                    features.requires_grad = False
                    output = adapted_regressor(features)
                
                # Calculate count and error against total ground truth
                pred_count = output.sum().item()
                mae_error = abs(total_count - pred_count)
                total_ae += mae_error
                valid_images += 1
                
                # Prepare data for saving
                density_map_for_data = output[0, 0]  # Remove batch and channel dimensions
                
                # Save combined inference data entry
                data_entry = save_qualitative_data(
                    image_transformed, combined_exemplars, density_map_for_data,
                    img_id, img_name, pos_count, neg_count, total_count, 
                    pred_count, mae_error, dataset_output_dir
                )
                all_results['results']['images'].append(data_entry)
                
            except Exception as e:
                print(f"Error processing combined inference for {img_name}: {e}")
                print(f"Combined exemplars: {combined_exemplars}")
                continue
                
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
    print(f"Quantitative results saved to: /home/khanhnguyen/DICTA25-RESULTS/LearningToCountEverything-quantitative-combined/{dataset_folder_name}/")
    print(f"\nFiles created:")
    print(f"  - combined_inference_data.json")
    print(f"  - combined_inference_data.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LearningToCountEverything Combined Inference')
    parser.add_argument('--annotation_file', type=str, required=True,
                       help='Path to annotation file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to image directory')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--model_name', type=str, default='LearningToCountEverything',
                       help='Model name for saving results')
    parser.add_argument('--adapt', action='store_true',
                       help='If specified, perform test time adaptation')
    parser.add_argument('--gradient_steps', type=int, default=100,
                       help='Number of gradient steps for the adaptation')
    parser.add_argument('--learning_rate', type=float, default=1e-7,
                       help='Learning rate for adaptation')
    parser.add_argument('--weight_mincount', type=float, default=1e-9,
                       help='Weight multiplier for Mincount Loss')
    parser.add_argument('--weight_perturbation', type=float, default=1e-4,
                       help='Weight multiplier for Perturbation Loss')
    parser.add_argument('--output_limit', type=int, default=None,
                       help='Limit number of processed images (for testing)')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU id. Default 0 for the first GPU. Use -1 for CPU.')
    
    args = parser.parse_args()
    
    # Set up GPU
    if not torch.cuda.is_available() or args.gpu_id < 0:
        print("===> Using CPU mode.")
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    print("Starting LearningToCountEverything combined inference (2 positive + 1 negative exemplars)...")
    print(f"Model: {args.model_name}")
    print(f"Annotation file: {args.annotation_file}")
    print(f"Image directory: {args.image_dir}")
    print(f"Test-time adaptation: {'Enabled' if args.adapt else 'Disabled'}")
    
    evaluate_combined(args)
    print("\nLearningToCountEverything combined inference completed!")
