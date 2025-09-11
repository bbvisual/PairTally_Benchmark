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

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'GeCo'))

from models.geco_infer import build_model
from utils.data import resize_and_pad, xywh_to_x1y1x2y2
from utils.arg_parser import get_argparser
from torchvision import transforms as T
from torchvision.transforms import functional as TVF

class CustomFSC147Dataset(Dataset):
    """Custom dataset that loads from annotation file and handles positive/negative exemplars like GeCo"""
    
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
        
        # Pad to ensure we have exactly 3 boxes (GeCo expects this)
        if len(positive_exemplars) < 3:
            last_box = positive_exemplars[-1:] if len(positive_exemplars) > 0 else torch.zeros(1, 4)
            padding_needed = 3 - len(positive_exemplars)
            padding = last_box.repeat(padding_needed, 1)
            positive_exemplars = torch.cat([positive_exemplars, padding], dim=0)
        
        if len(negative_exemplars) < 3:
            last_box = negative_exemplars[-1:] if len(negative_exemplars) > 0 else torch.zeros(1, 4)
            padding_needed = 3 - len(negative_exemplars)
            padding = last_box.repeat(padding_needed, 1)
            negative_exemplars = torch.cat([negative_exemplars, padding], dim=0)
        
        # Apply GeCo evaluation preprocessing (need to pass dummy values to get full returns)
        dummy_density_map = torch.zeros(1, 1024, 1024)
        dummy_gt_bboxes = torch.zeros(0, 4)
        
        result_pos = resize_and_pad(
            img, positive_exemplars, 
            density_map=dummy_density_map, 
            gt_bboxes=dummy_gt_bboxes, 
            full_stretch=False if not self.zero_shot else True, 
            size=1024.0
        )
        
        result_neg = resize_and_pad(
            img, negative_exemplars,
            density_map=dummy_density_map,
            gt_bboxes=dummy_gt_bboxes,
            full_stretch=False if not self.zero_shot else True,
            size=1024.0
        )
        
        # Unpack results (with density_map and gt_bboxes, returns 6 values)
        img_pos, pos_bboxes, _, _, scaling_factor_pos, padwh_pos = result_pos
        img_neg, neg_bboxes, _, _, scaling_factor_neg, padwh_neg = result_neg
        
        # Check if bboxes are too small and need larger image size (like in GeCo evaluation)
        if (pos_bboxes[:, 2] - pos_bboxes[:, 0]).min() < 25 and (pos_bboxes[:, 3] - pos_bboxes[:, 1]).min() < 25 and not self.zero_shot:
            result_pos = resize_and_pad(
                img, positive_exemplars,
                density_map=dummy_density_map,
                gt_bboxes=dummy_gt_bboxes,
                full_stretch=False,
                size=1536.0
            )
            img_pos, pos_bboxes, _, _, scaling_factor_pos, padwh_pos = result_pos
        
        if (neg_bboxes[:, 2] - neg_bboxes[:, 0]).min() < 25 and (neg_bboxes[:, 3] - neg_bboxes[:, 1]).min() < 25 and not self.zero_shot:
            result_neg = resize_and_pad(
                img, negative_exemplars,
                density_map=dummy_density_map,
                gt_bboxes=dummy_gt_bboxes,
                full_stretch=False,
                size=1536.0
            )
            img_neg, neg_bboxes, _, _, scaling_factor_neg, padwh_neg = result_neg
        
        # Apply normalization like GeCo
        img_pos = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_pos)
        img_neg = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_neg)
        
        # Get ground truth counts from points
        pos_count = len(annotation['points'])  # Positive points
        neg_count = len(annotation['negative_points'])  # Negative points
        
        return {
            'image_pos': img_pos,
            'image_neg': img_neg,
            'image_name': img_name,
            'image_id': idx,
            'positive_exemplars': pos_bboxes,
            'negative_exemplars': neg_bboxes,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'padwh_pos': padwh_pos,
            'padwh_neg': padwh_neg,
            'scaling_factor_pos': scaling_factor_pos,
            'scaling_factor_neg': scaling_factor_neg
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

def save_qualitative_data(img_tensor, exemplar_bboxes, pred_bboxes, pred_scores, img_id, img_name, 
                         num_gt, num_pred, mae_error, class_type, output_dir):
    """Save qualitative data to structured format"""
    
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
        'image_shape': list(img_tensor.shape),   # [C, H, W]
        'coordinate_format': 'normalized_xyxy',  # Format specification
        'notes': {
            'exemplar_boxes_shown': min(3, len(exemplar_bboxes_list)),
            'total_exemplar_boxes': len(exemplar_bboxes_list),
            'total_predicted_boxes': len(pred_bboxes_list)
        }
    }
    
    return data_entry

def save_quantitative_results(all_results, dataset_folder_name, model_name):
    """Save quantitative metrics (MAE, RMSE) to separate folder"""
    
    # Create quantitative results directory
    quant_output_dir = "../../results/GeCo-quantitative"
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
        f.write(f"GeCo Quantitative Results Summary\n")
        f.write(f"================================\n\n")
        f.write(f"Dataset: {dataset_folder_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Image Size: {all_results['evaluation_info']['image_size']}\n\n")
        
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

@torch.no_grad()
def evaluate_qualitative(args):
    """Main evaluation function for saving qualitative data"""
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
    dataset = CustomFSC147Dataset(
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
    base_output_dir = "../../results/GeCo-qualitative"
    dataset_output_dir = os.path.join(base_output_dir, dataset_folder_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    print(f"Saving qualitative data to: {dataset_output_dir}")
    
    # Store all results
    all_results = {
        'dataset': dataset_folder_name,
        'model_name': args.model_name,
        'model_path': args.model_path,
        'evaluation_info': {
            'image_size': 1024,  # GeCo always uses 1024x1024 images
            'annotation_file': annotation_file,
            'image_dir': image_dir
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
        img_pos = batch_data['image_pos'].to(device)
        img_neg = batch_data['image_neg'].to(device)
        img_name = batch_data['image_name'][0]
        img_id = batch_data['image_id'].item()
        positive_exemplars = batch_data['positive_exemplars'][0].to(device)
        negative_exemplars = batch_data['negative_exemplars'][0].to(device)
        pos_count = batch_data['pos_count'].item()
        neg_count = batch_data['neg_count'].item()
        # Get padding info - handle the tensor format properly
        padwh_pos_raw = batch_data['padwh_pos'][0]
        padwh_neg_raw = batch_data['padwh_neg'][0]
        
        # Convert to tuple of ints
        if isinstance(padwh_pos_raw, torch.Tensor):
            padwh_pos = (padwh_pos_raw[0].item(), padwh_pos_raw[1].item()) if len(padwh_pos_raw) >= 2 else (padwh_pos_raw[0].item(), 0)
        else:
            padwh_pos = padwh_pos_raw
            
        if isinstance(padwh_neg_raw, torch.Tensor):
            padwh_neg = (padwh_neg_raw[0].item(), padwh_neg_raw[1].item()) if len(padwh_neg_raw) >= 2 else (padwh_neg_raw[0].item(), 0)
        else:
            padwh_neg = padwh_neg_raw
        
        # ===== POSITIVE CLASS INFERENCE =====
        if len(positive_exemplars) >= 3:
            # Run inference for positive class (imgs already have batch dimension from dataloader)
            pos_bboxes = positive_exemplars.unsqueeze(0)  # [1, 3, 4]
            outputs, ref_points, centerness, outputs_coord, masks = model(img_pos, pos_bboxes)
            
            # Post-process predictions
            nms_bboxes, nms_scores = postprocess_for_data_extraction(
                img_pos, pos_bboxes, outputs, padwh_pos, device
            )
            
            # Calculate error for positive class
            pred_count_pos = len(nms_bboxes[0])
            mae_error_pos = abs(pos_count - pred_count_pos)
            total_ae_pos += mae_error_pos
            
            # Prepare data for saving
            pred_bboxes_pos = nms_bboxes[0].cpu() if len(nms_bboxes[0]) > 0 else torch.zeros((0, 4))
            pred_scores_pos = nms_scores[0].cpu() if len(nms_scores[0]) > 0 else torch.zeros((0,))
            
            # Save positive class data entry (remove batch dimension for save function)
            normalized_pos_bboxes = pos_bboxes[0].cpu() / 1024.0
            data_entry_pos = save_qualitative_data(
                img_pos[0], normalized_pos_bboxes, pred_bboxes_pos, pred_scores_pos,
                img_id, img_name, pos_count, pred_count_pos, mae_error_pos, 
                'positive', dataset_output_dir
            )
            all_results['class_results']['positive']['images'].append(data_entry_pos)
        
        # ===== NEGATIVE CLASS INFERENCE =====
        if len(negative_exemplars) >= 3:
            # Run inference for negative class (imgs already have batch dimension from dataloader)
            neg_bboxes = negative_exemplars.unsqueeze(0)  # [1, 3, 4]
            outputs, ref_points, centerness, outputs_coord, masks = model(img_neg, neg_bboxes)
            
            # Post-process predictions
            nms_bboxes, nms_scores = postprocess_for_data_extraction(
                img_neg, neg_bboxes, outputs, padwh_neg, device
            )
            
            # Calculate error for negative class
            pred_count_neg = len(nms_bboxes[0])
            mae_error_neg = abs(neg_count - pred_count_neg)
            total_ae_neg += mae_error_neg
            
            # Prepare data for saving
            pred_bboxes_neg = nms_bboxes[0].cpu() if len(nms_bboxes[0]) > 0 else torch.zeros((0, 4))
            pred_scores_neg = nms_scores[0].cpu() if len(nms_scores[0]) > 0 else torch.zeros((0,))
            
            # Save negative class data entry (remove batch dimension for save function)
            normalized_neg_bboxes = neg_bboxes[0].cpu() / 1024.0
            data_entry_neg = save_qualitative_data(
                img_neg[0], normalized_neg_bboxes, pred_bboxes_neg, pred_scores_neg,
                img_id, img_name, neg_count, pred_count_neg, mae_error_neg, 
                'negative', dataset_output_dir
            )
            all_results['class_results']['negative']['images'].append(data_entry_neg)
        
        # Optional: limit number of processed images
        if args.output_limit is not None and batch_idx >= args.output_limit:
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
    print(f"Quantitative results saved to: ../../results/GeCo-quantitative/{dataset_folder_name}/")
    print(f"\nQualitative files created:")
    print(f"  - positive_qualitative_data.json")
    print(f"  - negative_qualitative_data.json") 
    print(f"  - complete_qualitative_data.json")
    print(f"  - complete_qualitative_data.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('GeCo Qualitative Data Extraction', parents=[get_argparser()])
    parser.add_argument('--output_limit', type=int, default=None, 
                       help='Limit number of processed images (for testing)')
    parser.add_argument('--annotation_file', type=str, required=True,
                       help='Path to annotation JSON file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to images directory')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Dataset name for output directory (will be inferred if not provided)')
    args = parser.parse_args()
    
    print("Starting qualitative data extraction...")
    print(f"Model: {args.model_name}")
    print(f"Annotation file: {args.annotation_file}")
    print(f"Image directory: {args.image_dir}")
    
    evaluate_qualitative(args)
    print("\nQualitative data extraction completed!") 