import glob
import torch
from PIL import Image
import numpy as np
import argparse
import json
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import scipy.ndimage as ndimage
from torch.utils.data import Dataset, DataLoader
import time
import re
from transformers import AutoTokenizer, AutoProcessor, MllamaForConditionalGeneration
from transformers import pipeline


# Custom Dataset class for DICTA25 data (same as CountGD)
class DICTA25Dataset(Dataset):
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
            'pos_gt_count': pos_gt_count,
            'neg_gt_count': neg_gt_count,
            'pos_prompt': pos_prompt,
            'neg_prompt': neg_prompt,
            'annotation': annotation
        }


def get_args_parser():
    parser = argparse.ArgumentParser("DICTA25 Llama Vision Evaluation", add_help=False)
    
    # dataset parameters
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")

    # training parameters
    parser.add_argument("--note", default="", help="add some notes to the experiment")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument(
        "--model_id",
        help="Hugging Face model ID",
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
    )
    parser.add_argument(
        "--annotations_file",
        help="path to DICTA25 annotations file",
        default="../test_bbx_frames/annotations/annotation_FSC147_384.json",
    )
    parser.add_argument(
        "--images_folder",
        help="path to DICTA25 images folder",
        default="../test_bbx_frames/images_384_VarV2",
    )
    parser.add_argument(
        "--output_dir",
        help="output directory for results",
        default="./Llama_DICTA25_Results",
    )
    parser.add_argument(
        "--base_data_path",
        help="base path to DICTA25 data",
        default="../../../DICTA25",
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
        "--output_limit", help="limit number of output images for testing", default=None, type=int
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")
    parser.add_argument(
        "--max_new_tokens", default=20, type=int, help="maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", default=0.1, type=float, help="temperature for generation"
    )

    return parser


def load_llama_vision_model(model_id, device):
    """Load Llama Vision model and processor"""
    print(f"Loading model: {model_id}")
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_id)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Model loaded successfully!")
    return model, processor


def run_llama_vision_inference(model, processor, image, prompt, args):
    """Run inference on a single image with Llama Vision"""
    
    # Create simplified counting prompt - count only, no pointing
    counting_prompt = f"Count the number of {prompt} in this image. Provide only the total count in this format: <count>N</count>. If you see no {prompt} or are unsure, respond with: <count>0</count>."
    
    print(f"    Llama Vision inference with simplified count-only prompt: '{counting_prompt}'")
    
    # Prepare inputs
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": counting_prompt}
        ]}
    ]
    
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode response
    response = processor.decode(output[0], skip_special_tokens=True)
    
    # Extract the generated part (after the input)
    generated_text = response.split(counting_prompt)[-1].strip()
    
    return generated_text


def extract_count_and_points_from_response(response_text):
    """Extract numerical count from model response with strict parsing.
    
    Parsing logic:
    1. <count>N</count> - use explicit count tag
    2. If no proper format detected, default to 0
    """
    # Clean the response
    original_response = response_text.strip()
    response_text = response_text.strip().lower()
    
    # Method 1: Extract explicit count in format <count>N</count>
    count_pattern = r'<count>\s*(\d+)\s*</count>'
    count_matches = re.findall(count_pattern, response_text)
    explicit_count = None
    if count_matches:
        try:
            explicit_count = int(count_matches[-1])  # Use the last count if multiple
        except ValueError:
            pass
    
    # Simplified parsing logic - only use explicit count format, otherwise default to 0
    if explicit_count is not None:
        # Case 1: Explicit count available
        count = explicit_count
        parsing_info = f"Explicit count: {count}"
    
    else:
        # Case 2: No proper format detected - default to 0
        count = 0
        parsing_info = "No proper <count>N</count> format detected, defaulting to 0"
    
    # Validation and final safety check
    if count < 0:
        count = 0
        parsing_info += " (negative count corrected to 0)"
    
    # Return count only (no points since we're not using them anymore)
    return count, [], parsing_info


def save_qualitative_data(img_id, img_name, num_gt, num_pred, mae_error, class_type, 
                         prompt, response_text, parsing_info, predicted_points):
    """Save qualitative data to structured format matching CountGD pattern"""
    
    # Create data structure (adapted for Llama Vision - count only, no points)
    data_entry = {
        'image_id': int(img_id),
        'image_name': img_name,
        'class_type': class_type,  # 'positive' or 'negative'
        'gt_count': int(num_gt),
        'pred_count': int(num_pred),
        'mae_error': float(mae_error),
        'predicted_boxes': [],     # Empty for Llama Vision (no box detection)
        'predicted_points': [],    # Empty for simplified version (no point detection)
        'prediction_scores': [],   # Empty for Llama Vision (no confidence scores)
        'prompt': prompt,
        'inference_mode': 'llama_vision_count_only',
        'coordinate_format': 'none',    # No coordinates in simplified version
        'llama_response': {
            'raw_response': response_text,
            'parsing_info': parsing_info,
            'full_prompt': f"Count the number of {prompt} in this image. Provide only the total count in this format: <count>N</count>. If you see no {prompt} or are unsure, respond with: <count>0</count>.",
            'extracted_points': []  # No points in simplified version
        },
        'notes': {
            'total_predicted_boxes': 0,
            'total_predicted_points': 0,
            'llama_vision_count_only': True,
            'no_bounding_boxes': True,
            'no_point_detection': True
        }
    }
    
    return data_entry


def save_quantitative_results(all_results, dataset_folder_name, model_name):
    """Save quantitative metrics (MAE, RMSE) to separate folder matching CountGD pattern"""
    
    # Create quantitative results directory
    quant_output_dir = "../../results/Llama-Vision-quantitative"
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
        f.write(f"Llama Vision Quantitative Results Summary\n")
        f.write(f"=====================================\n\n")
        f.write(f"Dataset: {dataset_folder_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Inference Mode: Count-Only (Llama Vision)\n")
        f.write(f"Max New Tokens: {all_results['evaluation_info']['max_new_tokens']}\n")
        f.write(f"Temperature: {all_results['evaluation_info']['temperature']}\n\n")
        
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
    parser = argparse.ArgumentParser("DICTA25 Llama Vision Evaluation", parents=[get_args_parser()])
    args = parser.parse_args()
    
    # If single_dataset is provided, use it; otherwise use dataset_name
    dataset_to_use = args.single_dataset if args.single_dataset else args.dataset_name
    
    # Update paths based on dataset
    if args.base_data_path and dataset_to_use:
        args.annotations_file = os.path.join(args.base_data_path, dataset_to_use, "annotations", "annotation_FSC147_384.json")
        args.images_folder = os.path.join(args.base_data_path, dataset_to_use, "images_384_VarV2")
    
    # Get dataset folder name for output organization
    if args.single_dataset:
        dataset_folder_name = args.single_dataset
    elif args.dataset_name:
        dataset_folder_name = args.dataset_name
    else:
        dataset_folder_name = os.path.basename(os.path.dirname(args.annotations_file))
    
    print("Llama Vision DICTA25 Evaluation")
    print("==============================")
    print(f"Dataset: {dataset_folder_name}")
    print(f"Model: {args.model_id}")
    print(f"Annotations file: {args.annotations_file}")
    print(f"Images folder: {args.images_folder}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Output limit: {args.output_limit}")
    
    # Load Llama Vision model
    print("Loading Llama Vision model...")
    model, processor = load_llama_vision_model(args.model_id, args.device)
    
    # Create output directory structure matching CountGD pattern
    base_output_dir = "../../results/Llama-Vision-qualitative"
    dataset_output_dir = os.path.join(base_output_dir, dataset_folder_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    print(f"Saving qualitative data to: {dataset_output_dir}")
    
    # Load dataset
    print("Loading DICTA25 dataset...")
    dataset = DICTA25Dataset(args.annotations_file, args.images_folder, transform=None)
    print(f"Loaded {len(dataset)} images")
    
    # Store all results in CountGD-compatible format
    all_results = {
        'dataset': dataset_folder_name,
        'model_name': 'Llama-Vision',
        'model_path': args.model_id,
        'evaluation_info': {
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'annotation_file': args.annotations_file,
            'image_dir': args.images_folder,
            'model_id': args.model_id,
            'inference_mode': 'llama_vision_count_only'
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
        
        # Ground truth counts from point annotations
        gt_pos_count = data['pos_gt_count']
        gt_neg_count = data['neg_gt_count']
        
        # ===== POSITIVE CLASS INFERENCE =====
        print(f"  Running positive inference with prompt: '{pos_prompt}'")
        pos_response = run_llama_vision_inference(model, processor, image, pos_prompt, args)
        pred_pos_count, pos_points, pos_parsing_info = extract_count_and_points_from_response(pos_response)
        
        # Calculate error for positive class
        pos_mae = abs(pred_pos_count - gt_pos_count)
        total_ae_pos += pos_mae
        
        # Save positive class data entry
        data_entry_pos = save_qualitative_data(
            i, image_name, gt_pos_count, pred_pos_count, pos_mae, 
            'positive', pos_prompt, pos_response, pos_parsing_info, []  # No points
        )
        all_results['class_results']['positive']['images'].append(data_entry_pos)
        
        print(f"  Positive: GT={gt_pos_count}, Pred={pred_pos_count}, MAE={pos_mae}")
        print(f"  Response: '{pos_response}' -> {pos_parsing_info}")
        
        # ===== NEGATIVE CLASS INFERENCE =====
        print(f"  Running negative inference with prompt: '{neg_prompt}'")
        neg_response = run_llama_vision_inference(model, processor, image, neg_prompt, args)
        pred_neg_count, neg_points, neg_parsing_info = extract_count_and_points_from_response(neg_response)
        
        # Calculate error for negative class
        neg_mae = abs(pred_neg_count - gt_neg_count)
        total_ae_neg += neg_mae
        
        # Save negative class data entry
        data_entry_neg = save_qualitative_data(
            i, image_name, gt_neg_count, pred_neg_count, neg_mae,
            'negative', neg_prompt, neg_response, neg_parsing_info, []  # No points
        )
        all_results['class_results']['negative']['images'].append(data_entry_neg)
        
        print(f"  Negative: GT={gt_neg_count}, Pred={pred_neg_count}, MAE={neg_mae}")
        print(f"  Response: '{neg_response}' -> {neg_parsing_info}")
    
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
    save_quantitative_results(all_results, dataset_folder_name, 'Llama-Vision')
    
    print(f"\n=== Final Summary ===")
    print(f"Dataset folder: {dataset_folder_name}")
    print(f"Model: Llama-Vision")
    print(f"Qualitative results saved to: {dataset_output_dir}")
    print(f"Quantitative results saved to: ../../results/Llama-Vision-quantitative/{dataset_folder_name}/")
    print(f"\nQualitative files created:")
    print(f"  - positive_qualitative_data.json")
    print(f"  - negative_qualitative_data.json") 
    print(f"  - complete_qualitative_data.json")
    print(f"  - complete_qualitative_data.pkl")
    print("\nLlama Vision evaluation completed successfully!")


if __name__ == "__main__":
    main() 