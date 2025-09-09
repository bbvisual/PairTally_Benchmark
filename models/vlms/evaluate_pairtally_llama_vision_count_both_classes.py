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


# Custom Dataset class for DICTA25 data (adapted for Llama Vision Combined)
class DICTA25CombinedDataset(Dataset):
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
            'combined_prompt': combined_prompt,
            'annotation': annotation
        }


def get_args_parser():
    parser = argparse.ArgumentParser("DICTA25 Llama Vision Combined Evaluation", add_help=False)
    
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


def run_llama_vision_combined_inference(model, processor, image, combined_prompt, args):
    """Run combined inference on a single image with Llama Vision"""
    
    # Create combined counting prompt - more direct and natural for LlamaVision
    counting_prompt = f"Count the total number of {combined_prompt} in this image. Respond with only a single number representing the total count of all objects combined. Examples: '15', '42', '0'. Do not include any other text, explanations, or formatting."
    
    print(f"    Llama Vision combined inference with prompt: '{counting_prompt}'")
    
    # Prepare messages
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": counting_prompt}
        ]}
    ]
    
    # Process inputs
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        images=image,
        text=input_text,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True if args.temperature > 0 else False
        )
    
    # Decode response
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0] if output_text else ""


def extract_combined_count_from_response(response_text):
    """Extract numerical count from model response with flexible parsing.
    
    Parsing logic for LlamaVision:
    1. Look for standalone numbers at the beginning/end of response
    2. Extract numbers from natural language patterns
    3. Handle various formats the model might use
    """
    # Clean the response
    original_response = response_text.strip()
    response_text = response_text.strip()
    
    # Method 1: Check if response is just a number
    if response_text.isdigit():
        count = int(response_text)
        parsing_info = f"Direct number response: {count}"
        return max(0, count), parsing_info
    
    # Method 2: Extract number from start of response (e.g., "15" from "15 objects")
    start_number_match = re.match(r'^(\d+)', response_text)
    if start_number_match:
        count = int(start_number_match.group(1))
        parsing_info = f"Number at start: {count} from '{response_text[:50]}...'"
        return max(0, count), parsing_info
    
    # Method 3: Look for "There are X objects" pattern
    there_are_pattern = r'(?:there are|there\'s|i (?:see|count|observe))\s*(\d+)'
    there_are_match = re.search(there_are_pattern, response_text.lower())
    if there_are_match:
        count = int(there_are_match.group(1))
        parsing_info = f"'There are X' pattern: {count}"
        return max(0, count), parsing_info
    
    # Method 4: Look for "total" with number
    total_pattern = r'total[:\s]*(\d+)'
    total_match = re.search(total_pattern, response_text.lower())
    if total_match:
        count = int(total_match.group(1))
        parsing_info = f"Total pattern: {count}"
        return max(0, count), parsing_info
    
    # Method 5: Look for standalone numbers in the response
    numbers = re.findall(r'\b(\d+)\b', response_text)
    if numbers:
        # If multiple numbers, try to pick the most likely count
        numbers = [int(n) for n in numbers]
        
        # Filter out very large numbers (likely not counts)
        reasonable_numbers = [n for n in numbers if n <= 10000]
        
        if reasonable_numbers:
            # Pick the largest reasonable number (often the total count)
            count = max(reasonable_numbers)
            parsing_info = f"Best number from response: {count} (from numbers: {numbers})"
            return max(0, count), parsing_info
    
    # Method 6: Legacy <count>N</count> format (in case some responses use it)
    count_pattern = r'<count>\s*(\d+)\s*</count>'
    count_matches = re.findall(count_pattern, response_text.lower())
    if count_matches:
        count = int(count_matches[-1])
        parsing_info = f"Explicit count tag: {count}"
        return max(0, count), parsing_info
    
    # Default: No number found
    count = 0
    parsing_info = f"No valid number found in response: '{original_response[:100]}...'"
    return count, parsing_info


def save_quantitative_results(all_results, dataset_folder_name, model_name):
    """Save quantitative metrics (MAE, RMSE) to separate folder matching CountGD pattern"""
    
    # Create quantitative results directory
    quant_output_dir = "../../results/LlamaVision-combined-quantitative"
    dataset_quant_dir = os.path.join(quant_output_dir, dataset_folder_name)
    os.makedirs(dataset_quant_dir, exist_ok=True)
    
    # Extract quantitative metrics
    quantitative_results = {
        'dataset': dataset_folder_name,
        'model_name': model_name,
        'evaluation_info': all_results['evaluation_info'],
        'combined_results': {}
    }
    
    combined_images = all_results.get('image_results', {})
    combined_count = len(combined_images)
    
    if combined_count > 0:
        # Calculate MAE and RMSE for combined evaluation
        mae_sum = sum(img_data['mae'] for img_data in combined_images.values())
        se_sum = sum(img_data['squared_error'] for img_data in combined_images.values())
        
        combined_mae = mae_sum / combined_count
        combined_rmse = np.sqrt(se_sum / combined_count)
        
        # Store combined metrics
        quantitative_results['combined_results'] = {
            'total_images': combined_count,
            'mae': float(combined_mae),
            'rmse': float(combined_rmse),
            'total_absolute_error': float(mae_sum),
            'total_squared_error': float(se_sum)
        }
        
        print(f"\nCombined Evaluation Results:")
        print(f"  MAE: {combined_mae:.2f}")
        print(f"  RMSE: {combined_rmse:.2f}")
        print(f"  Total images: {combined_count}")
    
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
        f.write(f"Llama Vision Combined Evaluation Results Summary\n")
        f.write(f"===============================================\n\n")
        f.write(f"Dataset: {dataset_folder_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Evaluation Type: Combined (counting both positive and negative objects)\n")
        f.write(f"Inference Mode: Combined count-only (Llama Vision)\n")
        f.write(f"Max New Tokens: {all_results['evaluation_info']['max_new_tokens']}\n")
        f.write(f"Temperature: {all_results['evaluation_info']['temperature']}\n\n")
        
        if 'combined_results' in quantitative_results:
            f.write(f"Combined Results (Positive + Negative Objects):\n")
            f.write(f"  MAE:  {quantitative_results['combined_results']['mae']:.2f}\n")
            f.write(f"  RMSE: {quantitative_results['combined_results']['rmse']:.2f}\n")
            f.write(f"  Total Images: {quantitative_results['combined_results']['total_images']}\n")
    
    print(f"\nQuantitative results saved to: {dataset_quant_dir}")
    print(f"Files created:")
    print(f"  - {model_name}_quantitative_results.json")
    print(f"  - {model_name}_quantitative_results.pkl")
    print(f"  - {model_name}_summary.txt")
    
    return quant_file


def main():
    parser = argparse.ArgumentParser("DICTA25 Llama Vision Combined Evaluation", parents=[get_args_parser()])
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
    
    print("Llama Vision DICTA25 Combined Evaluation")
    print("=======================================")
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
    
    # Create output directory structure following DICTA25-RESULTS pattern
    qual_output_dir = "../../results/LlamaVision-combined-qualitative"
    dataset_output_dir = os.path.join(qual_output_dir, dataset_folder_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    print(f"Saving combined qualitative data to: {dataset_output_dir}")
    
    # Load dataset
    print("Loading DICTA25 combined dataset...")
    dataset = DICTA25CombinedDataset(args.annotations_file, args.images_folder, transform=None)
    print(f"Loaded {len(dataset)} images")
    
    model_name = "LlamaVision-Combined"
    
    # Store all results in CountGD-compatible format
    all_results = {
        'model_name': model_name,
        'dataset_name': dataset_folder_name,
        'evaluation_info': {
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'annotation_file': args.annotations_file,
            'image_dir': args.images_folder,
            'model_id': args.model_id,
            'inference_mode': 'llama_vision_combined_count_only',
            'evaluation_type': 'combined_positive_and_negative'
        },
        'image_results': {}
    }
    
    mae_sum = 0
    se_sum = 0
    total_count = 0
    
    # Determine processing limit
    limit = len(dataset) if args.output_limit is None else min(args.output_limit, len(dataset))
    print(f"Processing {limit} images...")

    for i in range(limit):
        data = dataset[i]
        image = data['image']
        image_name = data['image_name']
        combined_prompt = data['combined_prompt']
        combined_gt_count = data['combined_gt_count']
        
        print(f"\nProcessing {i+1}/{limit}: {image_name}")
        print(f"  Combined GT count: {combined_gt_count} (pos: {data['pos_gt_count']}, neg: {data['neg_gt_count']})")
        print(f"  Combined prompt: '{combined_prompt}'")
        
        # ===== COMBINED INFERENCE =====
        combined_response = run_llama_vision_combined_inference(model, processor, image, combined_prompt, args)
        pred_combined_count, parsing_info = extract_combined_count_from_response(combined_response)
        
        # Calculate error for combined evaluation
        mae = abs(pred_combined_count - combined_gt_count)
        se = (pred_combined_count - combined_gt_count) ** 2
        
        mae_sum += mae
        se_sum += se
        total_count += 1
        
        print(f"  Predicted: {pred_combined_count}, MAE: {mae:.2f}")
        print(f"  Response: '{combined_response[:100]}...' -> {parsing_info}")
        
        # Store results
        all_results['image_results'][image_name] = {
            'predicted_count': int(pred_combined_count),
            'combined_gt_count': int(combined_gt_count),
            'pos_gt_count': int(data['pos_gt_count']),
            'neg_gt_count': int(data['neg_gt_count']),
            'mae': float(mae),
            'squared_error': float(se),
            'combined_prompt': combined_prompt,
            'inference_mode': 'llama_vision_combined_count_only',
            'llama_vision_response': {
                'raw_response': combined_response,
                'parsing_info': parsing_info,
                'full_prompt': f"Count the total number of {combined_prompt} in this image. Respond with only a single number representing the total count of all objects combined. Examples: '15', '42', '0'. Do not include any other text, explanations, or formatting.",
            },
            'notes': {
                'vision_language_model_combined_count': True,
                'no_bounding_boxes': True,
                'no_point_detection': True,
                'combined_evaluation': True
            }
        }
    
    # Calculate overall metrics
    overall_mae = mae_sum / total_count if total_count > 0 else 0
    overall_rmse = np.sqrt(se_sum / total_count) if total_count > 0 else 0
    
    print(f"\nCombined Evaluation Results:")
    print(f"  MAE: {overall_mae:.2f}")
    print(f"  RMSE: {overall_rmse:.2f}")
    print(f"  Total images: {total_count}")
    
    # Save detailed results
    combined_file = os.path.join(dataset_output_dir, f'{model_name}_detailed_results.json')
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Also save as pickle for easier Python loading
    pickle_file = os.path.join(dataset_output_dir, f'{model_name}_detailed_results.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Save quantitative results (MAE, RMSE) matching CountGD pattern
    save_quantitative_results(all_results, dataset_folder_name, model_name)
    
    print(f"\n=== Final Summary ===")
    print(f"Dataset folder: {dataset_folder_name}")
    print(f"Model: Llama Vision Combined")
    print(f"Combined qualitative results saved to: {dataset_output_dir}")
    print(f"Combined quantitative results saved to: ../../results/LlamaVision-combined-quantitative/{dataset_folder_name}/")
    print(f"\nQualitative files created:")
    print(f"  - {model_name}_detailed_results.json")
    print(f"  - {model_name}_detailed_results.pkl")
    print("\nLlama Vision combined evaluation completed successfully!")


if __name__ == "__main__":
    main()
