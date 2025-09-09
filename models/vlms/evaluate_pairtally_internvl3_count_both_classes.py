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
import math
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoConfig
import torchvision.transforms as T


# Custom Dataset class for DICTA25 data (adapted for InternVL3 Combined)
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
            'image_path': image_path,
            'image_name': image_name,
            'pos_gt_count': pos_gt_count,
            'neg_gt_count': neg_gt_count,
            'combined_gt_count': combined_gt_count,
            'combined_prompt': combined_prompt,
            'annotation': annotation
        }


def get_args_parser():
    parser = argparse.ArgumentParser("DICTA25 InternVL3 Combined Evaluation", add_help=False)
    
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
        default="OpenGVLab/InternVL3-14B-Instruct",
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
        default="./InternVL_DICTA25_Results",
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
        "--max_new_tokens", default=1000, type=int, help="maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", default=0.1, type=float, help="temperature for generation"
    )

    return parser


def split_model(model_path):
    """Create optimal device map for InternVL3-38B across multiple GPUs"""
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    
    print(f"Creating device map for {num_layers} layers across {world_size} GPUs")
    
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    
    print(f"Layers per GPU: {num_layers_per_gpu}")
    
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    
    # Keep vision model and critical components on GPU 0
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        image = image_file.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def build_transform(input_size):
    """Build transform for InternVL"""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def load_internvl3_model(model_id, device):
    """Load InternVL model using AutoModel with optimal multi-GPU device mapping"""
    print(f"Loading model: {model_id}")
    
    try:
        # Load tokenizer using official configuration
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        
        # Create optimal device map for multi-GPU distribution
        device_map = split_model(model_id)
        print(f"Device map created with {len(device_map)} components distributed across GPUs")
        
        # First try: Use optimal InternVL3 configuration with flash attention
        try:
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map=device_map
            ).eval()
            print("Model loaded successfully with flash attention across multiple GPUs!")
            
        except Exception as flash_error:
            print(f"Flash attention failed: {flash_error}")
            print("Falling back to standard configuration...")
            
            # Fallback: Load without flash attention
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=device_map
            ).eval()
            print("Model loaded successfully with standard configuration across multiple GPUs!")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading {model_id}: {e}")
        raise e


def run_internvl3_combined_inference(model, tokenizer, image_path, combined_prompt, args):
    """Run combined inference on a single image with InternVL3 using the correct .chat() method"""
    
    # Create combined counting prompt - count both objects together
    question = f"""<image>
Count the total number of {combined_prompt} in this image.

INSTRUCTIONS:
1. Look carefully at the image and count all {combined_prompt} you can see
2. Provide only the number as your response (e.g., 15, 42, 0)
3. Do not include any explanation or additional text, just the number
4. Be precise and only count clearly visible {combined_prompt}

Task: How many {combined_prompt} are in this image?"""
    
    print(f"    InternVL3 combined inference with prompt: '{question}'")
    
    try:
        # Load and preprocess image using official InternVL3 method
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        
        # Place on GPU 0 to match vision model device (as per optimal device mapping)
        print(f"    Loaded pixel_values with shape: {pixel_values.shape}")
        
        # Create generation config (exact same as test script)
        generation_config = dict(
            max_new_tokens=args.max_new_tokens, 
            do_sample=False
        )
        
        # Use InternVL's chat method exactly as shown in the test script
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        
        return response
        
    except Exception as e:
        print(f"    InternVL chat method failed: {e}")
        raise e


def extract_combined_count_from_response(response_text):
    """Extract numerical count from model response with improved parsing.
    
    Parsing logic:
    1. Direct number response (e.g., "15", "42") - most common
    2. Number at start (e.g., "15 objects")
    3. "There are X" pattern
    4. "Total X" pattern  
    5. Any reasonable number in response
    6. <count>N</count> format (legacy)
    7. Default to 0 if no valid number found
    """
    # Clean the response
    original_response = response_text.strip()
    response_text = response_text.strip()
    
    # Method 1: Check if response is just a number
    if response_text.isdigit():
        count = int(response_text)
        parsing_info = f"Direct number response: {count}"
        return count, parsing_info
    
    # Method 2: Extract number from start of response (e.g., "15" from "15 objects")
    start_number_match = re.match(r'^(\d+)', response_text)
    if start_number_match:
        count = int(start_number_match.group(1))
        parsing_info = f"Number at start: {count}"
        return count, parsing_info
    
    # Method 3: Look for "There are X objects" pattern
    there_are_pattern = r'(?:there are|there\'s|i (?:see|count|observe))\s*(\d+)'
    there_are_match = re.search(there_are_pattern, response_text.lower())
    if there_are_match:
        count = int(there_are_match.group(1))
        parsing_info = f"'There are X' pattern: {count}"
        return count, parsing_info
    
    # Method 4: Look for "total" with number
    total_pattern = r'total[:\s]*(\d+)'
    total_match = re.search(total_pattern, response_text.lower())
    if total_match:
        count = int(total_match.group(1))
        parsing_info = f"Total pattern: {count}"
        return count, parsing_info
    
    # Method 5: Look for standalone numbers in the response
    numbers = re.findall(r'\b(\d+)\b', response_text)
    if numbers:
        # Filter reasonable numbers (avoid dates, IDs, etc.)
        reasonable_numbers = [int(n) for n in numbers if 0 <= int(n) <= 2000]
        if reasonable_numbers:
            count = reasonable_numbers[0]  # Take first reasonable number
            parsing_info = f"First reasonable number: {count} (from {numbers})"
            return count, parsing_info
    
    # Method 6: Legacy <count>N</count> format (in case some responses use it)
    count_pattern = r'<count>\s*(\d+)\s*</count>'
    count_matches = re.findall(count_pattern, response_text.lower())
    if count_matches:
        count = int(count_matches[-1])
        parsing_info = f"Explicit count tag: {count}"
        return count, parsing_info
    
    # Default: No number found
    count = 0
    parsing_info = f"No valid number found in response: '{original_response[:100]}...'"
    return count, parsing_info


def save_quantitative_results(all_results, dataset_folder_name, model_name):
    """Save quantitative metrics (MAE, RMSE) to separate folder matching CountGD pattern"""
    
    # Create quantitative results directory
    quant_output_dir = "../../results/InternVL3-combined-quantitative"
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
        f.write(f"InternVL3 Combined Evaluation Results Summary\n")
        f.write(f"===========================================\n\n")
        f.write(f"Dataset: {dataset_folder_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Evaluation Type: Combined (counting both positive and negative objects)\n")
        f.write(f"Inference Mode: Combined count-only (InternVL3)\n")
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
    parser = argparse.ArgumentParser("DICTA25 InternVL3 Combined Evaluation", parents=[get_args_parser()])
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
    
    print("InternVL3 DICTA25 Combined Evaluation")
    print("====================================")
    print(f"Dataset: {dataset_folder_name}")
    print(f"Model: {args.model_id}")
    print(f"Annotations file: {args.annotations_file}")
    print(f"Images folder: {args.images_folder}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Output limit: {args.output_limit}")
    
    # Load InternVL3 model
    print("Loading InternVL3 model...")
    model, tokenizer = load_internvl3_model(args.model_id, args.device)
    
    # Create output directory structure following DICTA25-RESULTS pattern
    qual_output_dir = "../../results/InternVL3-combined-qualitative"
    dataset_output_dir = os.path.join(qual_output_dir, dataset_folder_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    print(f"Saving combined qualitative data to: {dataset_output_dir}")
    
    # Load dataset
    print("Loading DICTA25 combined dataset...")
    dataset = DICTA25CombinedDataset(args.annotations_file, args.images_folder, transform=None)
    print(f"Loaded {len(dataset)} images")
    
    model_name = "InternVL3-Combined"
    
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
            'inference_mode': 'internvl3_combined_count_only',
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
        image_path = data['image_path']
        image_name = data['image_name']
        combined_prompt = data['combined_prompt']
        combined_gt_count = data['combined_gt_count']
        
        print(f"\nProcessing {i+1}/{limit}: {image_name}")
        print(f"  Combined GT count: {combined_gt_count} (pos: {data['pos_gt_count']}, neg: {data['neg_gt_count']})")
        print(f"  Combined prompt: '{combined_prompt}'")
        
        # ===== COMBINED INFERENCE =====
        combined_response = run_internvl3_combined_inference(model, tokenizer, image_path, combined_prompt, args)
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
            'inference_mode': 'internvl3_combined_count_only',
            'internvl3_response': {
                'raw_response': combined_response,
                'parsing_info': parsing_info,
                'full_prompt': f"Count the total number of {combined_prompt} in this image. Provide only the number as your response (e.g., 15, 42, 0). Do not include any explanation or additional text, just the number.",
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
    print(f"Model: InternVL3 Combined")
    print(f"Combined qualitative results saved to: {dataset_output_dir}")
    print(f"Combined quantitative results saved to: ../../results/InternVL3-combined-quantitative/{dataset_folder_name}/")
    print(f"\nQualitative files created:")
    print(f"  - {model_name}_detailed_results.json")
    print(f"  - {model_name}_detailed_results.pkl")
    print("\nInternVL3 combined evaluation completed successfully!")


if __name__ == "__main__":
    main()
