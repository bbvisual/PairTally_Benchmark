import glob
import torch
from PIL import Image
import numpy as np
import argparse
import json
import os
import pickle
# Unused visualization imports removed - only saving results dictionaries
import scipy.ndimage as ndimage
from torch.utils.data import Dataset, DataLoader
import time
import re


# Custom Dataset class for PairTally data (adapted for Ovis2 Combined)
class PairTallyCombinedDataset(Dataset):
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
    parser = argparse.ArgumentParser("PairTally Ovis2 Combined Evaluation", add_help=False)
    
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
        default="AIDC-AI/Ovis2-16B",
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
        default="./Ovis2_PairTally_Results",
    )
    parser.add_argument(
        "--base_data_path",
        help="base path to PairTally data",
        default="../../../pairtally_dataset",
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
        "--max_new_tokens", default=50, type=int, help="maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", default=0.1, type=float, help="temperature for generation"
    )
    parser.add_argument(
        "--do_sample", action="store_true", help="enable sampling during generation"
    )

    return parser


def load_ovis2_model(model_id):
    """Load Ovis2 model using the official runner"""
    print(f"Loading model: {model_id}")
    
    try:
        # Import Ovis components
        from ovis.serve.runner import RunnerArguments, OvisRunner
        
        # Create runner arguments
        runner_args = RunnerArguments(
            model_path=model_id,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.1,
            max_partition=9
        )
        
        # Initialize the runner
        runner = OvisRunner(runner_args)
        
        print("Ovis2 model loaded successfully!")
        return runner
        
    except ImportError:
        print("Error: Ovis library not found. Please install it first.")
        print("pip install git+https://github.com/AIDC-AI/Ovis.git")
        raise
    except Exception as e:
        print(f"Error loading Ovis2 model: {e}")
        raise


def run_ovis2_combined_inference(runner, image_path, combined_prompt, args):
    """Run combined inference on a single image with Ovis2"""
    
    # Create unified counting prompt as specified in paper
    counting_prompt = f"Count the number of {combined_prompt} in this image. Provide only the total count in this format: <count>N</count>. If you see no {combined_prompt} or are unsure, respond with <count>0</count>."
    
    print(f"    Ovis2 combined inference with prompt: '{counting_prompt}'")
    
    try:
        # Load image (proper format for Ovis2)
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # Run inference using Ovis runner - proper format from working script
        response = runner.run([image, counting_prompt])
        
        # Extract the text output from the response
        response_text = response.get('output', '').strip() if response else ""
        
        return response_text
        
    except Exception as e:
        print(f"Error during Ovis2 inference: {e}")
        return ""


def extract_combined_count_from_response(response_text):
    """Extract numerical count from model response with strict parsing as specified in paper.
    
    Parsing logic:
    1. Extract ONLY the integer within <count>...</count> tags
    2. If no valid <count>N</count> format found, default to 0
    3. No other parsing methods allowed
    """
    # Clean the response
    original_response = response_text.strip()
    response_text = response_text.strip().lower()
    
    # Method 1: Extract explicit count in format <count>N</count>
    count_pattern = r'<count>\s*(\d+)\s*</count>'
    count_matches = re.findall(count_pattern, response_text)
    
    if count_matches:
        try:
            count = int(count_matches[-1])  # Use the last count if multiple
            parsing_info = f"Explicit count tag: {count}"
            return count, parsing_info
        except ValueError:
            pass
    
    # Default: No valid <count>N</count> format found
    count = 0
    parsing_info = f"No valid <count>N</count> format found in response: '{original_response[:100]}...'"
    return count, parsing_info


def save_quantitative_results(all_results, dataset_folder_name, model_name):
    """Save quantitative metrics (MAE, RMSE) to separate folder matching CountGD pattern"""
    
    # Create quantitative results directory
    quant_output_dir = "../../results/Ovis2-combined-quantitative"
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
        f.write(f"Ovis2 Combined Evaluation Results Summary\n")
        f.write(f"========================================\n\n")
        f.write(f"Dataset: {dataset_folder_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Evaluation Type: Combined (counting both positive and negative objects)\n")
        f.write(f"Inference Mode: Combined count-only (Ovis2)\n")
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
    parser = argparse.ArgumentParser("PairTally Ovis2 Combined Evaluation", parents=[get_args_parser()])
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
    
    print("Ovis2 PairTally Combined Evaluation")
    print("=====================================")
    print(f"Dataset: {dataset_folder_name}")
    print(f"Model: {args.model_id}")
    print(f"Annotations file: {args.annotations_file}")
    print(f"Images folder: {args.images_folder}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Output limit: {args.output_limit}")
    
    # Load Ovis2 model
    print("Loading Ovis2 model...")
    runner = load_ovis2_model(args.model_id)
    
    # Create output directory structure following PairTally-RESULTS pattern
    qual_output_dir = "../../results/Ovis2-combined-qualitative"
    dataset_output_dir = os.path.join(qual_output_dir, dataset_folder_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    print(f"Saving combined qualitative data to: {dataset_output_dir}")
    
    # Load dataset
    print("Loading PairTally combined dataset...")
    dataset = PairTallyCombinedDataset(args.annotations_file, args.images_folder, transform=None)
    print(f"Loaded {len(dataset)} images")
    
    model_name = "Ovis2-Combined"
    
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
            'inference_mode': 'ovis2_combined_count_only',
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
        combined_response = run_ovis2_combined_inference(runner, image_path, combined_prompt, args)
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
            'inference_mode': 'ovis2_combined_count_only',
            'ovis2_response': {
                'raw_response': combined_response,
                'parsing_info': parsing_info,
                'full_prompt': f"Count the number of {combined_prompt} in this image. Provide only the total count in this format: <count>N</count>. If you see no {combined_prompt} or are unsure, respond with <count>0</count>.",
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
    print(f"Model: Ovis2 Combined")
    print(f"Combined qualitative results saved to: {dataset_output_dir}")
    print(f"Combined quantitative results saved to: ../../results/Ovis2-combined-quantitative/{dataset_folder_name}/")
    print(f"\nQualitative files created:")
    print(f"  - {model_name}_detailed_results.json")
    print(f"  - {model_name}_detailed_results.pkl")
    print("\nOvis2 combined evaluation completed successfully!")


if __name__ == "__main__":
    main()
