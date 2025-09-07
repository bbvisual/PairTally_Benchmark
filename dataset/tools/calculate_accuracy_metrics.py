#!/usr/bin/env python3
"""
Calculate MAE and RMSE metrics for DICTA25 Fine-Grained Counting Dataset

This script calculates Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
by comparing the actual object counts (from filenames) with the predicted counts 
(number of bounding boxes annotated).

Metrics are calculated for:
1. Overall dataset
2. By test type (INTER vs INTRA)
3. By super category (FOO, FUN, HOU, OFF, OTR)
4. For INTRA images: by fine-grained attributes (colour, size, texture/shape)
5. For INTRA images: by super category and attribute combinations
"""

import json
import argparse
import math
from collections import defaultdict
from tabulate import tabulate

def parse_filename_components(filename):
    """Parse filename to extract all components."""
    name_without_ext = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    parts = name_without_ext.split('_')
    
    components = {
        'obj1_name': None,
        'obj2_name': None,
        'test_type': None,
        'super_category': None,
        'pos_code': None,
        'neg_code': None,
        'pos_count': None,
        'neg_count': None,
        'id1': None,
        'id2': None
    }
    
    if len(parts) >= 10:
        try:
            components['obj1_name'] = parts[0]
            components['obj2_name'] = parts[1]
            components['test_type'] = parts[2].upper()
            components['super_category'] = parts[3].upper()
            components['pos_code'] = parts[4]
            components['neg_code'] = parts[5]
            components['pos_count'] = int(parts[6])
            components['neg_count'] = int(parts[7])
            components['id1'] = parts[8]
            components['id2'] = parts[9]
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not fully parse filename components: {filename}, error: {e}")
    
    return components

def get_base_object_code(object_code):
    """Extract base object code without suffix (e.g., COI1 -> COI, PEG2 -> PEG)"""
    import re
    match = re.match(r'^([A-Z]+)\d*$', object_code)
    if match:
        return match.group(1)
    return object_code

# Object code to fine-grained attribute mapping
OBJECT_ATTRIBUTES = {
    "FOO": {
        "PAS": "texture/shape", "RIC": "colour", "LIM": "size", "PEP": "colour",
        "TOM": "size", "CHI": "texture/shape", "PNT": "colour", "BEA": "colour",
        "SED": "texture/shape", "CFC": "colour", "ONI": "none", "CAN": "none", "GAR": "none"
    },
    "FUN": {
        "CHK": "colour", "MAH": "texture/shape", "LEG": "colour", "CHS": "colour",
        "PZP": "texture/shape", "PUZ": "texture/shape", "PKC": "colour", "PLC": "colour",
        "MAR": "size", "DIC": "colour", "CSC": "texture/shape"
    },
    "HOU": {
        "TPK": "texture/shape", "CTB": "texture/shape", "PIL": "colour", "BAT": "size",
        "HCP": "colour", "MNY": "colour", "COI": "size", "BOT": "texture/shape",
        "BBT": "texture/shape", "ULT": "texture/shape"
    },
    "OFF": {
        "PPN": "texture/shape", "HST": "size", "CRS": "colour", "RUB": "colour",
        "STN": "colour", "PPC": "colour", "PEN": "texture/shape", "PNC": "none",
        "RHS": "texture/shape", "ZPT": "size", "SFP": "size", "LPP": "none", "WWO": "none"
    },
    "OTR": {
        "SCR": "texture/shape", "BOL": "texture/shape", "NUT": "texture/shape",
        "WAS": "texture/shape", "BUT": "colour", "NAI": "texture/shape", "BEA": "colour",
        "IKC": "colour", "IKE": "colour", "PEG": "colour", "STO": "colour"
    }
}

def get_attribute_for_object_code(super_category, object_code):
    """Get the fine-grained attribute for an object code"""
    base_code = get_base_object_code(object_code)
    
    if super_category in OBJECT_ATTRIBUTES:
        if base_code in OBJECT_ATTRIBUTES[super_category]:
            return OBJECT_ATTRIBUTES[super_category][base_code]
    
    return "unknown"

def calculate_metrics(actual_values, predicted_values):
    """
    Calculate MAE and RMSE given lists of actual and predicted values.
    
    Args:
        actual_values: List of actual counts (from filenames)
        predicted_values: List of predicted counts (number of bounding boxes)
    
    Returns:
        tuple: (mae, rmse, count)
    """
    if len(actual_values) != len(predicted_values):
        raise ValueError("Actual and predicted value lists must have the same length")
    
    if len(actual_values) == 0:
        return 0.0, 0.0, 0
    
    # Calculate MAE
    absolute_errors = [abs(actual - predicted) for actual, predicted in zip(actual_values, predicted_values)]
    mae = sum(absolute_errors) / len(absolute_errors)
    
    # Calculate RMSE
    squared_errors = [(actual - predicted) ** 2 for actual, predicted in zip(actual_values, predicted_values)]
    mse = sum(squared_errors) / len(squared_errors)
    rmse = math.sqrt(mse)
    
    return mae, rmse, len(actual_values)

def analyze_accuracy_metrics(annotations_data):
    """
    Analyze accuracy metrics (MAE and RMSE) for different categories.
    """
    results = {
        'overall': {'pos': {'actual': [], 'predicted': []}, 'neg': {'actual': [], 'predicted': []}},
        'by_test_type': defaultdict(lambda: {'pos': {'actual': [], 'predicted': []}, 'neg': {'actual': [], 'predicted': []}}),
        'by_super_category': defaultdict(lambda: {'pos': {'actual': [], 'predicted': []}, 'neg': {'actual': [], 'predicted': []}}),
        'intra_by_attribute': defaultdict(lambda: {'pos': {'actual': [], 'predicted': []}, 'neg': {'actual': [], 'predicted': []}}),
        'intra_by_super_category_attribute': defaultdict(lambda: defaultdict(lambda: {'pos': {'actual': [], 'predicted': []}, 'neg': {'actual': [], 'predicted': []}})),
        'image_details': []
    }
    
    for filename, annotations in annotations_data.items():
        # Parse filename components
        components = parse_filename_components(filename)
        
        if not all(components[key] is not None for key in ['test_type', 'super_category', 'pos_code', 'neg_code']):
            print(f"Warning: Could not parse all required components from {filename}")
            continue
        
        test_type = components['test_type']
        super_category = components['super_category']
        pos_code = components['pos_code']
        neg_code = components['neg_code']
        
        # Actual counts from filename
        actual_pos_count = components['pos_count']
        actual_neg_count = components['neg_count']
        
        # Predicted counts from bounding boxes
        predicted_pos_count = len(annotations.get('pos', []))
        predicted_neg_count = len(annotations.get('neg', []))
        
        # Store values for overall metrics
        results['overall']['pos']['actual'].append(actual_pos_count)
        results['overall']['pos']['predicted'].append(predicted_pos_count)
        results['overall']['neg']['actual'].append(actual_neg_count)
        results['overall']['neg']['predicted'].append(predicted_neg_count)
        
        # Store values by test type
        results['by_test_type'][test_type]['pos']['actual'].append(actual_pos_count)
        results['by_test_type'][test_type]['pos']['predicted'].append(predicted_pos_count)
        results['by_test_type'][test_type]['neg']['actual'].append(actual_neg_count)
        results['by_test_type'][test_type]['neg']['predicted'].append(predicted_neg_count)
        
        # Store values by super category
        results['by_super_category'][super_category]['pos']['actual'].append(actual_pos_count)
        results['by_super_category'][super_category]['pos']['predicted'].append(predicted_pos_count)
        results['by_super_category'][super_category]['neg']['actual'].append(actual_neg_count)
        results['by_super_category'][super_category]['neg']['predicted'].append(predicted_neg_count)
        
        # Special analysis for INTRA images
        if test_type == 'INTRA':
            # Get attributes for positive and negative codes
            pos_attribute = get_attribute_for_object_code(super_category, pos_code)
            neg_attribute = get_attribute_for_object_code(super_category, neg_code)
            
            # Should be the same attribute for INTRA images
            if pos_attribute == neg_attribute and pos_attribute != "unknown":
                attribute = pos_attribute
                
                # Store values by attribute
                results['intra_by_attribute'][attribute]['pos']['actual'].append(actual_pos_count)
                results['intra_by_attribute'][attribute]['pos']['predicted'].append(predicted_pos_count)
                results['intra_by_attribute'][attribute]['neg']['actual'].append(actual_neg_count)
                results['intra_by_attribute'][attribute]['neg']['predicted'].append(predicted_neg_count)
                
                # Store values by super category and attribute
                results['intra_by_super_category_attribute'][super_category][attribute]['pos']['actual'].append(actual_pos_count)
                results['intra_by_super_category_attribute'][super_category][attribute]['pos']['predicted'].append(predicted_pos_count)
                results['intra_by_super_category_attribute'][super_category][attribute]['neg']['actual'].append(actual_neg_count)
                results['intra_by_super_category_attribute'][super_category][attribute]['neg']['predicted'].append(predicted_neg_count)
        
        # Store image details for verification
        results['image_details'].append({
            'filename': filename,
            'test_type': test_type,
            'super_category': super_category,
            'pos_code': pos_code,
            'neg_code': neg_code,
            'actual_pos_count': actual_pos_count,
            'actual_neg_count': actual_neg_count,
            'predicted_pos_count': predicted_pos_count,
            'predicted_neg_count': predicted_neg_count,
            'pos_error': abs(actual_pos_count - predicted_pos_count),
            'neg_error': abs(actual_neg_count - predicted_neg_count),
            'pos_attribute': get_attribute_for_object_code(super_category, pos_code) if test_type == 'INTRA' else None,
            'neg_attribute': get_attribute_for_object_code(super_category, neg_code) if test_type == 'INTRA' else None
        })
    
    return results

def calculate_and_print_metrics(data_dict, title):
    """Calculate and print MAE/RMSE metrics for a data dictionary."""
    print(f"\n{title}")
    print("-" * len(title))
    
    table_data = []
    
    for category, data in sorted(data_dict.items()):
        # Calculate metrics for positive objects
        pos_mae, pos_rmse, pos_count = calculate_metrics(
            data['pos']['actual'], 
            data['pos']['predicted']
        )
        
        # Calculate metrics for negative objects
        neg_mae, neg_rmse, neg_count = calculate_metrics(
            data['neg']['actual'], 
            data['neg']['predicted']
        )
        
        # Combined metrics (positive + negative)
        combined_actual = data['pos']['actual'] + data['neg']['actual']
        combined_predicted = data['pos']['predicted'] + data['neg']['predicted']
        combined_mae, combined_rmse, combined_count = calculate_metrics(
            combined_actual, 
            combined_predicted
        )
        
        table_data.append([
            category,
            f"{pos_count}",
            f"{pos_mae:.3f}",
            f"{pos_rmse:.3f}",
            f"{neg_mae:.3f}",
            f"{neg_rmse:.3f}",
            f"{combined_mae:.3f}",
            f"{combined_rmse:.3f}"
        ])
    
    headers = ["Category", "Images", "Pos MAE", "Pos RMSE", "Neg MAE", "Neg RMSE", "Combined MAE", "Combined RMSE"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def print_overall_metrics(results):
    """Print overall accuracy metrics."""
    print("=" * 80)
    print("ACCURACY METRICS (MAE & RMSE) - DICTA25 DATASET")
    print("=" * 80)
    
    # Overall metrics
    pos_mae, pos_rmse, pos_count = calculate_metrics(
        results['overall']['pos']['actual'],
        results['overall']['pos']['predicted']
    )
    
    neg_mae, neg_rmse, neg_count = calculate_metrics(
        results['overall']['neg']['actual'],
        results['overall']['neg']['predicted']
    )
    
    combined_actual = results['overall']['pos']['actual'] + results['overall']['neg']['actual']
    combined_predicted = results['overall']['pos']['predicted'] + results['overall']['neg']['predicted']
    combined_mae, combined_rmse, combined_count = calculate_metrics(combined_actual, combined_predicted)
    
    print(f"OVERALL METRICS:")
    print(f"  Total Images: {pos_count}")
    print(f"  Positive Objects - MAE: {pos_mae:.3f}, RMSE: {pos_rmse:.3f}")
    print(f"  Negative Objects - MAE: {neg_mae:.3f}, RMSE: {neg_rmse:.3f}")
    print(f"  Combined - MAE: {combined_mae:.3f}, RMSE: {combined_rmse:.3f}")

def print_error_distribution(results):
    """Print distribution of errors."""
    print(f"\nERROR DISTRIBUTION ANALYSIS:")
    print("-" * 30)
    
    all_pos_errors = [detail['pos_error'] for detail in results['image_details']]
    all_neg_errors = [detail['neg_error'] for detail in results['image_details']]
    
    print(f"Positive Object Errors:")
    print(f"  Perfect predictions (error = 0): {sum(1 for e in all_pos_errors if e == 0)} images")
    print(f"  Small errors (error <= 5): {sum(1 for e in all_pos_errors if e <= 5)} images")
    print(f"  Large errors (error > 10): {sum(1 for e in all_pos_errors if e > 10)} images")
    print(f"  Max error: {max(all_pos_errors)}")
    
    print(f"Negative Object Errors:")
    print(f"  Perfect predictions (error = 0): {sum(1 for e in all_neg_errors if e == 0)} images")
    print(f"  Small errors (error <= 5): {sum(1 for e in all_neg_errors if e <= 5)} images")
    print(f"  Large errors (error > 10): {sum(1 for e in all_neg_errors if e > 10)} images")
    print(f"  Max error: {max(all_neg_errors)}")

def main():
    parser = argparse.ArgumentParser(description='Calculate MAE and RMSE metrics for DICTA25 dataset')
    parser.add_argument('input_file', help='Input parsed annotations file (parsed_annotations.json)')
    parser.add_argument('--output_metrics', default='accuracy_metrics_detailed.json',
                       help='Output file for detailed metrics (JSON format)')
    parser.add_argument('--show_worst_predictions', type=int, default=10,
                       help='Show N worst predictions (highest errors)')
    
    args = parser.parse_args()
    
    # Load annotations
    print(f"Loading annotations from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        annotations_data = json.load(f)
    
    print(f"Found {len(annotations_data)} annotated images")
    
    # Analyze accuracy metrics
    print("Calculating accuracy metrics...")
    results = analyze_accuracy_metrics(annotations_data)
    
    # Print overall metrics
    print_overall_metrics(results)
    
    # Print error distribution
    print_error_distribution(results)
    
    # Print metrics by test type
    calculate_and_print_metrics(results['by_test_type'], "ACCURACY METRICS BY TEST TYPE")
    
    # Print metrics by super category
    calculate_and_print_metrics(results['by_super_category'], "ACCURACY METRICS BY SUPER CATEGORY")
    
    # Print metrics by INTRA attributes
    if results['intra_by_attribute']:
        calculate_and_print_metrics(results['intra_by_attribute'], "ACCURACY METRICS FOR INTRA IMAGES BY ATTRIBUTE")
    
    # Print metrics by INTRA super category and attribute
    if results['intra_by_super_category_attribute']:
        print(f"\nACCURACY METRICS FOR INTRA IMAGES BY SUPER CATEGORY AND ATTRIBUTE")
        print("-" * 65)
        
        for super_category in sorted(results['intra_by_super_category_attribute'].keys()):
            if results['intra_by_super_category_attribute'][super_category]:
                print(f"\n{super_category} Category:")
                
                table_data = []
                for attribute, data in sorted(results['intra_by_super_category_attribute'][super_category].items()):
                    # Calculate metrics
                    pos_mae, pos_rmse, pos_count = calculate_metrics(
                        data['pos']['actual'], 
                        data['pos']['predicted']
                    )
                    
                    neg_mae, neg_rmse, neg_count = calculate_metrics(
                        data['neg']['actual'], 
                        data['neg']['predicted']
                    )
                    
                    combined_actual = data['pos']['actual'] + data['neg']['actual']
                    combined_predicted = data['pos']['predicted'] + data['neg']['predicted']
                    combined_mae, combined_rmse, combined_count = calculate_metrics(
                        combined_actual, 
                        combined_predicted
                    )
                    
                    table_data.append([
                        attribute.title(),
                        f"{pos_count}",
                        f"{pos_mae:.3f}",
                        f"{pos_rmse:.3f}",
                        f"{neg_mae:.3f}",
                        f"{neg_rmse:.3f}",
                        f"{combined_mae:.3f}",
                        f"{combined_rmse:.3f}"
                    ])
                
                headers = ["Attribute", "Images", "Pos MAE", "Pos RMSE", "Neg MAE", "Neg RMSE", "Combined MAE", "Combined RMSE"]
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Show worst predictions
    if args.show_worst_predictions > 0:
        print(f"\nWORST {args.show_worst_predictions} PREDICTIONS (HIGHEST ERRORS)")
        print("-" * 50)
        
        # Sort by combined error (pos + neg error)
        sorted_details = sorted(results['image_details'], 
                              key=lambda x: x['pos_error'] + x['neg_error'], 
                              reverse=True)
        
        worst_data = []
        for detail in sorted_details[:args.show_worst_predictions]:
            worst_data.append([
                detail['filename'][:50] + "..." if len(detail['filename']) > 50 else detail['filename'],
                detail['test_type'],
                detail['super_category'],
                f"{detail['actual_pos_count']}→{detail['predicted_pos_count']}",
                f"{detail['actual_neg_count']}→{detail['predicted_neg_count']}",
                f"{detail['pos_error']}",
                f"{detail['neg_error']}",
                f"{detail['pos_error'] + detail['neg_error']}"
            ])
        
        headers = ["Filename", "Type", "Category", "Pos Act→Pred", "Neg Act→Pred", "Pos Err", "Neg Err", "Total Err"]
        print(tabulate(worst_data, headers=headers, tablefmt="grid"))
    
    # Save detailed metrics
    # Convert defaultdict to regular dict for JSON serialization
    def convert_defaultdict(d):
        if isinstance(d, defaultdict):
            d = dict(d)
        for k, v in d.items():
            if isinstance(v, defaultdict):
                d[k] = convert_defaultdict(v)
        return d
    
    serializable_results = convert_defaultdict(results)
    
    with open(args.output_metrics, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nDetailed metrics saved to: {args.output_metrics}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"This analysis compares:")
    print(f"  - Actual counts: extracted from filename (ground truth)")
    print(f"  - Predicted counts: number of annotated bounding boxes")
    print(f"Lower MAE and RMSE values indicate better annotation accuracy.")

if __name__ == "__main__":
    main()
