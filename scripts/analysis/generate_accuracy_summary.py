#!/usr/bin/env python3
"""
Generate clean summary tables for MAE and RMSE metrics.
"""

import json
import argparse
from tabulate import tabulate

def create_accuracy_summary_tables(metrics_file):
    """Create formatted accuracy summary tables."""
    
    with open(metrics_file, 'r') as f:
        results = json.load(f)
    
    def calculate_metrics_from_data(data):
        """Calculate MAE and RMSE from actual/predicted lists."""
        import math
        
        pos_actual = data['pos']['actual']
        pos_predicted = data['pos']['predicted']
        neg_actual = data['neg']['actual']
        neg_predicted = data['neg']['predicted']
        
        # Calculate positive metrics
        pos_mae = sum(abs(a - p) for a, p in zip(pos_actual, pos_predicted)) / len(pos_actual) if pos_actual else 0
        pos_rmse = math.sqrt(sum((a - p) ** 2 for a, p in zip(pos_actual, pos_predicted)) / len(pos_actual)) if pos_actual else 0
        
        # Calculate negative metrics
        neg_mae = sum(abs(a - p) for a, p in zip(neg_actual, neg_predicted)) / len(neg_actual) if neg_actual else 0
        neg_rmse = math.sqrt(sum((a - p) ** 2 for a, p in zip(neg_actual, neg_predicted)) / len(neg_actual)) if neg_actual else 0
        
        # Calculate combined metrics
        combined_actual = pos_actual + neg_actual
        combined_predicted = pos_predicted + neg_predicted
        combined_mae = sum(abs(a - p) for a, p in zip(combined_actual, combined_predicted)) / len(combined_actual) if combined_actual else 0
        combined_rmse = math.sqrt(sum((a - p) ** 2 for a, p in zip(combined_actual, combined_predicted)) / len(combined_actual)) if combined_actual else 0
        
        return {
            'images': len(pos_actual),
            'pos_mae': pos_mae,
            'pos_rmse': pos_rmse,
            'neg_mae': neg_mae,
            'neg_rmse': neg_rmse,
            'combined_mae': combined_mae,
            'combined_rmse': combined_rmse
        }
    
    print("=" * 100)
    print("DICTA25 DATASET - ANNOTATION ACCURACY METRICS (MAE & RMSE)")
    print("=" * 100)
    print("Comparison: Actual counts (from filenames) vs Predicted counts (bounding boxes)")
    print("Lower values indicate better annotation accuracy\n")
    
    # Table 1: Overall metrics
    overall_metrics = calculate_metrics_from_data(results['overall'])
    print("1. OVERALL ACCURACY")
    print("-" * 20)
    overall_data = [[
        "All Images",
        f"{overall_metrics['images']}",
        f"{overall_metrics['pos_mae']:.2f}",
        f"{overall_metrics['pos_rmse']:.2f}",
        f"{overall_metrics['neg_mae']:.2f}",
        f"{overall_metrics['neg_rmse']:.2f}",
        f"{overall_metrics['combined_mae']:.2f}",
        f"{overall_metrics['combined_rmse']:.2f}"
    ]]
    
    headers = ["Dataset", "Images", "Pos MAE", "Pos RMSE", "Neg MAE", "Neg RMSE", "Combined MAE", "Combined RMSE"]
    print(tabulate(overall_data, headers=headers, tablefmt="grid"))
    
    # Table 2: Test Type comparison
    print("\n2. ACCURACY BY TEST TYPE")
    print("-" * 30)
    test_data = []
    for test_type in ['INTER', 'INTRA']:
        if test_type in results['by_test_type']:
            metrics = calculate_metrics_from_data(results['by_test_type'][test_type])
            test_data.append([
                test_type,
                f"{metrics['images']}",
                f"{metrics['pos_mae']:.2f}",
                f"{metrics['pos_rmse']:.2f}",
                f"{metrics['neg_mae']:.2f}",
                f"{metrics['neg_rmse']:.2f}",
                f"{metrics['combined_mae']:.2f}",
                f"{metrics['combined_rmse']:.2f}"
            ])
    
    print(tabulate(test_data, headers=headers, tablefmt="grid"))
    
    # Table 3: Super Category comparison
    print("\n3. ACCURACY BY SUPER CATEGORY")
    print("-" * 35)
    category_data = []
    for category in ['FOO', 'FUN', 'HOU', 'OFF', 'OTR']:
        if category in results['by_super_category']:
            metrics = calculate_metrics_from_data(results['by_super_category'][category])
            category_data.append([
                category,
                f"{metrics['images']}",
                f"{metrics['pos_mae']:.2f}",
                f"{metrics['pos_rmse']:.2f}",
                f"{metrics['neg_mae']:.2f}",
                f"{metrics['neg_rmse']:.2f}",
                f"{metrics['combined_mae']:.2f}",
                f"{metrics['combined_rmse']:.2f}"
            ])
    
    print(tabulate(category_data, headers=headers, tablefmt="grid"))
    
    # Table 4: INTRA by attribute
    print("\n4. INTRA IMAGES: ACCURACY BY FINE-GRAINED ATTRIBUTE")
    print("-" * 55)
    if results['intra_by_attribute']:
        attr_data = []
        for attr in ['colour', 'size', 'texture/shape']:
            if attr in results['intra_by_attribute']:
                metrics = calculate_metrics_from_data(results['intra_by_attribute'][attr])
                attr_data.append([
                    attr.title(),
                    f"{metrics['images']}",
                    f"{metrics['pos_mae']:.2f}",
                    f"{metrics['pos_rmse']:.2f}",
                    f"{metrics['neg_mae']:.2f}",
                    f"{metrics['neg_rmse']:.2f}",
                    f"{metrics['combined_mae']:.2f}",
                    f"{metrics['combined_rmse']:.2f}"
                ])
        
        print(tabulate(attr_data, headers=headers, tablefmt="grid"))
    
    # Table 5: Compact summary for publication
    print("\n5. COMPACT SUMMARY TABLE (for publication)")
    print("-" * 45)
    compact_data = []
    
    # Add test types
    for test_type in ['INTER', 'INTRA']:
        if test_type in results['by_test_type']:
            metrics = calculate_metrics_from_data(results['by_test_type'][test_type])
            compact_data.append([
                test_type,
                f"{metrics['images']}",
                f"{metrics['combined_mae']:.1f}",
                f"{metrics['combined_rmse']:.1f}"
            ])
    
    compact_data.append(["", "", "", ""])  # Separator
    
    # Add super categories
    for category in ['FOO', 'FUN', 'HOU', 'OFF', 'OTR']:
        if category in results['by_super_category']:
            metrics = calculate_metrics_from_data(results['by_super_category'][category])
            compact_data.append([
                category,
                f"{metrics['images']}",
                f"{metrics['combined_mae']:.1f}",
                f"{metrics['combined_rmse']:.1f}"
            ])
    
    compact_data.append(["", "", "", ""])  # Separator
    
    # Add INTRA attributes
    if results['intra_by_attribute']:
        for attr in ['colour', 'size', 'texture/shape']:
            if attr in results['intra_by_attribute']:
                metrics = calculate_metrics_from_data(results['intra_by_attribute'][attr])
                compact_data.append([
                    f"INTRA-{attr.title()}",
                    f"{metrics['images']}",
                    f"{metrics['combined_mae']:.1f}",
                    f"{metrics['combined_rmse']:.1f}"
                ])
    
    compact_headers = ["Category", "Images", "MAE", "RMSE"]
    print(tabulate(compact_data, headers=compact_headers, tablefmt="grid"))
    
    # Table 6: LaTeX table for publication
    print("\n6. LATEX TABLE FOR PUBLICATION")
    print("-" * 35)
    
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Annotation Accuracy Metrics for DICTA25 Dataset}
\\label{tab:accuracy_metrics}
\\begin{tabular}{lrrr}
\\toprule
\\textbf{Category} & \\textbf{Images} & \\textbf{MAE} & \\textbf{RMSE} \\\\
\\midrule"""
    
    # Add test types
    for test_type in ['INTER', 'INTRA']:
        if test_type in results['by_test_type']:
            metrics = calculate_metrics_from_data(results['by_test_type'][test_type])
            latex_table += f"\n{test_type} & {metrics['images']} & {metrics['combined_mae']:.1f} & {metrics['combined_rmse']:.1f} \\\\"
    
    latex_table += "\n\\midrule"
    
    # Add super categories
    for category in ['FOO', 'FUN', 'HOU', 'OFF', 'OTR']:
        if category in results['by_super_category']:
            metrics = calculate_metrics_from_data(results['by_super_category'][category])
            latex_table += f"\n{category} & {metrics['images']} & {metrics['combined_mae']:.1f} & {metrics['combined_rmse']:.1f} \\\\"
    
    latex_table += "\n\\midrule"
    
    # Add INTRA attributes
    if results['intra_by_attribute']:
        for attr in ['colour', 'size', 'texture/shape']:
            if attr in results['intra_by_attribute']:
                metrics = calculate_metrics_from_data(results['intra_by_attribute'][attr])
                attr_name = attr.replace('texture/shape', 'texture')  # Shorten for table
                latex_table += f"\nINTRA-{attr_name.title()} & {metrics['images']} & {metrics['combined_mae']:.1f} & {metrics['combined_rmse']:.1f} \\\\"
    
    # Add overall
    overall_metrics = calculate_metrics_from_data(results['overall'])
    latex_table += f"""
\\midrule
\\textbf{{Overall}} & \\textbf{{{overall_metrics['images']}}} & \\textbf{{{overall_metrics['combined_mae']:.1f}}} & \\textbf{{{overall_metrics['combined_rmse']:.1f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    print(latex_table)
    
    # Key insights
    print("\n7. KEY INSIGHTS")
    print("-" * 15)
    
    # Find best and worst performing categories
    category_metrics = {}
    for category in ['FOO', 'FUN', 'HOU', 'OFF', 'OTR']:
        if category in results['by_super_category']:
            metrics = calculate_metrics_from_data(results['by_super_category'][category])
            category_metrics[category] = metrics['combined_mae']
    
    best_category = min(category_metrics.items(), key=lambda x: x[1])
    worst_category = max(category_metrics.items(), key=lambda x: x[1])
    
    # INTRA attribute performance
    if results['intra_by_attribute']:
        attr_metrics = {}
        for attr in ['colour', 'size', 'texture/shape']:
            if attr in results['intra_by_attribute']:
                metrics = calculate_metrics_from_data(results['intra_by_attribute'][attr])
                attr_metrics[attr] = metrics['combined_mae']
        
        best_attr = min(attr_metrics.items(), key=lambda x: x[1])
        worst_attr = max(attr_metrics.items(), key=lambda x: x[1])
        
        print(f"• Best performing super category: {best_category[0]} (MAE: {best_category[1]:.1f})")
        print(f"• Worst performing super category: {worst_category[0]} (MAE: {worst_category[1]:.1f})")
        print(f"• Best performing INTRA attribute: {best_attr[0]} (MAE: {best_attr[1]:.1f})")
        print(f"• Worst performing INTRA attribute: {worst_attr[0]} (MAE: {worst_attr[1]:.1f})")
    
    # Test type comparison
    if 'INTER' in results['by_test_type'] and 'INTRA' in results['by_test_type']:
        inter_metrics = calculate_metrics_from_data(results['by_test_type']['INTER'])
        intra_metrics = calculate_metrics_from_data(results['by_test_type']['INTRA'])
        
        if inter_metrics['combined_mae'] < intra_metrics['combined_mae']:
            print(f"• INTER images have better accuracy than INTRA images")
            print(f"  (INTER MAE: {inter_metrics['combined_mae']:.1f} vs INTRA MAE: {intra_metrics['combined_mae']:.1f})")
        else:
            print(f"• INTRA images have better accuracy than INTER images")
            print(f"  (INTRA MAE: {intra_metrics['combined_mae']:.1f} vs INTER MAE: {inter_metrics['combined_mae']:.1f})")


def main():
    parser = argparse.ArgumentParser(description='Generate accuracy summary tables')
    parser.add_argument('--metrics_file', default='accuracy_metrics_detailed.json',
                       help='Input metrics file (JSON format)')
    
    args = parser.parse_args()
    
    try:
        create_accuracy_summary_tables(args.metrics_file)
    except FileNotFoundError:
        print(f"Error: Metrics file '{args.metrics_file}' not found.")
        print("Please run the accuracy calculation script first:")
        print("python calculate_accuracy_metrics.py parsed_annotations.json")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
