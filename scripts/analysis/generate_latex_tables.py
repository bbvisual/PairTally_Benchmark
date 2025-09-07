#!/usr/bin/env python3
"""
LaTeX Table Generator for DICTA25 Experimental Results
Generates professional LaTeX tables from quantitative experiment results.
"""

import json
import os
from pathlib import Path
import argparse
from collections import defaultdict
import math

class LatexTableGenerator:
    def __init__(self, results_dir="/home/khanhnguyen/DICTA25-RESULTS"):
        self.results_dir = Path(results_dir)
        
        # Model mappings and categories
        self.model_mapping = {
            # Class-Agnostic Models
            'DAVE': {
                'dir': 'DAVE-quantitative',
                'file_pattern': 'DAVE_3_shot_quantitative_results.json',
                'display_name': 'DAVE',
                'category': 'Class-Agnostic',
                'citation': r'\cite{Pelhan2024_DAVE}'
            },
            'GeCo': {
                'dir': 'GeCo-quantitative', 
                'file_pattern': 'GeCo_updated_quantitative_results.json',
                'display_name': 'GeCo',
                'category': 'Class-Agnostic',
                'citation': r'\cite{pelhan2024novel}'
            },
            'LOCA': {
                'dir': 'LOCA-quantitative',
                'file_pattern': 'loca_few_shot_quantitative_results.json', 
                'display_name': 'LoCA',
                'category': 'Class-Agnostic',
                'citation': r'\cite{somecitation}'
            },
            'LearningToCountEverything': {
                'dir': 'LearningToCountEverything-quantitative',
                'file_pattern': 'LearningToCountEverything_quantitative_results.json',
                'display_name': 'CounTR',
                'category': 'Class-Agnostic', 
                'citation': r'\cite{somecitation}'
            },
            
            # Open-World Models
            'CountGD': {
                'dir': 'CountGD-quantitative',
                'file_pattern': 'CountGD_quantitative_results.json',
                'display_name': 'Count GD',
                'category': 'Open-World',
                'citation': r'\cite{AminiNaieni2024}'
            },
            'CountGD-TextOnly': {
                'dir': 'CountGD-TextOnly-quantitative', 
                'file_pattern': 'CountGD-TextOnly_quantitative_results.json',
                'display_name': 'Count GD (Text)',
                'category': 'Open-World',
                'citation': r'\cite{AminiNaieni2024}'
            }
        }
        
        # Dataset mappings
        self.dataset_mapping = {
            'test_bbx_frames': 'Overall',
            'test_bbx_frames_inter': 'Inter',
            'test_bbx_frames_intra': 'Intra',
            'test_bbx_frames_off': 'OFF',
            'test_bbx_frames_hou': 'HOU', 
            'test_bbx_frames_foo': 'FOO',
            'test_bbx_frames_fun': 'FUN',
            'test_bbx_frames_otr': 'OTR'
        }
        
    def load_results(self, model_key, dataset):
        """Load quantitative results for a specific model and dataset"""
        model_info = self.model_mapping[model_key]
        results_path = self.results_dir / model_info['dir'] / dataset / model_info['file_pattern']
        
        if not results_path.exists():
            print(f"⚠️  Results not found: {results_path}")
            return None
            
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"❌ Error loading {model_key} results for {dataset}: {e}")
            return None
    
    def collect_all_results(self):
        """Collect results for all models and datasets"""
        all_results = defaultdict(dict)
        
        for model_key in self.model_mapping.keys():
            for dataset in self.dataset_mapping.keys():
                results = self.load_results(model_key, dataset)
                if results:
                    all_results[model_key][dataset] = results
                    
        return all_results
    
    def format_number(self, value, precision=2):
        """Format numbers for LaTeX tables"""
        if value is None or math.isnan(value):
            return "--"
        return f"{value:.{precision}f}"
    
    def generate_overall_table(self, all_results, dataset='test_bbx_frames'):
        """Generate overall performance table"""
        
        latex_table = r"""\subsection{Overall Performance}

\begin{table}[t]
\centering
\caption{Overall performance across all categories. Lower MAE and RMSE indicate better performance.}
\label{tab:overall_results}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Venue} & \textbf{MAE $\downarrow$} & \textbf{RMSE $\downarrow$} \\
\midrule
\multicolumn{4}{l}{\textit{Class-Agnostic}} \\
\midrule
"""
        
        # Class-Agnostic models
        for model_key, model_info in self.model_mapping.items():
            if model_info['category'] == 'Class-Agnostic':
                if model_key in all_results and dataset in all_results[model_key]:
                    results = all_results[model_key][dataset]['overall']
                    mae = self.format_number(results['mae'])
                    rmse = self.format_number(results['rmse'])
                else:
                    mae = rmse = "--"
                
                latex_table += f"{model_info['display_name']} {model_info['citation']} & -- & {mae} & {rmse} \\\\\n"
        
        # Open-World models
        latex_table += r"""\midrule
\multicolumn{4}{l}{\textit{Open-World}} \\
\midrule
"""
        
        for model_key, model_info in self.model_mapping.items():
            if model_info['category'] == 'Open-World':
                if model_key in all_results and dataset in all_results[model_key]:
                    results = all_results[model_key][dataset]['overall']
                    mae = self.format_number(results['mae'])
                    rmse = self.format_number(results['rmse'])
                else:
                    mae = rmse = "--"
                
                latex_table += f"{model_info['display_name']} {model_info['citation']} & -- & {mae} & {rmse} \\\\\n"
        
        # VLMs section (placeholder)
        latex_table += r"""\midrule
\multicolumn{4}{l}{\textit{VLMs}} \\
\midrule
GPT-4.5 \cite{somecitation} & -- & -- & -- \\
Qwen2.5-VL \cite{somecitation} & -- & -- & -- \\
LLaMA-3.2-11B \cite{somecitation} & -- & -- & -- \\
InternVL2-2B \cite{somecitation} & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
"""
        
        return latex_table
    
    def generate_inter_intra_table(self, all_results):
        """Generate inter/intra category comparison table"""
        
        latex_table = r"""\subsection{Inter vs Intra Category Performance}

\begin{table}[t]
\centering
\caption{Performance comparison on inter-category vs intra-category scenes. Lower MAE and RMSE indicate better performance.}
\label{tab:inter_intra_comparison}
\begin{tabular}{lcccc}
\toprule
\multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c}{\textbf{Inter-Category}} & \multicolumn{2}{c}{\textbf{Intra-Category}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
& \textbf{MAE $\downarrow$} & \textbf{RMSE $\downarrow$} & \textbf{MAE $\downarrow$} & \textbf{RMSE $\downarrow$} \\
\midrule
"""
        
        # Process all models for inter/intra comparison
        for model_key, model_info in self.model_mapping.items():
            # Get inter results
            inter_mae = inter_rmse = "--"
            if model_key in all_results and 'test_bbx_frames_inter' in all_results[model_key]:
                inter_data = all_results[model_key]['test_bbx_frames_inter']['overall']
                inter_mae = self.format_number(inter_data['mae'])
                inter_rmse = self.format_number(inter_data['rmse'])
            
            # Get intra results  
            intra_mae = intra_rmse = "--"
            if model_key in all_results and 'test_bbx_frames_intra' in all_results[model_key]:
                intra_data = all_results[model_key]['test_bbx_frames_intra']['overall']
                intra_mae = self.format_number(intra_data['mae'])
                intra_rmse = self.format_number(intra_data['rmse'])
            
            latex_table += f"{model_info['display_name']} {model_info['citation']} & {inter_mae} & {inter_rmse} & {intra_mae} & {intra_rmse} \\\\\n"
        
        latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        return latex_table
    
    def generate_inter_table(self, all_results):
        """Generate inter-category performance table"""
        
        latex_table = r"""\subsection{Inter-Category Performance}

\begin{table}[t]
\centering
\caption{Performance on inter-category scenes. Lower MAE and RMSE indicate better performance.}
\label{tab:inter_category_results}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Venue} & \textbf{MAE $\downarrow$} & \textbf{RMSE $\downarrow$} \\
\midrule
\multicolumn{4}{l}{\textit{Class-Agnostic}} \\
\midrule
"""
        
        # Class-Agnostic models
        for model_key, model_info in self.model_mapping.items():
            if model_info['category'] == 'Class-Agnostic':
                if model_key in all_results and 'test_bbx_frames_inter' in all_results[model_key]:
                    results = all_results[model_key]['test_bbx_frames_inter']['overall']
                    mae = self.format_number(results['mae'])
                    rmse = self.format_number(results['rmse'])
                else:
                    mae = rmse = "--"
                
                latex_table += f"{model_info['display_name']} {model_info['citation']} & -- & {mae} & {rmse} \\\\\n"
        
        # Open-World models
        latex_table += r"""\midrule
\multicolumn{4}{l}{\textit{Open-World}} \\
\midrule
"""
        
        for model_key, model_info in self.model_mapping.items():
            if model_info['category'] == 'Open-World':
                if model_key in all_results and 'test_bbx_frames_inter' in all_results[model_key]:
                    results = all_results[model_key]['test_bbx_frames_inter']['overall']
                    mae = self.format_number(results['mae'])
                    rmse = self.format_number(results['rmse'])
                else:
                    mae = rmse = "--"
                
                latex_table += f"{model_info['display_name']} {model_info['citation']} & -- & {mae} & {rmse} \\\\\n"
        
        # VLMs section (placeholder)
        latex_table += r"""\midrule
\multicolumn{4}{l}{\textit{VLMs}} \\
\midrule
GPT-4.5 \cite{somecitation} & -- & -- & -- \\
Qwen2.5-VL \cite{somecitation} & -- & -- & -- \\
LLaMA-3.2-11B \cite{somecitation} & -- & -- & -- \\
InternVL2-2B \cite{somecitation} & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
"""
        
        return latex_table
    
    def generate_intra_table(self, all_results):
        """Generate intra-category performance table"""
        
        latex_table = r"""\subsection{Intra-Category Performance}

\begin{table}[t]
\centering
\caption{Performance on intra-category scenes. Lower MAE and RMSE indicate better performance.}
\label{tab:intra_category_results}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Venue} & \textbf{MAE $\downarrow$} & \textbf{RMSE $\downarrow$} \\
\midrule
\multicolumn{4}{l}{\textit{Class-Agnostic}} \\
\midrule
"""
        
        # Class-Agnostic models
        for model_key, model_info in self.model_mapping.items():
            if model_info['category'] == 'Class-Agnostic':
                if model_key in all_results and 'test_bbx_frames_intra' in all_results[model_key]:
                    results = all_results[model_key]['test_bbx_frames_intra']['overall']
                    mae = self.format_number(results['mae'])
                    rmse = self.format_number(results['rmse'])
                else:
                    mae = rmse = "--"
                
                latex_table += f"{model_info['display_name']} {model_info['citation']} & -- & {mae} & {rmse} \\\\\n"
        
        # Open-World models
        latex_table += r"""\midrule
\multicolumn{4}{l}{\textit{Open-World}} \\
\midrule
"""
        
        for model_key, model_info in self.model_mapping.items():
            if model_info['category'] == 'Open-World':
                if model_key in all_results and 'test_bbx_frames_intra' in all_results[model_key]:
                    results = all_results[model_key]['test_bbx_frames_intra']['overall']
                    mae = self.format_number(results['mae'])
                    rmse = self.format_number(results['rmse'])
                else:
                    mae = rmse = "--"
                
                latex_table += f"{model_info['display_name']} {model_info['citation']} & -- & {mae} & {rmse} \\\\\n"
        
        # VLMs section (placeholder)
        latex_table += r"""\midrule
\multicolumn{4}{l}{\textit{VLMs}} \\
\midrule
GPT-4.5 \cite{somecitation} & -- & -- & -- \\
Qwen2.5-VL \cite{somecitation} & -- & -- & -- \\
LLaMA-3.2-11B \cite{somecitation} & -- & -- & -- \\
InternVL2-2B \cite{somecitation} & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
"""
        
        return latex_table
    
    def generate_super_categories_table(self, all_results):
        """Generate super categories performance table"""
        
        latex_table = r"""\subsection{Super Category Performance}

\begin{table*}[t]
\centering
\small
\caption{Performance comparison of models across super categories. Lower MAE and RMSE values indicate better performance.}
\label{tab:super_categories_results}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l@{\hspace{8pt}}cc@{\hspace{8pt}}cc@{\hspace{8pt}}cc@{\hspace{8pt}}cc@{\hspace{8pt}}cc}
\toprule
\multirow{2}{*}{\textbf{Model}} 
& \multicolumn{2}{c}{\textbf{OFF}} 
& \multicolumn{2}{c}{\textbf{HOU}} 
& \multicolumn{2}{c}{\textbf{FOO}} 
& \multicolumn{2}{c}{\textbf{FUN}} 
& \multicolumn{2}{c}{\textbf{OTR}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}
& \textbf{MAE $\downarrow$} & \textbf{RMSE $\downarrow$} 
& \textbf{MAE $\downarrow$} & \textbf{RMSE $\downarrow$} 
& \textbf{MAE $\downarrow$} & \textbf{RMSE $\downarrow$} 
& \textbf{MAE $\downarrow$} & \textbf{RMSE $\downarrow$} 
& \textbf{MAE $\downarrow$} & \textbf{RMSE $\downarrow$} \\
\midrule
"""
        
        super_cats = ['test_bbx_frames_off', 'test_bbx_frames_hou', 'test_bbx_frames_foo', 'test_bbx_frames_fun', 'test_bbx_frames_otr']
        
        for model_key, model_info in self.model_mapping.items():
            row = f"{model_info['display_name']} {model_info['citation']}"
            
            for cat_dataset in super_cats:
                mae = rmse = "--"
                if model_key in all_results and cat_dataset in all_results[model_key]:
                    cat_data = all_results[model_key][cat_dataset]['overall']
                    mae = self.format_number(cat_data['mae'])
                    rmse = self.format_number(cat_data['rmse'])
                
                row += f" & {mae} & {rmse}"
            
            row += " \\\\\n"
            latex_table += row
        
        latex_table += r"""\bottomrule
\end{tabular}
\end{table*}
"""
        
        return latex_table
    
    def generate_all_tables(self, output_file="latex_tables.tex"):
        """Generate all LaTeX tables and save to file"""
        print("🚀 Generating LaTeX Tables for DICTA25 Results\n")
        
        # Collect all results
        print("📊 Collecting experimental results...")
        all_results = self.collect_all_results()
        
        # Print summary
        print(f"✅ Found results for {len(all_results)} models:")
        for model_key, datasets in all_results.items():
            print(f"   {model_key}: {len(datasets)} datasets")
        
        print("\n📝 Generating LaTeX tables...\n")
        
        # Generate all tables
        overall_table = self.generate_overall_table(all_results)
        inter_table = self.generate_inter_table(all_results)
        intra_table = self.generate_intra_table(all_results)
        inter_intra_table = self.generate_inter_intra_table(all_results)
        super_cats_table = self.generate_super_categories_table(all_results)
        
        # Combine all tables
        full_latex = f"""% LaTeX Tables for DICTA25 Experimental Results
% Generated automatically from quantitative results

\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{multirow}}
\\usepackage{{array}}
\\usepackage{{geometry}}
\\geometry{{margin=0.8in}}

\\begin{{document}}

{overall_table}

{inter_table}

{intra_table}

{inter_intra_table}

{super_cats_table}

\\end{{document}}
"""
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(full_latex)
        
        print(f"💾 Saved complete LaTeX document: {output_file}")
        
        # Also save individual tables
        with open("table_overall.tex", 'w') as f:
            f.write(overall_table)
        
        with open("table_inter.tex", 'w') as f:
            f.write(inter_table)
            
        with open("table_intra.tex", 'w') as f:
            f.write(intra_table)
        
        with open("table_inter_intra_comparison.tex", 'w') as f:
            f.write(inter_intra_table)
            
        with open("table_super_categories.tex", 'w') as f:
            f.write(super_cats_table)
        
        print("📄 Individual table files saved:")
        print("   - table_overall.tex")
        print("   - table_inter.tex")
        print("   - table_intra.tex") 
        print("   - table_inter_intra_comparison.tex") 
        print("   - table_super_categories.tex")
        
        return all_results
    
    def print_results_summary(self, all_results):
        """Print a summary of collected results"""
        print("\n📈 Results Summary:")
        print("=" * 50)
        
        for model_key, datasets in all_results.items():
            print(f"\n{self.model_mapping[model_key]['display_name']}:")
            for dataset, data in datasets.items():
                mae = data['overall']['mae']
                rmse = data['overall']['rmse']
                print(f"  {self.dataset_mapping[dataset]}: MAE={mae:.2f}, RMSE={rmse:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables from DICTA25 experimental results')
    parser.add_argument('--results_dir', default='/home/khanhnguyen/DICTA25-RESULTS', 
                       help='Results directory path')
    parser.add_argument('--output', default='latex_tables.tex', 
                       help='Output LaTeX file name')
    parser.add_argument('--summary', action='store_true', 
                       help='Print results summary')
    
    args = parser.parse_args()
    
    # Create generator and run
    generator = LatexTableGenerator(args.results_dir)
    all_results = generator.generate_all_tables(args.output)
    
    if args.summary:
        generator.print_results_summary(all_results)
    
    print("\n✅ LaTeX table generation completed!")
    print(f"📖 Use the generated tables in your LaTeX document")

if __name__ == '__main__':
    main() 