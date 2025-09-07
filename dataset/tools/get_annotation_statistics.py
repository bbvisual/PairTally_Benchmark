#!/usr/bin/env python3
"""Compute statistics for DICTA25 annotation files.

Statistics reported:
1. Total number of distinct object classes across dataset.
2. For each super category, number of distinct (positive, negative) pairs.
3. Mean and median of ground-truth counts (positive, negative, total).

The script reuses helper parsing functions defined in convert_cvat_to_fsc147.py.
"""
import os
import argparse
import statistics
from collections import defaultdict
import xml.etree.ElementTree as ET
import re

from convert_cvat_to_fsc147 import (
    parse_filename_for_counts,
    parse_filename_for_prompts,
    parse_filename_for_super_category,
)


def gather_statistics(annotation_file: str):
    """Scan annotation XML for image filenames and compute statistics."""
    
    try:
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        all_files = [image.get('name') for image in root.findall('image')]
    except (ET.ParseError, FileNotFoundError) as e:
        raise RuntimeError(f"Error parsing XML file {annotation_file}: {e}")

    if not all_files:
        raise RuntimeError(f"No image tags found in {annotation_file}")

    # Containers
    unique_objects = set()
    supercat_pairs = defaultdict(set)
    pos_counts, neg_counts, total_counts = [], [], []

    for fname in all_files:
        if not fname or fname.startswith('.'):
            continue
        # Extract object codes (5th and 6th parts, 0-based index 4 and 5)
        parts = re.sub(r'\.(jpg|jpeg|png)$', '', fname).split('_')
        obj1_code = obj2_code = None
        if len(parts) >= 6:
            obj1_code = parts[4][:3]
            obj2_code = parts[5][:3]
            unique_objects.add(obj1_code)
            unique_objects.add(obj2_code)
        # Super category and pair (keep as before)
        super_cat = parse_filename_for_super_category(fname)
        pos_prompt, neg_prompt = parse_filename_for_prompts(fname)
        pair = tuple(sorted([pos_prompt, neg_prompt]))
        supercat_pairs[super_cat].add(pair)
        # Counts
        pos_cnt, neg_cnt = parse_filename_for_counts(fname)
        pos_counts.append(pos_cnt)
        neg_counts.append(neg_cnt)
        total_counts.append(pos_cnt + neg_cnt)

    stats = {
        "total_distinct_objects": len(unique_objects),
        "objects": sorted(unique_objects),
        "pairs_per_super_category": {k: len(v) for k, v in supercat_pairs.items()},
        "mean_pos_count": statistics.mean(pos_counts) if pos_counts else 0,
        "median_pos_count": statistics.median(pos_counts) if pos_counts else 0,
        "mean_neg_count": statistics.mean(neg_counts) if neg_counts else 0,
        "median_neg_count": statistics.median(neg_counts) if neg_counts else 0,
        "mean_total_count": statistics.mean(total_counts) if total_counts else 0,
        "median_total_count": statistics.median(total_counts) if total_counts else 0,
    }
    return stats, sorted(unique_objects)


def main():
    parser = argparse.ArgumentParser(description="Compute statistics for DICTA25 annotations")
    parser.add_argument("annotation_file", type=str, help="Path to annotation XML file (e.g., DICTA25/annotations_4.xml)")
    args = parser.parse_args()

    stats, object_list = gather_statistics(args.annotation_file)

    print("\n=== DICTA25 Annotation Statistics ===")
    print(f"Total distinct object classes (by code): {stats['total_distinct_objects']}")
    print("\nDistinct pairs per super category:")
    for cat, cnt in sorted(stats['pairs_per_super_category'].items()):
        print(f"  {cat}: {cnt} pairs")

    print("\nGround-truth count statistics:")
    print(f"  Positive counts  – mean: {stats['mean_pos_count']:.2f}, median: {stats['median_pos_count']}")
    print(f"  Negative counts  – mean: {stats['mean_neg_count']:.2f}, median: {stats['median_neg_count']}")
    print(f"  Total counts     – mean: {stats['mean_total_count']:.2f}, median: {stats['median_total_count']}")

    print("\nList of all distinct object codes:")
    print(", ".join(object_list))


if __name__ == "__main__":
    main() 