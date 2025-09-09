#!/usr/bin/env python3
"""
Verify PairTally dataset structure and integrity
"""

import json
import os
from pathlib import Path

def verify_dataset():
    """Verify the PairTally dataset is properly set up"""
    
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / "pairtally_dataset"
    
    print("🔍 Verifying PairTally Dataset Structure...")
    print(f"Dataset directory: {dataset_dir}")
    
    # Check directory structure
    annotations_dir = dataset_dir / "annotations"
    images_dir = dataset_dir / "images"
    
    if not dataset_dir.exists():
        print("❌ ERROR: pairtally_dataset directory not found!")
        print("   Please download and extract the dataset first.")
        return False
    
    if not annotations_dir.exists():
        print("❌ ERROR: annotations directory not found!")
        return False
        
    if not images_dir.exists():
        print("❌ ERROR: images directory not found!")
        return False
    
    print("✅ Directory structure OK")
    
    # Check annotation files - we now have both simple and augmented versions
    required_simple_files = [
        "pairtally_annotations_simple.json",
        "pairtally_annotations_inter_simple.json", 
        "pairtally_annotations_intra_simple.json",
        "pairtally_splits_simple.json",
        "pairtally_splits_inter_simple.json",
        "pairtally_splits_intra_simple.json"
    ]
    
    required_augmented_files = [
        "pairtally_annotations_augmented.json",
        "pairtally_annotations_inter_augmented.json", 
        "pairtally_annotations_intra_augmented.json",
        "pairtally_splits_augmented.json",
        "pairtally_splits_inter_augmented.json",
        "pairtally_splits_intra_augmented.json"
    ]
    
    required_common_files = [
        "filename_mapping.json",
        "image_metadata.json"
    ]
    
    # Check both versions
    simple_missing = []
    augmented_missing = []
    common_missing = []
    
    for file in required_simple_files:
        if not (annotations_dir / file).exists():
            simple_missing.append(file)
    
    for file in required_augmented_files:
        if not (annotations_dir / file).exists():
            augmented_missing.append(file)
    
    for file in required_common_files:
        if not (annotations_dir / file).exists():
            common_missing.append(file)
    
    if common_missing:
        print(f"❌ ERROR: Missing common files: {common_missing}")
        return False
    
    simple_complete = len(simple_missing) == 0
    augmented_complete = len(augmented_missing) == 0
    
    if simple_complete and augmented_complete:
        print("✅ All annotation files present (both simple and augmented versions)")
    elif simple_complete:
        print("✅ Simple annotation files complete")
        print(f"⚠️  Missing augmented files: {augmented_missing}")
    elif augmented_complete:
        print("✅ Augmented annotation files complete")
        print(f"⚠️  Missing simple files: {simple_missing}")
    else:
        print(f"❌ ERROR: Missing files - Simple: {simple_missing}, Augmented: {augmented_missing}")
        return False
    
    # Check main annotation file (try simple first, then augmented)
    main_anno_file = None
    version_used = None
    if simple_complete:
        main_anno_file = annotations_dir / "pairtally_annotations_simple.json"
        version_used = "simple"
    elif augmented_complete:
        main_anno_file = annotations_dir / "pairtally_annotations_augmented.json"
        version_used = "augmented"
    
    if main_anno_file and main_anno_file.exists():
        try:
            with open(main_anno_file, 'r') as f:
                annotations = json.load(f)
            
            num_images_anno = len(annotations)
            print(f"✅ Main annotation file loaded ({version_used}): {num_images_anno} images")
            
            if num_images_anno != 681:
                print(f"⚠️  WARNING: Expected 681 images, found {num_images_anno}")
        
        except Exception as e:
            print(f"❌ ERROR: Could not load main annotation file: {e}")
            return False
    else:
        print("❌ ERROR: No main annotation file available")
        return False
    
    # Check images
    image_files = list(images_dir.glob("*.jpg"))
    num_images = len(image_files)
    
    if num_images == 0:
        print("❌ ERROR: No images found!")
        print("\n📥 DOWNLOAD REQUIRED:")
        print("   The PairTally images are not included in this repository.")
        print("   Please download them from Google Drive:")
        print("   https://drive.google.com/file/d/1TnenXS4yFicjo81NnmClfzgc8ltmmeBv/view")
        print("\n📋 Setup Instructions:")
        print("   1. Download PairTally-Images-Only.zip")
        print("   2. Extract: unzip PairTally-Images-Only.zip")
        print("   3. Move images: mv PairTally-Images-Only/* dataset/pairtally_dataset/images/")
        print("   4. Re-run: python verify_dataset.py")
        return False
    elif num_images != 681:
        print(f"⚠️  WARNING: Expected 681 images, found {num_images}")
        if num_images < 681:
            print("   Some images may be missing from your download.")
    else:
        print(f"✅ Found {num_images} image files")
    
    # Check if annotation and image counts match
    if num_images != num_images_anno:
        print(f"⚠️  WARNING: Annotation count ({num_images_anno}) != Image count ({num_images})")
    
    # Check subset files
    # Check INTER and INTRA subsets (use same version as main)
    if version_used == "simple":
        inter_file = annotations_dir / "pairtally_annotations_inter_simple.json"
        intra_file = annotations_dir / "pairtally_annotations_intra_simple.json"
    else:
        inter_file = annotations_dir / "pairtally_annotations_inter_augmented.json"
        intra_file = annotations_dir / "pairtally_annotations_intra_augmented.json"
    
    try:
        with open(inter_file, 'r') as f:
            inter_data = json.load(f)
        with open(intra_file, 'r') as f:
            intra_data = json.load(f)
            
        print(f"✅ INTER subset: {len(inter_data)} images")
        print(f"✅ INTRA subset: {len(intra_data)} images") 
        print(f"✅ Total subsets: {len(inter_data) + len(intra_data)} images")
        
        if len(inter_data) + len(intra_data) != num_images_anno:
            print("⚠️  WARNING: Subset counts don't add up to total")
            
    except Exception as e:
        print(f"❌ ERROR: Could not verify subset files: {e}")
        return False
    
    # Sample verification
    sample_image = list(annotations.keys())[0]
    sample_path = images_dir / sample_image
    
    if sample_path.exists():
        print(f"✅ Sample image verification: {sample_image} exists")
    else:
        print(f"❌ ERROR: Sample image missing: {sample_image}")
        return False
    
    # Verify annotation structure
    sample_anno = annotations[sample_image]
    required_keys = ['points', 'negative_points', 'box_examples_coordinates', 
                    'negative_box_exemples_coordinates', 'positive_prompt', 'negative_prompt']
    
    missing_keys = [key for key in required_keys if key not in sample_anno]
    if missing_keys:
        print(f"❌ ERROR: Sample annotation missing keys: {missing_keys}")
        return False
    
    print("✅ Sample annotation structure OK")
    
    # Check original CVAT annotations (now in same directory)
    cvat_files_found = 0
    print("\n📋 Checking original CVAT annotations...")
    
    cvat_files = ["bbx_anno_valid.json", "parsed_annotations.json", "image_metadata.json"]
    for cvat_file in cvat_files:
        if (annotations_dir / cvat_file).exists():
            cvat_files_found += 1
            try:
                with open(annotations_dir / cvat_file, 'r') as f:
                    cvat_data = json.load(f)
                if cvat_file == "bbx_anno_valid.json":
                    print(f"  ✅ {cvat_file}: {len(cvat_data)} images")
                elif cvat_file == "parsed_annotations.json":
                    print(f"  ✅ {cvat_file}: {len(cvat_data)} images")
                else:
                    print(f"  ✅ {cvat_file}: metadata available")
            except Exception as e:
                print(f"  ⚠️  {cvat_file}: could not load ({e})")
        else:
            print(f"  ❌ {cvat_file}: not found")
    
    print("\n🎉 SUCCESS: PairTally dataset verification complete!")
    print("\nDataset Summary:")
    print(f"  📁 Total images: {num_images}")
    print(f"  📋 PairTally annotations: {num_images_anno}")
    print(f"  🔄 INTER-category: {len(inter_data)}")
    print(f"  🔄 INTRA-category: {len(intra_data)}")
    print(f"  📋 CVAT files found: {cvat_files_found}/3")
    print(f"  📍 Location: {dataset_dir}")
    
    print("\n✅ Ready to run evaluations!")
    return True

if __name__ == "__main__":
    verify_dataset()
