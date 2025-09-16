# PairTally Dataset Download & Setup Guide

This guide provides step-by-step instructions for downloading and setting up the PairTally dataset images.

## Quick Setup

### 1. Download Images from Google Drive
- **Link**: [https://drive.google.com/file/d/1TnenXS4yFicjo81NnmClfzgc8ltmmeBv/view](https://drive.google.com/file/d/1TnenXS4yFicjo81NnmClfzgc8ltmmeBv/view)
- **File**: `PairTally-Images-Only.zip` (~2.1GB)
- **Contents**: 681 high-resolution images for the PairTally benchmark

### 2. Extract and Setup
```bash
# Navigate to your repository root
cd PairTally-Benchmark-Release

# Extract the downloaded zip file
unzip PairTally-Images-Only.zip

# Move images to the correct location
mv PairTally-Images-Only/* dataset/pairtally_dataset/images/

# Clean up
rmdir PairTally-Images-Only
rm PairTally-Images-Only.zip  # optional
```

### 3. Verify Setup
```bash
# Run the verification script
cd dataset
python verify_dataset.py

# Expected output: "Ready to run evaluations!"
```

## What You'll Get

After successful setup, your dataset directory will contain:
- **681 images** in `dataset/pairtally_dataset/images/`
- **Annotation files** for both simple and augmented prompts
- **Dataset splits** for INTER/INTRA category evaluation
- **Metadata files** with comprehensive image information

## File Structure After Setup

```
dataset/pairtally_dataset/
├── annotations/
│   ├── pairtally_annotations_simple.json      # 681 images, simple prompts
│   ├── pairtally_splits_simple.json           # Dataset splits (simple)
│   ├── bbx_anno_valid.json                    # Original CVAT format
│   ├── parsed_annotations.json                # Subset CVAT format (legacy)
│   ├── image_metadata.json                    # Comprehensive metadata
└── images/                                     # 681 image files
    ├── HOU_INTER_BOT0_COI0_078_039_445b90.jpg
    ├── FOO_INTRA_PAS0_PAS1_045_041_8b2c1f.jpg
    └── ... (679 more images)
```
