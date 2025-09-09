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

# Expected output: "✅ Ready to run evaluations!"
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
│   ├── pairtally_annotations_augmented.json   # 681 images, detailed prompts
│   ├── pairtally_annotations_inter_simple.json     # INTER subset
│   ├── pairtally_annotations_intra_simple.json     # INTRA subset
│   ├── pairtally_splits_simple.json           # Dataset splits
│   ├── bbx_anno_valid.json                    # Original CVAT format
│   ├── image_metadata.json                    # Comprehensive metadata
│   └── filename_mapping.json                  # Original to compact mapping
└── images/                                     # 681 image files
    ├── HOU_INTER_BOT0_COI0_078_039_445b90.jpg
    ├── FOO_INTRA_PAS0_PAS1_045_041_8b2c1f.jpg
    └── ... (679 more images)
```

## Troubleshooting

### Common Issues

**"No images found" error:**
- Ensure you downloaded `PairTally-Images-Only.zip` from the correct Google Drive link
- Check that you extracted the zip file completely
- Verify images are in `dataset/pairtally_dataset/images/` directory

**"Permission denied" error:**
- Make sure you have write permissions in the repository directory
- Try using `sudo` if needed for file operations

**"Wrong number of images" warning:**
- Re-download the zip file if some images are missing
- Check that the extraction completed without errors

### Verification Script Output

The `verify_dataset.py` script checks:
- ✅ Directory structure exists
- ✅ All annotation files present (simple & augmented versions)
- ✅ Main annotation files load correctly (681 images)
- ✅ Image files present (681 .jpg files)
- ✅ INTER/INTRA subsets correct (368 + 313 = 681)
- ✅ CVAT format files available
- ✅ Sample annotations valid

## Next Steps

Once setup is complete:
1. **Environment Setup**: `./scripts/setup/setup_all_environments.sh`
2. **Run Evaluations**: `./scripts/evaluation/run_all_evaluations.sh`
3. **Generate Results**: `python scripts/analysis/generate_summary_tables.py`

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Run `python verify_dataset.py` for detailed diagnostics
3. See the main README.md for additional setup information
4. Contact the paper authors for dataset-specific questions
