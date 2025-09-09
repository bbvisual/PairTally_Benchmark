# PairTally Images Directory

This directory should contain the 681 PairTally dataset images.

## Download Instructions

The images are not included in this Git repository due to size constraints. Please download them from Google Drive:

**Download Link**: [https://drive.google.com/file/d/1TnenXS4yFicjo81NnmClfzgc8ltmmeBv/view](https://drive.google.com/file/d/1TnenXS4yFicjo81NnmClfzgc8ltmmeBv/view)

**Setup Instructions**:
1. Download `PairTally-Images-Only.zip` from the link above
2. Extract the zip file: `unzip PairTally-Images-Only.zip`
3. Move all images to this directory: `mv PairTally-Images-Only/* dataset/pairtally_dataset/images/`
4. Verify setup: `cd dataset && python verify_dataset.py`

## Expected Contents

After downloading and extracting, this directory should contain:
- 681 image files with compact filenames (e.g., `HOU_INTER_BOT0_COI0_078_039_445b90.jpg`)
- Images span 5 super-categories: Food, Fun, Household, Office, Other
- Each image contains two object categories for fine-grained counting evaluation

## File Naming Convention

Images use compact filenames with the format:
`{SUPERCATEGORY}_{TYPE}_{OBJ1}_{OBJ2}_{COUNT1}_{COUNT2}_{HASH}.jpg`

Where:
- `SUPERCATEGORY`: FOO (Food), FUN (Fun), HOU (Household), OFF (Office), OTH (Other)
- `TYPE`: INTER (different categories) or INTRA (same category, different attributes)
- `OBJ1`/`OBJ2`: Object type identifiers
- `COUNT1`/`COUNT2`: Object counts in the image
- `HASH`: Unique identifier

## Troubleshooting

If you encounter issues:
1. Ensure you have downloaded the correct file (`PairTally-Images-Only.zip`)
2. Check that all 681 images are present in this directory
3. Run `python verify_dataset.py` from the `dataset/` directory to validate setup
4. See the main README.md for additional setup instructions
