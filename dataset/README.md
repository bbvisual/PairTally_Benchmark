# PairTally Dataset

This directory contains the **PairTally** benchmark dataset for evaluating fine-grained visual counting models on "what we mean, not what we see" scenarios.

## Dataset Overview

The PairTally dataset is specifically designed to evaluate AI models' ability to count objects in scenes with **object pairs**, testing both semantic understanding and fine-grained visual discrimination. Each image contains two object categories, requiring models to distinguish and count based on subtle differences in shape, size, color, or semantics.

### Key Statistics
- **Total Images**: 681 high-resolution images
- **Object Categories**: 54 distinct categories with 98 subcategories
- **Super Categories**: 5 (Food, Fun, Household, Office, Other)
- **Subcategory Pairs**: 97 total (47 INTRA-category + 50 INTER-category)
- **Test Types**: INTER-category and INTRA-category counting
- **Annotation Format**: Bounding boxes with object counts (3 exemplars per category)
- **Fine-grained Attributes**: Color (43.5%), Texture/Shape (42.5%), Size (14.1%)

## Directory Structure

```
dataset/
├── README.md                    # This file
├── pairtally_dataset/       # PairTally-compatible format (RECOMMENDED)
│   ├── annotations/
│   │   ├── pairtally_annotations_simple.json      # Main annotation file (681 images) - simple prompts
│   │   ├── pairtally_annotations_augmented.json   # Main annotation file (681 images) - detailed prompts
│   │   ├── pairtally_annotations_inter_simple.json # INTER-category only (368 images) - simple prompts
│   │   ├── pairtally_annotations_inter_augmented.json # INTER-category only (368 images) - detailed prompts  
│   │   ├── pairtally_annotations_intra_simple.json # INTRA-category only (313 images) - simple prompts
│   │   ├── pairtally_annotations_intra_augmented.json # INTRA-category only (313 images) - detailed prompts
│   │   ├── pairtally_splits_simple.json           # Dataset splits for simple format
│   │   ├── pairtally_splits_augmented.json        # Dataset splits for augmented format
│   │   ├── pairtally_splits_inter_simple.json     # INTER splits for simple format
│   │   ├── pairtally_splits_inter_augmented.json  # INTER splits for augmented format
│   │   ├── pairtally_splits_intra_simple.json     # INTRA splits for simple format
│   │   ├── pairtally_splits_intra_augmented.json  # INTRA splits for augmented format
│   │   ├── bbx_anno_valid.json                    # Original CVAT format (681 images)
│   │   ├── parsed_annotations.json                # Subset CVAT format (legacy)
│   │   ├── filename_mapping.json                  # Original to compact name mapping
│   │   └── image_metadata.json                    # Comprehensive image metadata
│   └── images/        # 681 images with compact filenames
├── annotations/                 # Original CVAT annotation files
│   ├── bbx_anno_valid.json     # Complete annotations (681 images) - CVAT format
│   ├── parsed_annotations.json  # Subset annotations (342 images) - CVAT format  
│   └── image_metadata.json     # Original image metadata and statistics
├── images/                     # Original images (if available from Google Drive)
└── tools/                     # Dataset processing tools
    ├── convert_cvat_to_fsc147.py      # Convert CVAT to FSC147 format
    ├── get_annotation_statistics.py   # Calculate dataset statistics
    └── verify_dataset.py             # Verify dataset integrity
```

**Note**: The `pairtally_dataset/` format is the recommended format for running evaluations as it's compatible with existing FSC147-based counting models.

## Annotation Formats

The PairTally dataset provides annotations in **three** formats:

1. **PairTally Simple Format** - **Recommended for most evaluations** - Simple, concise object descriptions
2. **PairTally Augmented Format** - Detailed, descriptive object descriptions for enhanced training
3. **Original CVAT Format** - For analysis and custom processing

### PairTally Simple Format (Recommended for Evaluation)

The simple format uses concise object descriptions like "pasta", "dice", "coin" for straightforward evaluation:

**Main Files:**
- `pairtally_annotations_simple.json`: Complete dataset (681 images) with simple prompts
- `pairtally_annotations_inter_simple.json`: INTER-category subset (368 images) 
- `pairtally_annotations_intra_simple.json`: INTRA-category subset (313 images)
- `pairtally_splits_simple.json`: Dataset splits for simple format
- `pairtally_splits_inter_simple.json`: INTER splits for simple format
- `pairtally_splits_intra_simple.json`: INTRA splits for simple format

### PairTally Augmented Format (Detailed Descriptions)

The augmented format uses detailed object descriptions like "Curved beige pieces with smooth surface and slight ridges" for enhanced model training:

**Main Files:**
- `pairtally_annotations_augmented.json`: Complete dataset (681 images) with detailed prompts
- `pairtally_annotations_inter_augmented.json`: INTER-category subset (368 images) 
- `pairtally_annotations_intra_augmented.json`: INTRA-category subset (313 images)
- `pairtally_splits_augmented.json`: Dataset splits for augmented format
- `pairtally_splits_inter_augmented.json`: INTER splits for augmented format
- `pairtally_splits_intra_augmented.json`: INTRA splits for augmented format

**Annotation Structure (per image):**
```json
{
  "image_name.jpg": {
    "points": [[x1, y1], [x2, y2], ...],           # Point annotations for positive objects
    "negative_points": [[x1, y1], [x2, y2], ...],  # Point annotations for negative objects  
    "box_examples_coordinates": [                    # Bounding boxes for positive exemplars
      [[x1,y1], [x2,y1], [x2,y2], [x1,y2]], ...
    ],
    "negative_box_exemples_coordinates": [           # Bounding boxes for negative exemplars
      [[x1,y1], [x2,y1], [x2,y2], [x1,y2]], ...
    ],
    "positive_prompt": "Description of positive objects",
    "negative_prompt": "Description of negative objects"
  }
}
```

**Compact Filenames:**
Images use compact filenames for efficiency: `{CATEGORY}_{TYPE}_{CODE1}_{CODE2}_{count1}_{count2}_{hash}.jpg`

Example: `FOO_INTER_LIM0_CHI0_035_038_0db2b3.jpg` 
- FOO = Food category
- INTER = Inter-category counting  
- LIM0 = Lime objects (35 count)
- CHI0 = Chili objects (38 count)

### Original CVAT Format

The `annotations/` directory contains the original annotations exported from CVAT:

**Files:**
- `bbx_anno_valid.json`: Complete dataset (681 images) in CVAT export format
- `parsed_annotations.json`: Subset (342 images) used in initial experiments  
- `image_metadata.json`: Comprehensive image metadata and statistics

**CVAT Annotation Structure (per image):**
```json
{
  "image_name.jpg": {
    "pos": [
      {"bbox": [x1, y1, x2, y2], "obj": "object_name", "attr": "attribute"},
      ...
    ],
    "neg": [
      {"bbox": [x1, y1, x2, y2], "obj": "object_name", "attr": "attribute"},
      ...
    ]
  }
}
```

**Key Features:**
- Original long-form image names with full object descriptions
- Bounding box coordinates in [x1, y1, x2, y2] format
- Object names and attributes as annotated in CVAT
- Separate positive and negative object annotations

### Converting Between Formats

Use the provided conversion tool to convert CVAT format to FSC147 format:

```bash
cd dataset/tools
python convert_cvat_to_fsc147.py ../annotations/bbx_anno_valid.json --output_dir ../final_dataset_recreated/annotations/
```

### Original File Naming Convention (CVAT Export)

Original images follow a structured naming convention:
```
{object1}_{object2}_{TYPE}_{CATEGORY}_{CODE1}_{CODE2}_{count1}_{count2}_{version}_{id}.jpg
```

Where:
- `object1`, `object2`: Object class names (e.g., "red-apple", "green-apple")
- `TYPE`: "INTER" (different classes) or "INTRA" (same class, different attributes)
- `CATEGORY`: Super category (FOO, FUN, HOU, OFF, OTR)
- `CODE1`, `CODE2`: 3-letter object codes
- `count1`, `count2`: Ground truth counts for each object
- `version`: Version number
- `id`: Unique image identifier

### Example Filenames

**INTER-class (different object types):**
```
tomato_chili_INTER_FOO_TOM0_CHI0_00050_00075_1_00998.jpg
```
- 50 tomatoes, 75 chilis, food category

**INTRA-class (same object, different attributes):**
```
red-apple_green-apple_INTRA_FOO_APP1_APP2_00023_00031_1_00156.jpg
```
- 23 red apples, 31 green apples, food category

## Super Categories

### Food
Food items, ingredients, and edible objects
- Examples: pasta (spiral vs penne), lime (citrus vs calamansi), peppercorn (black vs white), tomato (normal vs baby), chili (long vs short), peanut (with/without skin), beans, seeds, coffee candy, garlic, shallot

### Fun
Entertainment, games, toys, and recreational objects  
- Examples: checker pieces (black vs white), mahjong tiles (bamboo vs character), lego pieces (green vs pink), chess pieces (black vs white), puzzle pieces (edge vs center), poker chips (blue vs white), playing cards (red vs black), marbles (big vs small), dice (green vs white)

### Household
Common household items and tools
- Examples: toothpicks (straight vs plastic), cotton buds (wooden vs plastic), pills (white vs yellow), batteries (AAA vs AA), hair clippers (black vs brown), bills (1000 vs 5000 VND), coins (5¢ vs 10¢), bottle caps (beer vs plastic), shirt buttons (2 vs 4 holes), utensils (spoon vs fork)

### Office
Office supplies and equipment
- Examples: push pins (normal vs round), heart stickers (big vs small), craft sticks (red/orange vs blue/purple), rubber bands (yellow vs blue), sticky notes (green shades), paper clips (silver vs colored), pens (with/without cap), pencils, rhinestones (round vs star), zip ties (short vs long), safety pins (big vs small)

### Other
Miscellaneous objects not fitting other categories
- Examples: screws (bronze vs silver), bolts (hex vs mushroom), nuts (hex vs square), washers (metal vs nylon), beads (blue/pink shades), ikea clips (green vs red), pegs (grey vs white), stones (red vs yellow), novelty buttons (beige vs transparent)

## Test Types

### INTER-category Counting (50 pairs)
Counting objects of **different categories** in the same image.
- Challenge: Models must distinguish between semantically different objects
- Example: Counting tomatoes vs. chilis, mahjong tiles vs. poker chips

### INTRA-category Counting (47 pairs)
Counting objects of the **same category** with different fine-grained attributes.
- Challenge: Models must distinguish subtle differences within the same object class
- Attributes tested:
  - **Color (43.5%)**: red poker chips vs. blue poker chips, black vs. white checker pieces
  - **Size (14.1%)**: big marbles vs. small marbles, AAA vs. AA batteries
  - **Texture/Shape (42.5%)**: spiral pasta vs. penne pasta, hex nuts vs. square nuts

## Usage

### 1. Download Images

**Important**: The PairTally images are not included in this Git repository due to size constraints.

**Download from Google Drive**:
- **Download Link**: [https://drive.google.com/file/d/1TnenXS4yFicjo81NnmClfzgc8ltmmeBv/view](https://drive.google.com/file/d/1TnenXS4yFicjo81NnmClfzgc8ltmmeBv/view)
- File: `PairTally-Images-Only.zip` (contains all 681 images)

**Extract and Setup**:
```bash
# After downloading PairTally-Images-Only.zip:
unzip PairTally-Images-Only.zip

# Move images to the correct location:
mv PairTally-Images-Only/* pairtally_dataset/images/

# Verify setup
python verify_dataset.py
```

### 2. Convert Annotations
```bash
# Convert CVAT annotations to FSC147 format
python tools/convert_cvat_to_fsc147.py annotations/parsed_annotations.json
```

### 3. Calculate Statistics
```bash
# Get dataset statistics
python tools/get_annotation_statistics.py annotations/parsed_annotations.json

# Validate annotation accuracy
python tools/calculate_accuracy_metrics.py annotations/parsed_annotations.json
```

## Annotation Files

### parsed_annotations.json
Main annotation file in CVAT format containing bounding box coordinates for all objects.

```json
{
  "image_name.jpg": {
    "pos": [[x1, y1, x2, y2], ...],  # Bounding boxes for positive class
    "neg": [[x1, y1, x2, y2], ...]   # Bounding boxes for negative class
  }
}
```

### bbx_anno_valid.json
Validated bounding box annotations with quality checks.

### image_metadata.json
Metadata about images including object counts and categories.

## Quality Assurance

The dataset includes multiple quality assurance measures:
1. **Manual annotation validation**
2. **Cross-verification of object counts**
3. **Bounding box accuracy validation**
4. **Consistency checks across similar images**

## Evaluation Metrics

Models are evaluated using:
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual counts
- **Root Mean Square Error (RMSE)**: Square root of mean squared errors
- **Detection metrics**: Precision, Recall, F1-score for bounding box detection

## Citation

If you use the PairTally dataset, please cite:

```bibtex
@inproceedings{nguyen2025pairtally,
  title={Can Current AI Models Count What We Mean, Not What They See? 
         A Benchmark and Systematic Evaluation},
  author={Nguyen, Gia Khanh and Huang, Yifeng and Hoai, Minh},
  booktitle={Digital Image Computing: Techniques and Applications (DICTA)},
  year={2025}
}
```

## License

This dataset is released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license for academic research purposes.
