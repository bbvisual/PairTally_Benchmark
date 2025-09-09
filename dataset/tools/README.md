# Dataset Tools

This directory contains tools for processing and working with the PairTally dataset.

## Available Tools

### `verify_dataset.py`
**Purpose**: Verify the integrity and structure of the PairTally dataset

**Usage**:
```bash
cd dataset
python verify_dataset.py
```

**Features**:
- Checks directory structure
- Verifies annotation file integrity
- Counts images and annotations
- Validates sample annotations
- Checks both FSC147 and CVAT formats
- Provides detailed status report

### `convert_cvat_to_pairtally.py`
**Purpose**: Convert CVAT annotations to PairTally-compatible format

**Usage**:
```bash
python convert_cvat_to_pairtally.py input_file.json [options]
```

**Options**:
- `--output_dir`: Output directory for converted files
- `--annotation_file`: Output annotation filename (default: pairtally_annotations.json)
- `--mapping_strategy`: Filename mapping strategy (default: compact)
- `--create_super_category_files`: Create per-category annotation files

**Example**:
```bash
python convert_cvat_to_pairtally.py ../annotations/bbx_anno_valid.json \
    --output_dir ../converted_fsc147/annotations/ \
    --mapping_strategy compact
```

**Output Files**:
- `pairtally_annotations.json` - Main FSC147-format annotations
- `pairtally_annotations_inter.json` - INTER-category subset
- `pairtally_annotations_intra.json` - INTRA-category subset
- `Train_Test_Val_FSC_147.json` - Dataset splits
- `filename_mapping.json` - Original to compact filename mapping
- `image_metadata.json` - Comprehensive image metadata

### `get_annotation_statistics.py` (if available)
**Purpose**: Calculate detailed statistics about the dataset annotations

**Usage**:
```bash
python get_annotation_statistics.py annotation_file.json
```

**Features**:
- Object count distributions
- Category breakdowns
- Attribute analysis (color, size, texture/shape)
- INTER vs INTRA category statistics

## Format Conversion

### CVAT to FSC147 Conversion

The conversion process transforms CVAT export format to PairTally-compatible format:

**CVAT Format** (input):
```json
{
  "long_descriptive_filename.jpg": {
    "pos": [{"bbox": [x1,y1,x2,y2], "obj": "object_name", "attr": "attribute"}],
    "neg": [{"bbox": [x1,y1,x2,y2], "obj": "object_name", "attr": "attribute"}]
  }
}
```

**FSC147 Format** (output):
```json
{
  "CATEGORY_TYPE_CODE1_CODE2_count1_count2_hash.jpg": {
    "points": [[x,y], ...],
    "negative_points": [[x,y], ...],
    "box_examples_coordinates": [[[x1,y1],[x2,y1],[x2,y2],[x1,y2]], ...],
    "negative_box_exemples_coordinates": [[[x1,y1],[x2,y1],[x2,y2],[x1,y2]], ...],
    "positive_prompt": "Description of positive objects",
    "negative_prompt": "Description of negative objects"
  }
}
```

### Key Transformations

1. **Filename Mapping**: Long descriptive names → Compact structured names
2. **Coordinate Format**: [x1,y1,x2,y2] → [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
3. **Point Generation**: Bounding box centers → Point annotations
4. **Text Prompts**: Object attributes → Natural language descriptions
5. **Dataset Splits**: All images assigned to 'test' split for evaluation

## Usage Examples

### Quick Dataset Check
```bash
cd dataset
python verify_dataset.py
```

### Convert Custom Annotations
```bash
cd dataset/tools
python convert_cvat_to_pairtally.py my_annotations.json --output_dir ./converted/
```

### Recreate FSC147 Format
```bash
cd dataset/tools
python convert_cvat_to_pairtally.py ../annotations/bbx_anno_valid.json \
    --output_dir ../recreated_fsc147/annotations/ \
    --create_super_category_files
```

## File Requirements

- **Input**: CVAT-exported JSON annotations
- **Images**: Corresponding image files (for filename verification)
- **Output**: PairTally-compatible directory structure

## Troubleshooting

**Common Issues**:
1. **Missing images**: Ensure image files match annotation filenames
2. **Path errors**: Use absolute paths or verify working directory
3. **Memory issues**: Large datasets may require processing in chunks
4. **Format errors**: Verify CVAT export format matches expected structure

**Getting Help**:
- Run tools with `--help` for detailed options
- Check annotation file structure with `verify_dataset.py`
- Examine sample outputs for format verification
