# Dataset Loader Module

This module provides shared utilities for loading and registering datasets across all notebooks.

## Why This Approach?

Each Jupyter notebook runs in its **own isolated kernel**, even when using the same virtual environment. Variables and registrations don't carry over between notebooks. 

The professional solution is to create **shared Python modules** that can be imported by any notebook.

## Usage

### In Any Notebook

```python
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path.home() / "VS_Code" / "INSEG_PRO"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import and use the dataset loader
from data.dataset_loader import setup_dataset

# Load dataset (auto-detects Cityscapes or falls back to COCO)
dataset_name, dataset_dicts, metadata = setup_dataset(
    use_cityscapes=True, 
    split="val"
)

print(f"✅ Loaded {len(dataset_dicts)} images from {dataset_name}")
```

## Available Functions

### `setup_dataset(use_cityscapes=True, split="val")`
Convenience function that automatically sets up either Cityscapes or COCO dataset.

**Returns:** `(dataset_name, dataset_dicts, metadata)`

### `register_cityscapes_dataset(data_root, split="val")`
Registers Cityscapes dataset with Detectron2.

**Returns:** `(dataset_name, dataset_dicts, metadata)`

### `load_cityscapes_custom(image_dir, gt_dir, split)`
Low-level loader function that reads Cityscapes annotations from JSON files.

**Returns:** List of dataset dictionaries in Detectron2 format

## Benefits

✅ **No code duplication** - Write once, use everywhere  
✅ **Easy maintenance** - Update in one place  
✅ **Professional structure** - Standard Python package organization  
✅ **Version control friendly** - Easy to track changes  
✅ **Testable** - Can write unit tests for your loader  

## Dataset Classes

The module defines 8 Cityscapes instance segmentation classes:
- person
- rider  
- car
- truck
- bus
- train
- motorcycle
- bicycle
