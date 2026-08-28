"""
Cityscapes Dataset Loader for Detectron2
Shared utility for loading and registering Cityscapes dataset across notebooks.
"""

import json
import cv2
from pathlib import Path
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


# Cityscapes class definitions (instance segmentation classes only)
CITYSCAPES_THING_CLASSES = [
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

CITYSCAPES_THING_COLORS = [
    (220, 20, 60),   # person - red
    (255, 0, 0),     # rider - red
    (0, 0, 142),     # car - blue
    (0, 0, 70),      # truck - dark blue
    (0, 60, 100),    # bus - dark cyan
    (0, 80, 100),    # train - teal
    (0, 0, 230),     # motorcycle - blue
    (119, 11, 32),   # bicycle - maroon
]


def load_cityscapes_custom(image_dir: str, gt_dir: str, split: str):
    """
    Custom Cityscapes loader that reads JSON polygon annotations directly.
    Avoids dependency on cityscapesscripts package.
    
    Args:
        image_dir: Path to leftImg8bit/{split}
        gt_dir: Path to gtFine/{split}
        split: 'train' or 'val'
    
    Returns:
        List of dataset dicts in Detectron2 format
    """
    dataset_dicts = []
    image_dir = Path(image_dir)
    gt_dir = Path(gt_dir)
    
    cities = sorted([d for d in image_dir.iterdir() if d.is_dir()])
    
    for city_dir in cities:
        city_name = city_dir.name
        gt_city_dir = gt_dir / city_name
        image_files = sorted(city_dir.glob("*_leftImg8bit.png"))
        
        for img_path in image_files:
            base_name = img_path.stem.replace("_leftImg8bit", "")
            json_path = gt_city_dir / f"{base_name}_gtFine_polygons.json"
            
            # Skip reading image if we're only metadata building
            # (In Cityscapes, images are usually fixed size 1024x2048)
            height, width = 1024, 2048 
            
            record = {
                "file_name": str(img_path),
                "image_id": base_name,
                "height": height,
                "width": width,
                "annotations": [],
            }
            
            with open(json_path, "r") as f:
                data = json.load(f)
            
            for obj in data.get("objects", []):
                label = obj.get("label", "")
                if label not in CITYSCAPES_THING_CLASSES:
                    continue
                
                polygon = obj.get("polygon", [])
                if len(polygon) < 3:
                    continue
                
                poly_flat = []
                for point in polygon:
                    poly_flat.extend([float(point[0]), float(point[1])])
                
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                bbox = [x_min, y_min, x_max, y_max]
                bbox_area = (x_max - x_min) * (y_max - y_min)
                
                annotation = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly_flat],
                    "category_id": CITYSCAPES_THING_CLASSES.index(label),
                    "iscrowd": 0,
                    "area": bbox_area,
                }
                record["annotations"].append(annotation)
            
            if record["annotations"]:
                dataset_dicts.append(record)
    
    return dataset_dicts


def register_cityscapes_dataset(data_root: Path, split: str = "val"):
    """
    Register Cityscapes dataset with Detectron2.
    
    Args:
        data_root: Path to data/raw folder containing leftImg8bit and gtFine
        split: 'train' or 'val'
    
    Returns:
        tuple: (dataset_name, dataset_dicts, metadata)
    """
    cityscapes_img_path = data_root / "leftImg8bit"
    cityscapes_gt_path = data_root / "gtFine"
    
    # Check if dataset exists
    if not cityscapes_img_path.exists() or not cityscapes_gt_path.exists():
        raise FileNotFoundError(
            f"Cityscapes dataset not found at {data_root}. "
            f"Expected folders: leftImg8bit/ and gtFine/"
        )
    
    dataset_name = f"cityscapes_custom_{split}"
    
    # Remove old registration if exists
    if dataset_name in DatasetCatalog:
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)
    
    # Register
    DatasetCatalog.register(
        dataset_name,
        lambda: load_cityscapes_custom(
            str(cityscapes_img_path / split),
            str(cityscapes_gt_path / split),
            split
        )
    )
    
    # Set metadata
    MetadataCatalog.get(dataset_name).set(
        thing_classes=CITYSCAPES_THING_CLASSES,
        thing_colors=CITYSCAPES_THING_COLORS,
    )
    
    # Load and return
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    return dataset_name, dataset_dicts, metadata


def setup_dataset(use_cityscapes: bool = True, split: str = "val"):
    """
    Convenience function to setup either Cityscapes or COCO dataset.
    
    Args:
        use_cityscapes: If True, use Cityscapes; otherwise use COCO
        split: 'train' or 'val'
    
    Returns:
        tuple: (dataset_name, dataset_dicts, metadata)
    """
    if use_cityscapes:
        from pathlib import Path
        PROJECT_ROOT = Path.home() / "VS_Code" / "INSEG_PRO"
        DATA_DIR = PROJECT_ROOT / "data" / "raw"
        
        try:
            return register_cityscapes_dataset(DATA_DIR, split)
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            print("   Falling back to COCO dataset...")
            use_cityscapes = False
    
    if not use_cityscapes:
        # Use COCO
        dataset_name = f"coco_2017_{split}"
        dataset_dicts = DatasetCatalog.get(dataset_name)
        metadata = MetadataCatalog.get(dataset_name)
        return dataset_name, dataset_dicts, metadata
