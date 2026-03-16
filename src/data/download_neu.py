"""
NEU Surface Defect Dataset — Download & Prepare
Converts to YOLO annotation format for YOLOv8 training.

Dataset: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
Classes: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches
"""

import os
import shutil
import random
from pathlib import Path


# ── Class definitions ─────────────────────────────────────────────────────────
CLASSES = {
    "Crazing": 0,
    "Inclusion": 1,
    "Patches": 2,
    "Pitted_surface": 3,
    "Rolled-in_scale": 4,
    "Scratches": 5,
}

CLASS_NAMES = list(CLASSES.keys())


def prepare_neu_dataset(
    source_dir: str,
    output_dir: str = "data/neu_yolo",
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
):
    """
    Convert NEU Surface Defect dataset to YOLO format.

    NEU dataset structure (after download):
        NEU-DET/
            IMAGES/
                Crazing_1.jpg ... Crazing_300.jpg
                Inclusion_1.jpg ... etc
            ANNOTATIONS/
                Crazing_1.xml ... (Pascal VOC XML format)

    Output YOLO structure:
        neu_yolo/
            images/train/ val/ test/
            labels/train/ val/ test/
            neu_dataset.yaml
    """
    random.seed(seed)
    source = Path(source_dir)
    output = Path(output_dir)

    # Create output directories
    for split in ["train", "val", "test"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Collect all images per class
    all_files = []
    img_dir = source / "IMAGES"
    ann_dir = source / "ANNOTATIONS"

    for class_name, class_id in CLASSES.items():
        images = sorted(img_dir.glob(f"{class_name}_*.jpg"))
        if not images:
            images = sorted(img_dir.glob(f"{class_name.replace('-', '_')}_*.jpg"))

        for img_path in images:
            ann_path = ann_dir / (img_path.stem + ".xml")
            all_files.append((img_path, ann_path, class_id))

    random.shuffle(all_files)

    n = len(all_files)
    n_train = int(n * train_split)
    n_val   = int(n * val_split)

    splits = {
        "train": all_files[:n_train],
        "val":   all_files[n_train:n_train + n_val],
        "test":  all_files[n_train + n_val:],
    }

    for split_name, files in splits.items():
        for img_path, ann_path, class_id in files:
            # Copy image
            dst_img = output / "images" / split_name / img_path.name
            shutil.copy2(img_path, dst_img)

            # Convert annotation XML → YOLO format
            if ann_path.exists():
                yolo_label = xml_to_yolo(ann_path, class_id)
            else:
                # Fallback: full-image bounding box
                yolo_label = f"{class_id} 0.5 0.5 1.0 1.0\n"

            dst_lbl = output / "labels" / split_name / (img_path.stem + ".txt")
            dst_lbl.write_text(yolo_label)

        print(f"  {split_name}: {len(files)} images")

    # Write dataset YAML
    yaml_content = f"""# NEU Surface Defect Dataset — YOLO format
# Source: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
# 1800 images, 6 defect classes, 300 per class

path: {output.absolute()}
train: images/train
val:   images/val
test:  images/test

nc: {len(CLASSES)}
names: {CLASS_NAMES}
"""
    (output / "neu_dataset.yaml").write_text(yaml_content)
    print(f"\nDataset YAML written to: {output / 'neu_dataset.yaml'}")
    print(f"Total: {n} images → train:{n_train} val:{n_val} test:{n - n_train - n_val}")

    return str(output / "neu_dataset.yaml")


def xml_to_yolo(xml_path: Path, class_id: int) -> str:
    """Convert Pascal VOC XML annotation to YOLO format (normalized xywh)."""
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        lines = []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # Convert to YOLO normalized format
            cx = (xmin + xmax) / 2 / img_w
            cy = (ymin + ymax) / 2 / img_h
            w  = (xmax - xmin) / img_w
            h  = (ymax - ymin) / img_h

            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        return "\n".join(lines) + "\n"

    except Exception:
        # Fallback: centered full-image box
        return f"{class_id} 0.5 0.5 1.0 1.0\n"


if __name__ == "__main__":
    print("NEU Surface Defect Dataset Preparation")
    print("=" * 50)
    print("\nDownload the dataset from:")
    print("http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html")
    print("\nOr use Kaggle:")
    print("kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database")
    print("\nThen run:")
    print("python src/data/download_neu.py --source NEU-DET --output data/neu_yolo")
