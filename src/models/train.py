"""
YOLOv8 Training Script — NEU Surface Defect Detection
Manufacturing quality control for steel surface defect classification.

Usage:
    python src/models/train.py
    python src/models/train.py --model yolov8s --epochs 100
"""

import argparse
import json
import os
from pathlib import Path


def train(
    model_size: str = "s",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    data_yaml: str = "data/neu_yolo/neu_dataset.yaml",
    project: str = "results",
    name: str = "neu_defect_yolov8",
):
    """
    Train YOLOv8 on NEU Surface Defect dataset.

    Args:
        model_size:  YOLOv8 variant — n(ano), s(mall), m(edium), l(arge), x
        epochs:      Training epochs (100 recommended for convergence)
        imgsz:       Input image size (640 standard for YOLO)
        batch:       Batch size (16 for T4 GPU, 32 for A100)
        data_yaml:   Path to dataset YAML file
        project:     Output directory
        name:        Experiment name
    """
    from ultralytics import YOLO

    model_name = f"yolov8{model_size}.pt"
    print(f"\nTraining YOLOv8{model_size.upper()} on NEU Surface Defect Dataset")
    print(f"Model:   {model_name}")
    print(f"Epochs:  {epochs}")
    print(f"ImgSize: {imgsz}")
    print(f"Batch:   {batch}")
    print("=" * 50)

    # Load pretrained YOLOv8 (transfer learning from COCO)
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        # Augmentation — important for small manufacturing dataset
        hsv_h=0.015,       # hue augmentation
        hsv_s=0.7,         # saturation
        hsv_v=0.4,         # value/brightness
        degrees=10,        # rotation (defects can appear at any angle)
        translate=0.1,
        scale=0.5,
        fliplr=0.5,        # horizontal flip
        flipud=0.5,        # vertical flip (steel surface is symmetric)
        mosaic=1.0,        # mosaic augmentation
        # Training params
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=20,       # early stopping
        save=True,
        plots=True,
        verbose=True,
    )

    # Save metrics
    metrics = {
        "model": model_name,
        "epochs": epochs,
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
    }

    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/training_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"mAP@50:    {metrics['mAP50']:.4f}")
    print(f"mAP@50-95: {metrics['mAP50_95']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"\nBest model saved to: {project}/{name}/weights/best.pt")

    return model, metrics


def export_openvino(model_path: str = "results/neu_defect_yolov8/weights/best.pt"):
    """
    Export trained model to Intel OpenVINO IR format.
    Enables optimized inference on Intel CPUs, GPUs, and VPUs.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    export_path = model.export(format="openvino", imgsz=640)

    print(f"\nOpenVINO model exported to: {export_path}")
    print("Run inference with:")
    print("  from openvino.runtime import Core")
    print("  ie = Core()")
    print("  model = ie.read_model('best_openvino_model/best.xml')")
    print("  compiled = ie.compile_model(model, 'CPU')")

    return export_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="s",
                        choices=["n","s","m","l","x"],
                        help="YOLOv8 variant")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--data",    default="data/neu_yolo/neu_dataset.yaml")
    parser.add_argument("--export",  action="store_true",
                        help="Export to OpenVINO after training")
    args = parser.parse_args()

    model, metrics = train(
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        data_yaml=args.data,
    )

    if args.export:
        export_openvino()
