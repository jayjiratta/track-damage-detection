"""
FINAL EVALUATION -- test split only
=====================================
This is the ONLY script in track-segmentation-ver2 that reads the `test/`
folder. It must be run once, after training + model selection are complete,
never used to tune either method.

For every image in test/:
  - loads the ground-truth mask from its YOLO-seg polygon label
  - runs YOLOv26n segmentation -> predicted mask
  - runs the traditional CV method -> predicted mask
  - computes IoU (both methods) against GT
  - computes geometric features of GT/pred masks, and derived errors
    (area/centroid/width/height/perimeter/boundary) for each method

Outputs (into ../report/):
  - test_results.csv        (per-image metrics, both methods)
  - summary_stats.json      (mean/std summary used by visualize.py)
  - masks/<method>/<image>.png   (predicted masks, for visualization)

Usage:
    py evaluation/run_final_evaluation.py --weights ../runs/yolo26n_track_seg/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "roi"))

from mask_utils import (
    yolo_seg_label_to_mask, compute_iou, mask_features,
    feature_errors, boundary_distance, largest_component,
)
from traditional_segment import segment_track_traditional

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "Lane Segmentation ver2.v1i.yolo26"
TEST_IMG_DIR = DATASET_DIR / "test" / "images"
TEST_LBL_DIR = DATASET_DIR / "test" / "labels"
REPORT_DIR = PROJECT_DIR / "report"
MASKS_DIR = REPORT_DIR / "masks"


def predict_yolo(model, img_bgr: np.ndarray) -> np.ndarray:
    """Run YOLO seg model on one image, return binary mask (H,W) uint8."""
    h, w = img_bgr.shape[:2]
    # retina_masks=True returns masks in original-image coordinates. Without it,
    # masks.data is at the letterboxed inference size (384x640 incl. padding),
    # and resizing that straight to (w, h) shifts the mask by the padding.
    results = model.predict(img_bgr, verbose=False, retina_masks=True)[0]
    mask = np.zeros((h, w), dtype=np.uint8)
    if results.masks is None or len(results.masks.data) == 0:
        return mask
    # Union all predicted instances of the single 'track' class.
    for m in results.masks.data.cpu().numpy():
        mask[m > 0.5] = 255
    return largest_component(mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                         default=str(PROJECT_DIR.parent / "models" / "yolo26n_track_seg.pt"))
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}. Train first (train/train_yolo26n_seg.py).")

    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for method in ("yolo", "traditional", "gt"):
        (MASKS_DIR / method).mkdir(parents=True, exist_ok=True)

    image_paths = sorted(TEST_IMG_DIR.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {TEST_IMG_DIR}")

    rows = []
    for img_path in image_paths:
        stem = img_path.stem
        label_path = TEST_LBL_DIR / f"{stem}.txt"

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        gt_mask = yolo_seg_label_to_mask(str(label_path), w, h) if label_path.exists() else np.zeros((h, w), np.uint8)
        yolo_mask = predict_yolo(model, img)
        trad_mask = segment_track_traditional(img)

        cv2.imwrite(str(MASKS_DIR / "gt" / f"{stem}.png"), gt_mask)
        cv2.imwrite(str(MASKS_DIR / "yolo" / f"{stem}.png"), yolo_mask)
        cv2.imwrite(str(MASKS_DIR / "traditional" / f"{stem}.png"), trad_mask)

        gt_feats = mask_features(gt_mask)

        row = {"image": stem}
        for method, pred_mask in (("yolo", yolo_mask), ("traditional", trad_mask)):
            row[f"iou_{method}"] = compute_iou(pred_mask, gt_mask)
            pred_feats = mask_features(pred_mask)
            errs = feature_errors(pred_feats, gt_feats, w, h)
            for k, v in errs.items():
                row[f"{k}_{method}"] = v
            bd = boundary_distance(pred_mask, gt_mask)
            row[f"boundary_error_px_{method}"] = bd if bd is not None else np.nan
        rows.append(row)
        print(f"{stem}: IoU_yolo={row['iou_yolo']:.4f}  IoU_traditional={row['iou_traditional']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "test_results.csv", index=False)

    summary = {}
    for col in df.columns:
        if col == "image":
            continue
        summary[col] = {
            "mean": float(df[col].mean(skipna=True)),
            "std": float(df[col].std(skipna=True)),
            "median": float(df[col].median(skipna=True)),
            "min": float(df[col].min(skipna=True)),
            "max": float(df[col].max(skipna=True)),
        }
    with open(REPORT_DIR / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"N test images: {len(df)}")
    print(f"Mean IoU  YOLOv26n     : {summary['iou_yolo']['mean']:.4f}  (std {summary['iou_yolo']['std']:.4f})")
    print(f"Mean IoU  Traditional  : {summary['iou_traditional']['mean']:.4f}  (std {summary['iou_traditional']['std']:.4f})")
    print(f"Saved: {REPORT_DIR / 'test_results.csv'}")
    print(f"Saved: {REPORT_DIR / 'summary_stats.json'}")


if __name__ == "__main__":
    main()
