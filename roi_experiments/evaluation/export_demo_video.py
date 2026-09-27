"""
Export Demo Comparison Video -- YOLOv26n vs Traditional CV
=============================================================
Same purpose as track-segmentation/benchmark/export_video.py's
demo_comparison.mp4, adapted to this project: ver2's dataset is 45 discrete
test IMAGES (no source video), so this builds a slideshow-style video, one
test image per "shot" (held for HOLD_SECONDS), 2x2 panel layout:

    Original          | Ground Truth
    YOLOv26n (IoU)     | Traditional CV (IoU)

Reads test images + the masks already produced by run_final_evaluation.py
(report/masks/{gt,yolo,traditional}/) and its per-image IoU
(report/test_results.csv) -- does not re-run either model, so this is pure
visualization, no new test-set "use".

Usage:
    py evaluation/export_demo_video.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_IMG_DIR = PROJECT_DIR / "Lane Segmentation ver2.v1i.yolo26" / "test" / "images"
REPORT_DIR = PROJECT_DIR / "report"
MASKS_DIR = REPORT_DIR / "masks"
OUT_PATH = REPORT_DIR / "demo_comparison.mp4"

PANEL_W, PANEL_H = 640, 360
FPS = 30
HOLD_SECONDS = 1.5
FRAMES_PER_IMAGE = int(FPS * HOLD_SECONDS)

COLORS = {"gt": (0, 200, 0), "yolo": (0, 0, 255), "traditional": (255, 128, 0)}


def overlay_mask(img, mask, color, alpha=0.45):
    out = img.copy()
    m = mask > 0
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out


def label(panel, text, sub=None):
    panel = panel.copy()
    cv2.rectangle(panel, (0, 0), (PANEL_W, 34), (0, 0, 0), -1)
    cv2.putText(panel, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if sub is not None:
        cv2.rectangle(panel, (0, PANEL_H - 30), (PANEL_W, PANEL_H), (0, 0, 0), -1)
        cv2.putText(panel, sub, (10, PANEL_H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return panel


def main():
    df = pd.read_csv(REPORT_DIR / "test_results.csv").set_index("image")
    stems = sorted(df.index.tolist())
    if not stems:
        raise FileNotFoundError("report/test_results.csv is empty -- run run_final_evaluation.py first")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (PANEL_W * 2, PANEL_H * 2))

    for i, stem in enumerate(stems):
        img = cv2.imread(str(DATASET_IMG_DIR / f"{stem}.jpg"))
        img = cv2.resize(img, (PANEL_W, PANEL_H))

        def load_mask(method):
            m = cv2.imread(str(MASKS_DIR / method / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            return cv2.resize(m, (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST)

        gt_mask = load_mask("gt")
        yolo_mask = load_mask("yolo")
        trad_mask = load_mask("traditional")

        row = df.loc[stem]

        p_orig = label(img, f"Original  [{i+1}/{len(stems)}]  {stem[:22]}")
        p_gt = label(overlay_mask(img, gt_mask, COLORS["gt"]), "Ground Truth")
        p_yolo = label(overlay_mask(img, yolo_mask, COLORS["yolo"]), "YOLOv26n",
                       f"IoU: {row['iou_yolo']:.4f}")
        p_trad = label(overlay_mask(img, trad_mask, COLORS["traditional"]), "Traditional CV",
                        f"IoU: {row['iou_traditional']:.4f}")

        top = cv2.hconcat([p_orig, p_gt])
        bottom = cv2.hconcat([p_yolo, p_trad])
        frame = cv2.vconcat([top, bottom])

        for _ in range(FRAMES_PER_IMAGE):
            out.write(frame)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(stems)} images written...")

    out.release()
    print(f"\nSaved: {OUT_PATH}  ({len(stems)} images x {HOLD_SECONDS}s @ {FPS}fps)")


if __name__ == "__main__":
    main()
