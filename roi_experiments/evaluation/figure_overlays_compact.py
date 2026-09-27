"""
Compact report figure: Original | Ground Truth | YOLOv26n | Traditional CV for
a few test images, with per-image IoU. Reads report/masks + report/test_results.csv
(produced by run_final_evaluation.py).

Usage (from roi_experiments/):
    python evaluation/figure_overlays_compact.py [stem ...]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMG_DIR = PROJECT_DIR / "Lane Segmentation ver2.v1i.yolo26" / "test" / "images"
REPORT_DIR = PROJECT_DIR / "report"
DEFAULT = ["frame_002441_jpg_jpg.rf.7a43de9d2afd0c5187561fd0da2aa2c7",
           "frame_002881_jpg_jpg.rf.3fa542e014d7f7ed4a40002d010f0ea0",
           "frame_000921_jpg_jpg.rf.025b7ef275edc6d60512b9b2cf78a62c"]
W, H = 480, 270
COLORS = {"gt": (0, 200, 0), "yolo": (0, 0, 230), "traditional": (230, 120, 0)}


def cell(img, mask, color, text):
    out = cv2.resize(img, (W, H))
    if mask is not None:
        m = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) > 0
        out[m] = (0.5 * out[m] + 0.5 * np.array(color)).astype(np.uint8)
    cv2.rectangle(out, (0, 0), (W, 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return out


def main():
    stems = sys.argv[1:] or DEFAULT
    df = pd.read_csv(REPORT_DIR / "test_results.csv").set_index("image")
    rows = []
    for s in stems:
        img = cv2.imread(str(IMG_DIR / f"{s}.jpg"))
        m = {k: cv2.imread(str(REPORT_DIR / "masks" / k / f"{s}.png"), 0) for k in COLORS}
        r = df.loc[s]
        rows.append(cv2.hconcat([
            cell(img, None, None, "Original"),
            cell(img, m["gt"], COLORS["gt"], "Ground Truth"),
            cell(img, m["yolo"], COLORS["yolo"], f"YOLOv26n  IoU={r.iou_yolo:.3f}"),
            cell(img, m["traditional"], COLORS["traditional"], f"Traditional CV  IoU={r.iou_traditional:.3f}"),
        ]))
    out = REPORT_DIR / "figures" / "overlays_compact.jpg"
    cv2.imwrite(str(out), cv2.vconcat(rows))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
