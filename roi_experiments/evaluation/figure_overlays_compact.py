"""
Compact report figure: Original | Ground Truth | YOLOv26n | Traditional CV for
a few test images, with per-image IoU. Every mask is drawn in the same colour; red
boxes mark large errors against the ground truth (wrongly included or missed track
area). Reads report/masks + report/test_results.csv (run_final_evaluation.py).

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
MASK_COLOR = (0, 200, 0)
ERROR_MIN_PX = 2500          # error components larger than this (full resolution) get a red box


def cell(img, mask, text, gt=None):
    out = cv2.resize(img, (W, H))
    if mask is not None:
        m = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) > 0
        out[m] = (0.5 * out[m] + 0.5 * np.array(MASK_COLOR)).astype(np.uint8)
        if gt is not None:
            err = ((mask > 0) ^ (gt > 0)).astype(np.uint8)
            err = cv2.morphologyEx(err, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            n, _, stats, _ = cv2.connectedComponentsWithStats(err)
            sx, sy = W / mask.shape[1], H / mask.shape[0]
            for x, y, w, h, a in stats[1:]:
                if a >= ERROR_MIN_PX:
                    cv2.rectangle(out, (int(x * sx) - 3, int(y * sy) - 3), (int((x + w) * sx) + 3, int((y + h) * sy) + 3),
                                  (0, 0, 255), 3)
    cv2.rectangle(out, (0, 0), (W, 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return out


def main():
    stems = sys.argv[1:] or DEFAULT
    df = pd.read_csv(REPORT_DIR / "test_results.csv").set_index("image")
    rows = []
    for s in stems:
        img = cv2.imread(str(IMG_DIR / f"{s}.jpg"))
        m = {k: cv2.imread(str(REPORT_DIR / "masks" / k / f"{s}.png"), 0) for k in ("gt", "yolo", "traditional")}
        r = df.loc[s]
        rows.append(cv2.hconcat([
            cell(img, None, "Original"),
            cell(img, m["gt"], "Ground Truth"),
            cell(img, m["yolo"], f"YOLOv26n  IoU={r.iou_yolo:.3f}", gt=m["gt"]),
            cell(img, m["traditional"], f"Traditional CV  IoU={r.iou_traditional:.3f}", gt=m["gt"]),
        ]))
    out = REPORT_DIR / "figures" / "overlays_compact.jpg"
    cv2.imwrite(str(out), cv2.vconcat(rows))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
