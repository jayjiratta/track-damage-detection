"""
Derive the field-of-view envelope prior used by traditional_segment.py, from
the TRAIN split's ground-truth polygons only (no test-set involvement).

This is a one-off calibration helper, kept for reproducibility/documentation.
Its output (printed as a Python literal) is hand-copied into
traditional_segment.py's ENVELOPE_PRIOR constant -- it is not re-run as part
of the pipeline, so the traditional method has no runtime dependency on label
files being present.

Rationale: every image in this dataset comes from the same fixed,
chest/head-mounted camera walking down the track, so the track's plausible
on-screen position/width is dataset-consistent -- it converges toward a
vanishing point around y=24% of the frame height and widens going down.
This script measures that envelope directly from train GT (min/max track
x-extent at each row, 1st/99th percentile to ignore label outliers, plus a
5% safety margin), rather than guessing fixed ROI points by eye.

Usage:
    python roi/derive_envelope_prior.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "roi_experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR / "evaluation"))
from mask_utils import yolo_seg_label_to_mask

TRAIN_LABEL_DIR = EXPERIMENTS_DIR / "Lane Segmentation ver2.v1i.yolo26" / "train" / "labels"
IMG_W, IMG_H = 1280, 720   # dataset's native resolution

ROW_FRACS = list(np.round(np.arange(0.25, 1.001, 0.025), 4))
MARGIN = 0.05  # +/- 5% of image width, safety margin beyond the observed envelope


def main():
    label_files = sorted(TRAIN_LABEL_DIR.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No labels found in {TRAIN_LABEL_DIR}")

    per_row = {r: [] for r in ROW_FRACS}
    top_y_frac = 1.0

    for f in label_files:
        mask = yolo_seg_label_to_mask(str(f), IMG_W, IMG_H)
        ys = np.nonzero(mask.sum(axis=1))[0]
        if len(ys):
            top_y_frac = min(top_y_frac, ys.min() / IMG_H)
        for r in ROW_FRACS:
            y = min(int(IMG_H * r), IMG_H - 1)
            xs = np.nonzero(mask[y])[0]
            if len(xs):
                per_row[r].append((xs.min() / IMG_W, xs.max() / IMG_W))

    print(f"# Topmost row any train GT track pixel ever appears at: {top_y_frac:.4f}")
    print("# Paste this into traditional_segment.py's ENVELOPE_PRIOR:\n")
    print("ENVELOPE_PRIOR = [")
    print(f"    ({top_y_frac:.4f}, 0.50, 0.50),  # vanishing point -- track never appears above this row")
    for r in ROW_FRACS:
        vals = per_row[r]
        if not vals:
            continue
        p1 = max(0.0, np.percentile([v[0] for v in vals], 1) - MARGIN)
        p99 = min(1.0, np.percentile([v[1] for v in vals], 99) + MARGIN)
        print(f"    ({r}, {p1:.4f}, {p99:.4f}),")
    print("]")


if __name__ == "__main__":
    main()
