"""
Calibrate the damage threshold (no damage labels needed): p99.5 of PaDiM
anomaly scores over ROI pixels of held-out NORMAL frames -- frames PaDiM
never saw during fitting (models/padim_split.json "holdout"). Only the
calibration half (holdout[::2]) is used; the other half (holdout[1::2]) is
reserved for evaluation/evaluate.py, so reported false-positive rates are
not measured on the frames that set the threshold.
Stored per ROI method, since the ROI changes which pixels are scored.

Usage:
    py calibrate.py                    # both ROI methods
    py calibrate.py --roi yolo
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pipeline import DamagePipeline, erode_roi, THRESHOLD_PATH

DEFAULT_DATA = ROOT.parent / "damage-detection" / "data"
PERCENTILE = 99.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", nargs="+", default=["yolo", "traditional"], choices=["yolo", "traditional"])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = ap.parse_args()

    holdout = json.loads((ROOT / "models" / "padim_split.json").read_text())["holdout"][::2]
    saved = json.loads(THRESHOLD_PATH.read_text()) if THRESHOLD_PATH.exists() else {}

    for method in args.roi:
        pipe = DamagePipeline(method, threshold=0.0)  # threshold unused while calibrating
        scores = []
        for name in holdout:
            img = cv2.imread(str(args.data / name))
            roi = erode_roi(pipe.roi(img))
            scores.append(pipe.padim.score(img)[roi > 0])
        scores = np.concatenate(scores)
        thr = float(np.percentile(scores, PERCENTILE))
        saved[method] = {"threshold": thr, "percentile": PERCENTILE,
                         "n_frames": len(holdout), "n_pixels": int(scores.size)}
        print(f"{method}: threshold = {thr:.3f}  (p{PERCENTILE} of {scores.size:,} held-out ROI pixels)")

    THRESHOLD_PATH.write_text(json.dumps(saved, indent=2))
    print(f"saved: {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()
