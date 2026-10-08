"""
Calibrate the damage threshold (no damage labels needed): p99.5 of PaDiM
anomaly scores over ROI pixels of held-out NORMAL frames -- frames PaDiM
never saw during fitting (models/padim_split.json "holdout"). Only the
calibration half (holdout[::2]) is used; the other half (holdout[1::2]) is
reserved for evaluation, so reported false-alarm rates are not measured on
the frames that set the threshold.
Stored per ROI method + PaDiM backbone, since both change the scores.

Usage:
    python calibrate.py                          # method 1 and method 2
    python calibrate.py --configs yolo_r18+shared_r18
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pipeline import DamagePipeline, THRESHOLD_PATH, threshold_key

PERCENTILE = 99.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["yolo+resnet18", "yolo_r18+shared_r18"])
    args = ap.parse_args()

    split = json.loads((ROOT / "models" / "padim_split.json").read_text())
    data = ROOT / split["data"]
    holdout = split["holdout"][::2]
    saved = json.loads(THRESHOLD_PATH.read_text()) if THRESHOLD_PATH.exists() else {}

    for cfg in args.configs:
        roi_method, backbone = cfg.split("+")
        pipe = DamagePipeline(roi_method, backbone, threshold=0.0)  # threshold unused while calibrating
        scores = []
        for name in holdout:
            roi, anomaly = pipe.roi_and_anomaly(cv2.imread(str(data / name)))
            scores.append(anomaly[roi > 0])
        scores = np.concatenate(scores)
        thr = float(np.percentile(scores, PERCENTILE))
        saved[threshold_key(roi_method, backbone)] = {"threshold": thr, "percentile": PERCENTILE,
                                                       "n_frames": len(holdout), "n_pixels": int(scores.size)}
        print(f"{cfg}: threshold = {thr:.3f}  (p{PERCENTILE} of {scores.size:,} held-out ROI pixels)")

    THRESHOLD_PATH.write_text(json.dumps(saved, indent=2))
    print(f"saved: {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()
