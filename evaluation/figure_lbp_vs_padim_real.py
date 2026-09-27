"""
Qualitative figure on real frames only (no synthetic defects): original |
LBP demo | PaDiM (YOLOv26 ROI + paint filter), for a normal frame, a frame
with road markings, and a frame with a real crack.

Usage (from track-damage-pipeline/):
    python evaluation/figure_lbp_vs_padim_real.py <capture video>
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
import lbp_baseline as lbp
from pipeline import DamagePipeline

FRAMES = [1960, 4200, 21936]   # normal / road markings / real crack (capture_20260806_121427.mp4)
W, H = 560, 315


def overlay(img, dmg):
    vis = img.copy()
    vis[dmg > 0] = (0.35 * vis[dmg > 0] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    cnts, _ = cv2.findContours(dmg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "evaluation" / "qualitative_lbp_vs_padim_real.jpg")
    args = ap.parse_args()
    pipe = DamagePipeline("yolo")
    cap = cv2.VideoCapture(str(args.video))
    rows = []
    for f in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        _, _, d_lbp = lbp.run(img)
        d_padim = pipe(img).damage
        rows.append(cv2.hconcat([cv2.resize(x, (W, H)) for x in (img, overlay(img, d_lbp), overlay(img, d_padim))]))
    header = np.zeros((40, W * 3, 3), np.uint8)
    for j, t in enumerate(["Original", "LBP (demo system)", "PaDiM + YOLOv26 ROI"]):
        cv2.putText(header, t, (12 + W * j, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(args.out), cv2.vconcat([header] + rows))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
