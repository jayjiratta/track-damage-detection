"""
Figure: step-by-step output of the traditional (non-deep-learning) track ROI
method in roi/traditional_segment.py, for the report's theory section.

Usage:
    python evaluation/figure_traditional_steps.py <image> [--out results/evaluation/traditional_steps.jpg]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "roi"))
import traditional_segment as ts

TILE_W, TILE_H = 560, 315


def tile(img, label):
    t = cv2.resize(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (TILE_W, TILE_H))
    cv2.rectangle(t, (0, 0), (TILE_W, 34), (0, 0, 0), -1)
    cv2.putText(t, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return t


def overlay(img, mask, color):
    out = img.copy()
    out[mask > 0] = (0.5 * out[mask > 0] + 0.5 * np.array(color)).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "evaluation" / "traditional_steps.jpg")
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1]
    _, otsu = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    prior = ts.build_envelope_prior_mask(h, w)
    coarse = ts._coarse_color_mask(img, lab)
    final = ts.segment_track_traditional(img)

    prior_vis = overlay(img, otsu & prior, (0, 200, 255))
    cnts, _ = cv2.findContours(prior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(prior_vis, cnts, -1, (255, 0, 255), 3)
    a_vis = cv2.applyColorMap(cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET)

    tiles = [tile(img, "(a) Input image"), tile(a_vis, "(b) a* channel (CIE L*a*b*)"),
             tile(otsu, "(c) Otsu threshold on a*"), tile(prior_vis, "(d) FOV prior (magenta) applied"),
             tile(overlay(img, coarse, (0, 200, 255)), "(e) Closing + largest component"),
             tile(overlay(img, final, (0, 0, 255)), "(f) Edge snap + row fill (final)")]
    grid = cv2.vconcat([cv2.hconcat(tiles[:3]), cv2.hconcat(tiles[3:])])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), grid)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
