"""
LBP + local-STD anomaly baseline -- the earlier proof-of-concept demo,
ported unchanged from damage-detection/interactive_lbp_reviewer_fixed.py
(same constants and steps) so it can be compared against PaDiM.

  track mask   fixed trapezoid ROI minus white paint (HSV)
  score        0.4 * LBP(uniform, R=3, P=24) + 0.6 * local STD (15x15),
               each min-max normalized per image, Gaussian sigma=3
  damage       score > per-image 82nd percentile inside the track mask,
               close/open, components < 60 px dropped
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import local_binary_pattern
from skimage.filters import gaussian

LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS
ANOMALY_THRESH_PERCENTILE = 82
MORPH_KERNEL_SIZE = 7
MIN_DAMAGE_AREA_PX = 60
TRACK_ROI_NORM = np.array([[0.05, 0.98], [0.95, 0.98], [0.65, 0.30], [0.35, 0.30]], dtype=np.float32)


def make_track_mask(img_bgr: np.ndarray) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    roi = np.zeros((H, W), np.uint8)
    cv2.fillPoly(roi, [(TRACK_ROI_NORM * np.array([W, H])).astype(np.int32)], 255)
    white = cv2.inRange(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), np.array([0, 0, 180]), np.array([180, 40, 255]))
    mask = cv2.bitwise_and(roi, cv2.bitwise_not(white))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)


def anomaly_score(img_bgr: np.ndarray, track_mask: np.ndarray | None = None) -> np.ndarray:
    """LBP+STD score map. track_mask=None scores the whole frame (used for the
    probe-box comparison, where PaDiM is also scored without any ROI)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lbp = local_binary_pattern(gray.astype(np.uint8), LBP_N_POINTS, LBP_RADIUS, method="uniform").astype(np.float32)
    lbp_n = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-9)
    mean = ndimage.uniform_filter(gray, size=15)
    sq = ndimage.uniform_filter(gray ** 2, size=15)
    std = np.sqrt(np.maximum(sq - mean ** 2, 0))
    std_n = (std - std.min()) / (std.max() - std.min() + 1e-9)
    score = 0.4 * lbp_n + 0.6 * std_n
    if track_mask is not None:
        score[track_mask == 0] = 0
    score = gaussian(score, sigma=3).astype(np.float32)
    if track_mask is not None:
        score[track_mask == 0] = 0
    return score


def damage_mask(score: np.ndarray, track_mask: np.ndarray) -> np.ndarray:
    vals = score[track_mask == 255]
    if vals.size == 0:
        return np.zeros_like(track_mask)
    dmg = (score > np.percentile(vals, ANOMALY_THRESH_PERCENTILE)).astype(np.uint8) * 255
    dmg[track_mask == 0] = 0
    dmg = cv2.morphologyEx(dmg, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE,) * 2), iterations=2)
    dmg = cv2.morphologyEx(dmg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dmg, connectivity=8)
    out = np.zeros_like(dmg)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_DAMAGE_AREA_PX:
            out[labels == i] = 255
    return out


def run(img_bgr: np.ndarray):
    """Full demo pipeline. Returns (track_mask, score, damage)."""
    tm = make_track_mask(img_bgr)
    s = anomaly_score(img_bgr, tm)
    return tm, s, damage_mask(s, tm)
