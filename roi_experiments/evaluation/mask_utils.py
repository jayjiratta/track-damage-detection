"""
Shared mask / geometry utilities for track-segmentation-ver2
=============================================================
Used by:
  - traditional/traditional_segment.py   (produces predicted masks)
  - evaluation/run_final_evaluation.py   (loads GT + both predictions, scores them)

All masks are represented as uint8 numpy arrays of shape (H, W) with values
{0, 255} ("binary mask", track = 255).
"""
from __future__ import annotations

import numpy as np
import cv2


# ───────────────────────────── polygon <-> mask ─────────────────────────────

def yolo_seg_label_to_mask(label_path: str, img_w: int, img_h: int) -> np.ndarray:
    """
    Read a YOLO-segmentation label file (class + normalized polygon xy pairs,
    one or more object lines) and rasterize it into a single binary mask.
    Multiple polygon lines (e.g. disjoint track regions) are unioned together.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    with open(label_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 7:  # class + at least 3 points
            continue
        coords = list(map(float, parts[1:]))
        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h
        pts = pts.round().astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

    return mask


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary mask."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == largest_label] = 255
    return out


# ───────────────────────────── accuracy metric ─────────────────────────────

def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


# ───────────────────────────── geometric features ───────────────────────────
# Chosen to characterize the shape/position of the whole-track mask, analogous
# to the (A, B, C) curve-parameters used in the original track-segmentation
# project. These are absolute descriptors; the evaluator later turns them into
# *errors relative to ground truth* rather than comparing raw values, since raw
# track geometry legitimately varies frame to frame.

def mask_features(mask: np.ndarray) -> dict | None:
    """
    Compute geometric descriptors of a binary mask:
      - area_ratio     : mask area / image area
      - centroid_x/y   : normalized centroid position (0-1)
      - width_ratio    : bounding-box width / image width
      - height_ratio   : bounding-box height / image height
      - perimeter      : normalized perimeter length (contour length / image diagonal)
    Returns None if the mask is empty (no predicted track region).
    """
    h, w = mask.shape[:2]
    bin_mask = (mask > 0).astype(np.uint8)
    area = int(bin_mask.sum())
    if area == 0:
        return None

    ys, xs = np.nonzero(bin_mask)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    m = cv2.moments(bin_mask, binaryImage=True)
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
    diag = np.hypot(w, h)

    return {
        "area_ratio":   area / float(w * h),
        "centroid_x":   cx / w,
        "centroid_y":   cy / h,
        "width_ratio":  (x1 - x0 + 1) / float(w),
        "height_ratio": (y1 - y0 + 1) / float(h),
        "perimeter":    perimeter / diag,
    }


BORDER_MARGIN_PX = 6   # GT polygons stop up to 4 px short of the frame bottom (measured on test GT)


def _inner_boundary(m: np.ndarray) -> np.ndarray:
    """Mask pixels with a background 4-neighbour, excluding a BORDER_MARGIN_PX
    band along the image frame. The frame edge is not a track boundary: the
    track always runs off the bottom of the frame, and the GT polygons are
    rasterized 1-2 px short of the frame edge, which would otherwise create a
    spurious ~1000 px 'boundary' along the bottom of the GT mask only.
    (cv2.erode's default border also counts outside-the-image as foreground.)"""
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    edge = (m > 0) & (cv2.erode(m, k) == 0)
    b = BORDER_MARGIN_PX
    edge[:b], edge[-b:], edge[:, :b], edge[:, -b:] = False, False, False, False
    return edge


def boundary_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float | None:
    """
    Average symmetric boundary distance (px): mean, over the boundary pixels of
    both masks, of the distance to the nearest boundary pixel of the other mask.
    Image-border segments are excluded (see _inner_boundary).
    Returns None if either mask is empty or has no in-image boundary.
    """
    a = (mask_a > 0).astype(np.uint8)
    b = (mask_b > 0).astype(np.uint8)
    if a.sum() == 0 or b.sum() == 0:
        return None
    ba, bb = _inner_boundary(a), _inner_boundary(b)
    if not ba.any() or not bb.any():
        return None
    dt_a = cv2.distanceTransform((~ba).astype(np.uint8), cv2.DIST_L2, 5)
    dt_b = cv2.distanceTransform((~bb).astype(np.uint8), cv2.DIST_L2, 5)
    return float(np.concatenate([dt_b[ba], dt_a[bb]]).mean())


def feature_errors(pred_feats: dict | None, gt_feats: dict | None, img_w: int, img_h: int) -> dict:
    """
    Convert predicted vs ground-truth feature dicts into error metrics.
    If prediction is empty (None), errors are reported as the worst-case
    (max possible) rather than dropped, so a total-miss is penalized, not ignored.
    """
    if gt_feats is None:
        return {}

    if pred_feats is None:
        # total miss: max possible errors
        return {
            "area_error":     gt_feats["area_ratio"],
            "centroid_error": float(np.hypot(gt_feats["centroid_x"], gt_feats["centroid_y"])),
            "width_error":    gt_feats["width_ratio"],
            "height_error":   gt_feats["height_ratio"],
            "perimeter_error": gt_feats["perimeter"],
        }

    centroid_error = float(np.hypot(
        pred_feats["centroid_x"] - gt_feats["centroid_x"],
        pred_feats["centroid_y"] - gt_feats["centroid_y"],
    ))

    return {
        "area_error":      abs(pred_feats["area_ratio"] - gt_feats["area_ratio"]),
        "centroid_error":  centroid_error,
        "width_error":     abs(pred_feats["width_ratio"] - gt_feats["width_ratio"]),
        "height_error":    abs(pred_feats["height_ratio"] - gt_feats["height_ratio"]),
        "perimeter_error": abs(pred_feats["perimeter"] - gt_feats["perimeter"]),
    }
