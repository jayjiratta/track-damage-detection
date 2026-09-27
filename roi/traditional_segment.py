"""
Traditional Computer-Vision Segmentation of the Whole Running-Track Area
==========================================================================
Project: track-segmentation-ver2

APPROACH
--------
The running track in this dataset is a painted reddish/maroon rubber
surface that is visually distinct from its surroundings (green grass, gray
concrete curbs, water, sky, trees) in every sample image. We combine two
classical, non-learned signals:

  1. COLOR: convert BGR -> CIE LAB and Otsu-threshold the 'a' channel
     (green<->red axis, lighting-largely-invariant) to get a binary
     "reddish" mask. Otsu (rather than a fixed cutoff) re-derives the
     threshold per image, since ambient lighting shifts across the dataset
     (sun vs. overcast).
  2. FIELD-OF-VIEW PRIOR: every image comes from the same fixed,
     chest/head-mounted camera walking down the track, so the track's
     on-screen position is dataset-consistent -- narrow near a vanishing
     point around y=24% of the frame height, widening toward the bottom.
     ENVELOPE_PRIOR below is that corridor, measured directly from the
     TRAIN split's ground-truth polygons only (see derive_envelope_prior.py
     -- 1st/99th percentile of the track's left/right extent at each row,
     plus a 5% safety margin), never from validation or test data.
     Intersecting the color mask with this corridor BEFORE the
     connected-component step is what actually fixes the main failure mode
     of a color-only threshold on this dataset: dry, reddish-brown
     grass/dirt patches near the shoreline or embankment can share the
     track's hue and, once morphologically closed, get pulled into the
     "largest component" as one blob with the real track. Cutting the
     prior's corridor first disconnects those patches from the track blob
     at the point they'd otherwise bridge together, so they're excluded
     by the largest-component step instead of merged into it.
  3. Morphological CLOSE (elliptical kernel) to bridge the thin white
     lane-marking line down the middle of the track.
  4. Keep only the LARGEST connected component (now, thanks to step 2,
     this reliably picks out just the track). This is the COARSE mask;
     its per-row leftmost/rightmost x is the coarse left/right boundary.
  5. EDGE SNAP (hybrid refinement): most images in this dataset have a
     painted white stripe running along the track's outer edge (in
     addition to the center line), a strong, consistent, high-contrast
     feature -- much cleaner than the color-region boundary itself, which
     follows the rubber surface's texture noise. For each row, search a
     small window around the coarse boundary for a locally bright peak
     in the LAB 'L' channel (the white stripe) and snap the boundary to
     its outer edge if the peak is prominent enough AND close enough to
     the coarse boundary (see SNAP_* params). Otherwise keep the coarse
     boundary for that row (graceful fallback -- e.g. where the stripe is
     faded, occluded, or the color region itself was already wrong, this
     step intentionally does not try to relocate the boundary by more
     than SNAP_MAX_SHIFT_PX). A light median filter across rows then
     smooths residual per-row jitter. The final mask is built by filling
     between the two smoothed boundary curves per row -- which, unlike
     steps 3-4's region mask, has no interior holes by construction (a
     glare/highlight patch on the track surface no longer creates a gap,
     since interior pixel color is never consulted after the boundary is
     set).

PARAMETERS
----------
  MORPH_CLOSE_KERNEL = 25  (px, elliptical) -- bridges the ~10-15px-wide
      white center line at 1280x720 without over-smoothing the outer edge.
  ENVELOPE_PRIOR -- see derive_envelope_prior.py; derived from train GT only.
  SNAP_WINDOW_PX = 14      -- how far out from the coarse boundary to look
      for the white edge stripe. Kept small deliberately: a wider window
      also reaches the bright concrete curb just past the stripe, which
      would hijack the peak search and snap the boundary too far out.
  SNAP_PROMINENCE = 30     -- required brightness peak vs. the window's
      median, to accept it as "a real stripe" rather than surface noise.
  SNAP_DROP = 20           -- once at the peak, walk outward while
      brightness stays within this drop, to land on the stripe's outer
      edge (the true track/background boundary) rather than its center.
  SNAP_MAX_SHIFT_PX = 12   -- reject a snap that would move the boundary
      further than this from the coarse estimate (guards against locking
      onto an unrelated bright feature further away).
  ROW_MEDIAN_FILTER = 7    -- rows, smooths residual jitter/outliers in
      the per-row boundary curve.

WHY THIS METHOD FITS THIS DATASET
----------------------------------
  - Single, consistent track material/color across the dataset -- a global
    color threshold is a reasonable, cheap alternative to a learned model.
  - Single, consistent camera mounting/walking path across the dataset --
    a train-GT-derived field-of-view corridor is a reasonable geometric
    prior for the same reason the original track-segmentation project
    could calibrate a fixed ROI to its one camera setup.
  - No lane-graph/perspective BEV fitting is required (unlike the original
    project's 3-line pipeline), since the goal is one whole-track blob.

ASSUMPTIONS / LIMITATIONS
--------------------------
  - Assumes new images share the training data's camera framing (same
    mount height/angle, walking the same track). A camera pointed very
    differently would need ENVELOPE_PRIOR re-derived (or dropped).
  - Assumes the track is the largest reddish region inside that corridor.
    ENVELOPE_PRIOR is a *union* across the training set, so on a handful
    of frames it stays wide enough for a reddish-brown shoreline/dirt
    patch to survive alongside the real track within the corridor -- the
    edge-snap step does not fix this case (there is no white stripe
    between the two, since the patch itself isn't a real track edge), so
    a few frames still show this residual over-segmentation. This is a
    known limitation, not one the edge-snap step claims to solve.
  - Assumes the white edge stripe is visible/unbroken for the edge-snap
    step; where it's faded, occluded, or absent (SNAP_PROMINENCE not
    met), that row silently falls back to the coarse color-region
    boundary, which reintroduces that boundary's own texture-noise
    jaggedness locally.
  - Purely per-frame -- no temporal smoothing (each test image is
    evaluated independently, matching the evaluation protocol).
"""
from __future__ import annotations

import cv2
import numpy as np

MORPH_CLOSE_KERNEL_PX = 25

SNAP_WINDOW_PX = 14
SNAP_PROMINENCE = 30
SNAP_DROP = 20
SNAP_MAX_SHIFT_PX = 12
ROW_MEDIAN_FILTER = 7

# See derive_envelope_prior.py -- (row_frac, x_min_frac, x_max_frac) control
# points measured from the TRAIN split's GT polygons (1st/99th percentile of
# track extent per row + 5% margin). Rows above the first entry never
# contain track in the training data, so are excluded entirely.
ENVELOPE_PRIOR = [
    (0.2375, 0.50, 0.50),
    (0.25, 0.4539, 0.8475), (0.275, 0.3743, 0.6747), (0.3, 0.3060, 0.7013),
    (0.325, 0.2762, 0.7374), (0.35, 0.2457, 0.7772), (0.375, 0.2097, 0.8129),
    (0.4, 0.1734, 0.8543), (0.425, 0.1305, 0.8925), (0.45, 0.0904, 0.9291),
    (0.475, 0.0547, 0.9655), (0.5, 0.0216, 1.0000), (0.525, 0.0000, 1.0000),
    (1.0, 0.0000, 1.0000),
]

_prior_cache: dict[tuple[int, int], np.ndarray] = {}


def build_envelope_prior_mask(h: int, w: int) -> np.ndarray:
    """Binary mask (255 = allowed) for the field-of-view corridor at this
    image size, interpolated from ENVELOPE_PRIOR. Cached per (h, w)."""
    key = (h, w)
    if key in _prior_cache:
        return _prior_cache[key]

    mask = np.zeros((h, w), dtype=np.uint8)
    rows = np.array([e[0] for e in ENVELOPE_PRIOR])
    xmins = np.array([e[1] for e in ENVELOPE_PRIOR])
    xmaxs = np.array([e[2] for e in ENVELOPE_PRIOR])
    y_start = int(rows[0] * h)
    for y in range(y_start, h):
        r = y / h
        xmin = int(np.interp(r, rows, xmins) * w)
        xmax = int(np.interp(r, rows, xmaxs) * w)
        mask[y, xmin:xmax + 1] = 255

    _prior_cache[key] = mask
    return mask


def _coarse_color_mask(frame_bgr: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Steps 1-4 of the docstring: Otsu on LAB 'a' + envelope prior + close +
    largest connected component. This is the coarse/fallback region -- its
    per-row leftmost/rightmost x feeds the edge-snap step below."""
    h, w = frame_bgr.shape[:2]
    a_channel = lab[:, :, 1]
    _, reddish = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    prior = build_envelope_prior_mask(h, w)
    reddish = cv2.bitwise_and(reddish, prior)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (MORPH_CLOSE_KERNEL_PX, MORPH_CLOSE_KERNEL_PX))
    closed = cv2.morphologyEx(reddish, cv2.MORPH_CLOSE, close_kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 1:
        return np.zeros_like(closed)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def _row_median_filter(values: np.ndarray, k: int) -> np.ndarray:
    """Median filter over valid (>=0) entries of a 1D per-row array,
    ignoring invalid (-1) neighbors instead of treating them as zero."""
    out = values.copy()
    half = k // 2
    for i in range(len(values)):
        if values[i] < 0:
            continue
        window = values[max(0, i - half):i + half + 1]
        window = window[window >= 0]
        if len(window):
            out[i] = np.median(window)
    return out


def segment_track_traditional(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Segment the whole running-track area using classical image processing:
    a coarse color+geometric-prior region (see _coarse_color_mask), refined
    per row by snapping to the track's white edge stripe where visible
    (see module docstring's "EDGE SNAP" step). Returns a binary mask
    (uint8, {0, 255}) the same size as frame_bgr.
    """
    h, w = frame_bgr.shape[:2]
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L = cv2.GaussianBlur(lab[:, :, 0], (5, 1), 0).astype(np.float32)

    coarse = _coarse_color_mask(frame_bgr, lab)
    rows_with_track = np.nonzero(coarse.any(axis=1))[0]
    if len(rows_with_track) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    y0, y1 = int(rows_with_track.min()), int(rows_with_track.max())

    left_edge = np.full(h, -1, dtype=np.float32)
    right_edge = np.full(h, -1, dtype=np.float32)

    def snap(peak_search_lo, peak_search_hi, coarse_x, walk_sign):
        seg = L[y, peak_search_lo:peak_search_hi]
        if len(seg) <= 3:
            return coarse_x
        peak_i = int(np.argmax(seg))
        peak_v = seg[peak_i]
        if peak_v - np.median(seg) <= SNAP_PROMINENCE:
            return coarse_x
        j = peak_i
        while 0 <= j + walk_sign < len(seg) and seg[j + walk_sign] > peak_v - SNAP_DROP:
            j += walk_sign
        candidate = peak_search_lo + j
        return candidate if abs(candidate - coarse_x) <= SNAP_MAX_SHIFT_PX else coarse_x

    for y in range(y0, y1 + 1):
        xs = np.nonzero(coarse[y])[0]
        if len(xs) == 0:
            continue
        lb, rb = int(xs.min()), int(xs.max())
        left_edge[y] = snap(max(0, lb - SNAP_WINDOW_PX), min(w, lb + SNAP_WINDOW_PX), lb, walk_sign=-1)
        right_edge[y] = snap(max(0, rb - SNAP_WINDOW_PX), min(w, rb + SNAP_WINDOW_PX), rb, walk_sign=+1)

    left_edge = _row_median_filter(left_edge, ROW_MEDIAN_FILTER)
    right_edge = _row_median_filter(right_edge, ROW_MEDIAN_FILTER)

    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(y0, y1 + 1):
        if left_edge[y] < 0 or right_edge[y] < 0:
            continue
        l = int(max(0, left_edge[y]))
        r = int(min(w - 1, right_edge[y]))
        if r > l:
            mask[y, l:r + 1] = 255

    return mask


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Quick-look traditional CV track segmentation on one image.")
    parser.add_argument("image", type=str)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    mask = segment_track_traditional(img)

    out_path = args.out or str(Path(args.image).with_suffix(".mask.png"))
    cv2.imwrite(out_path, mask)
    overlay = img.copy()
    overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.imwrite(str(Path(out_path).with_name(Path(out_path).stem + "_overlay.jpg")), overlay)
    print(f"Saved: {out_path}")
