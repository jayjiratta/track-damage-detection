"""
Track damage detection = track ROI (ver2) + PaDiM anomaly map (damage-detection).

  1. ROI      whole-track mask (roi/roi.py), eroded by ROI_ERODE_PX so PaDiM
              patches that straddle the track border (half track, half grass)
              don't count as damage.
  2. Anomaly  PaDiM Mahalanobis-distance map (padim/padim.py).
  3. Damage   anomaly > threshold inside the ROI, small open, components
              smaller than MIN_DAMAGE_AREA_PX dropped, and components that are
              mostly white paint dropped (see PAINT_* below).
  4. Count    connected damage regions (area, bbox).

Paint filter: PaDiM models each patch position separately, so painted
markings that are rare and move around the frame (chevrons, distance text)
score as anomalies even though they're normal track features. A damage
component is dropped if more than PAINT_MAX_FRACTION of its area is white
paint (low-saturation, bright pixels -- same HSV rule the earlier LBP demo
used), dilated by PAINT_DILATE_PX so the blob's smoothed spill-over around
the paint also counts. Trade-off: real damage lying mostly on a painted
marking is dropped too.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "roi"))
sys.path.insert(0, str(ROOT / "padim"))
from roi import TrackROI
from padim import PaDiM

PADIM_WEIGHTS = ROOT / "models" / "padim_resnet18.pt"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"

ROI_ERODE_PX = 15
MIN_DAMAGE_AREA_PX = 100

PAINT_HSV_LO = (0, 0, 180)
PAINT_HSV_HI = (180, 40, 255)
PAINT_DILATE_PX = 9
PAINT_MAX_FRACTION = 0.30


@dataclass
class DamageResult:
    roi: np.ndarray
    anomaly: np.ndarray
    damage: np.ndarray
    threshold: float
    regions: list[dict] = field(default_factory=list)


def erode_roi(roi: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ROI_ERODE_PX + 1,) * 2)
    return cv2.erode(roi, k)


def paint_mask(img_bgr: np.ndarray) -> np.ndarray:
    white = cv2.inRange(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), PAINT_HSV_LO, PAINT_HSV_HI)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * PAINT_DILATE_PX + 1,) * 2)
    return cv2.dilate(white, k)


class DamagePipeline:
    def __init__(self, roi_method: str = "yolo", threshold: float | None = None,
                 paint_filter: bool = True):
        self.roi_method = roi_method
        self.paint_filter = paint_filter
        self.roi = TrackROI(roi_method)
        self.padim = PaDiM.load(PADIM_WEIGHTS)
        if threshold is None:
            if not THRESHOLD_PATH.exists():
                raise FileNotFoundError(f"{THRESHOLD_PATH} missing -- run calibrate.py first")
            saved = json.loads(THRESHOLD_PATH.read_text())
            if roi_method not in saved:
                raise KeyError(f"no calibrated threshold for roi_method={roi_method} -- run calibrate.py")
            threshold = saved[roi_method]["threshold"]
        self.threshold = float(threshold)

    def __call__(self, img_bgr: np.ndarray) -> DamageResult:
        roi = erode_roi(self.roi(img_bgr))
        anomaly = self.padim.score(img_bgr)

        dmg = ((anomaly > self.threshold) & (roi > 0)).astype(np.uint8) * 255
        dmg = cv2.morphologyEx(dmg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        paint = paint_mask(img_bgr) > 0 if self.paint_filter else None
        n, labels, stats, _ = cv2.connectedComponentsWithStats(dmg, connectivity=8)
        damage = np.zeros_like(dmg)
        regions = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < MIN_DAMAGE_AREA_PX:
                continue
            if paint is not None and paint[labels == i].mean() > PAINT_MAX_FRACTION:
                continue
            damage[labels == i] = 255
            regions.append({"id": len(regions) + 1, "area_px": area, "bbox": [x, y, w, h],
                            "max_score": float(anomaly[labels == i].max())})
        return DamageResult(roi, anomaly, damage, self.threshold, regions)


# ------------------------------------------------------------------ drawing
def _label(img, text, color):
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), (20, 20, 30), -1)
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img


def render_summary(img: np.ndarray, res: DamageResult, title: str,
                   tracks=None, unique_count: int = 0) -> np.ndarray:
    """2x2 panel: ROI / anomaly heatmap in ROI / damage mask / counted regions
    (or, for video with a DamageTracker, tracked IDs + unique count)."""
    p1 = img.copy()
    p1[res.roi > 0] = (0.6 * p1[res.roi > 0] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)

    p2 = img.copy()
    u8 = np.clip(res.anomaly / (res.threshold * 1.5) * 255, 0, 255).astype(np.uint8)
    hm = cv2.addWeighted(img, 0.45, cv2.applyColorMap(u8, cv2.COLORMAP_JET), 0.55, 0)
    p2[res.roi > 0] = hm[res.roi > 0]

    p3 = img.copy()
    p3[res.damage > 0] = (0, 0, 255)
    cnts, _ = cv2.findContours(res.damage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p3, cnts, -1, (0, 0, 255), 2)

    p4 = img.copy()
    roi_px = int((res.roi > 0).sum())
    dmg_px = int((res.damage > 0).sum())
    ratio = dmg_px / roi_px * 100 if roi_px else 0.0
    if tracks is None:
        for r in res.regions:
            x, y, w, h = r["bbox"]
            cv2.rectangle(p4, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(p4, str(r["id"]), (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        p4_title = f"4: Regions={len(res.regions)}  dmg={dmg_px:,}px ({ratio:.2f}% of ROI)"
    else:
        for t in tracks:
            x, y, w, h = t.bbox
            cv2.rectangle(p4, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(p4, f"ID {t.uid}", (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        p4_title = f"4: Unique damage so far = {unique_count}  (in view: {len(tracks)})"

    panels = [_label(p1, "1: Track ROI", (0, 255, 0)),
              _label(p2, "2: PaDiM anomaly (in ROI)", (255, 255, 0)),
              _label(p3, "3: Damage mask", (0, 0, 255)),
              _label(p4, p4_title, (0, 165, 255))]
    grid = cv2.vconcat([cv2.hconcat(panels[:2]), cv2.hconcat(panels[2:])])
    header = np.full((50, grid.shape[1], 3), (20, 20, 30), np.uint8)
    cv2.putText(header, title, (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return cv2.vconcat([header, grid])
