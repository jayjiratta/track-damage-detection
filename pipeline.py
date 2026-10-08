"""
Track damage detection = track ROI (YOLOv26 segmentation) + PaDiM anomaly map.

  1. ROI      whole-track mask (roi/roi.py), eroded by ROI_ERODE_PX so PaDiM
              patches that straddle the track border (half track, half grass)
              don't count as damage.
  2. Anomaly  PaDiM Mahalanobis-distance map (padim/padim.py).
  3. Damage   anomaly > threshold inside the ROI, small open, components
              smaller than MIN_DAMAGE_AREA_PX dropped.
  4. Regions  connected damage regions (area, bbox) -> tracker.py links them
              across frames and counts each damage once.

Two systems (the project's comparison):

  method 1  roi_method="yolo",     backbone="resnet18"
            separate backbones: YOLOv26n for the ROI and a separate ResNet-18
            for PaDiM -- two CNN forward passes per frame.
  method 2  roi_method="yolo_r18", backbone="shared_r18"
            shared backbone: the track model is a YOLO26-seg neck/head on a
            frozen ImageNet ResNet-18 and PaDiM uses that backbone's layer1-3
            from the same forward pass -- one CNN pass per frame.

No rule-based filtering: everything above the threshold inside the ROI is
reported as damage.
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

BACKBONES = ("resnet18", "shared_r18")
SHARED_ROI = {"shared_r18": "yolo_r18"}   # the shared backbone lives in this ROI model
METHODS = {1: ("yolo", "resnet18"), 2: ("yolo_r18", "shared_r18")}   # (roi_method, backbone)
THRESHOLD_PATH = ROOT / "models" / "threshold.json"

ROI_ERODE_PX = 15
MIN_DAMAGE_AREA_PX = 100


def padim_weights(backbone: str) -> Path:
    return ROOT / "models" / f"padim_{backbone}.pt"


def threshold_key(roi_method: str, backbone: str) -> str:
    return f"{roi_method}+{backbone}"


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


class DamagePipeline:
    def __init__(self, roi_method: str = "yolo", backbone: str = "resnet18", threshold: float | None = None):
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone: {backbone}")
        if backbone in SHARED_ROI and roi_method != SHARED_ROI[backbone]:
            raise ValueError(f"backbone='{backbone}' needs roi_method='{SHARED_ROI[backbone]}'")
        self.roi_method = roi_method
        self.backbone = backbone
        self.roi = TrackROI(roi_method)
        self.padim = PaDiM.load(padim_weights(backbone), yolo=self.roi.model if backbone in SHARED_ROI else None)
        if threshold is None:
            key = threshold_key(roi_method, backbone)
            saved = json.loads(THRESHOLD_PATH.read_text()) if THRESHOLD_PATH.exists() else {}
            if key not in saved:
                raise KeyError(f"no calibrated threshold for {key} -- run calibrate.py")
            threshold = saved[key]["threshold"]
        self.threshold = float(threshold)

    def roi_and_anomaly(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        roi = erode_roi(self.roi(img_bgr))
        if self.backbone in SHARED_ROI:    # reuse the features of the ROI forward pass
            anomaly = self.padim.score(img_bgr, feats=self.padim.hook.get())
        else:
            anomaly = self.padim.score(img_bgr)
        return roi, anomaly

    def __call__(self, img_bgr: np.ndarray) -> DamageResult:
        roi, anomaly = self.roi_and_anomaly(img_bgr)

        dmg = ((anomaly > self.threshold) & (roi > 0)).astype(np.uint8) * 255
        dmg = cv2.morphologyEx(dmg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        n, labels, stats, _ = cv2.connectedComponentsWithStats(dmg, connectivity=8)
        damage = np.zeros_like(dmg)
        regions = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < MIN_DAMAGE_AREA_PX:
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
