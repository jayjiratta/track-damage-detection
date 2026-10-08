"""
Track ROI (whole running-track area) -- from track-segmentation-ver2.

  yolo         YOLOv26n segmentation (default; ver2's final comparison found it
               more accurate and far more consistent at the track boundary)
  yolo_r18     YOLO26-seg neck/head on a frozen ImageNet ResNet-18 backbone
               (shares its backbone with PaDiM, backbone="shared_r18")
  traditional  ver2's classical CV method (color + FOV prior + edge snap);
               ~24x less memory than YOLO, but less accurate at the boundary
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS = ROOT / "models" / "yolo26n_track_seg.pt"
YOLO_R18_WEIGHTS = ROOT / "models" / "yolo26_r18frozen_track_seg.pt"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traditional_segment import segment_track_traditional


class TrackROI:
    def __init__(self, method: str = "yolo"):
        if method not in ("yolo", "yolo_r18", "traditional"):
            raise ValueError(f"unknown ROI method: {method}")
        self.method = method
        self.model = None
        if method in ("yolo", "yolo_r18"):
            from ultralytics import YOLO
            self.model = YOLO(str(YOLO_WEIGHTS if method == "yolo" else YOLO_R18_WEIGHTS))

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        """Binary mask (uint8 {0,255}) of the whole track area."""
        if self.method == "traditional":
            return segment_track_traditional(img_bgr)

        h, w = img_bgr.shape[:2]
        # retina_masks=True: masks in original-image coordinates (the default
        # masks.data is letterboxed, so a plain resize would shift it)
        r = self.model.predict(img_bgr, verbose=False, retina_masks=True)[0]
        mask = np.zeros((h, w), np.uint8)
        if r.masks is None or len(r.masks.data) == 0:
            return mask
        for m in r.masks.data.cpu().numpy():
            mask[m > 0.5] = 255
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n > 2:
            mask = np.where(labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])), 255, 0).astype(np.uint8)
        return mask
