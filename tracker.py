"""
Link damage regions across video frames so each physical damage is counted once.

1. Per frame, merge_regions groups the damage components into regions using the
   coarsest PaDiM feature cell (layer3, stride 16, in image pixels; see
   DamageResult.coarse_cell) as the spatial unit:
     - components whose boxes are at most one cell apart are one region: at
       that distance they share a layer3 feature, so the anomaly map cannot
       separate them (one crack often splits into several components);
     - a region is counted only if its damage area covers at least one cell,
       i.e. at least one full layer3 feature is anomalous.
2. DamageTracker is SORT (Bewley et al., "Simple Online and Realtime Tracking",
   ICIP 2016): a constant-velocity Kalman filter on [cx, cy, area, aspect] per
   track, Hungarian assignment on IoU between Kalman predictions and the
   frame's regions. Filter noise and the defaults iou_threshold=0.3,
   min_hits=3, max_age=1 are those of the authors' reference implementation
   (github.com/abewley/sort).
3. A track is counted once, the first time its hit streak reaches min_hits.

The counts have not been checked against reference counts from an inspector.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_THRESHOLD = 0.3   # SORT defaults
MIN_HITS = 3
MAX_AGE = 1


def merge_regions(regions: list[dict], cell: tuple[float, float]) -> list[list[int]]:
    """Union the boxes of regions at most one cell (w, h) apart, then keep merged
    boxes whose total damage area reaches one cell area. Returns [x, y, w, h]."""
    cw, ch = cell
    groups = [{"bbox": list(r["bbox"]), "area": r["area_px"]} for r in regions]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i]["bbox"], groups[j]["bbox"]
                gx = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
                gy = max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])
                if gx <= cw and gy <= ch:
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    groups[i] = {"bbox": [x0, y0, x1 - x0, y1 - y0], "area": groups[i]["area"] + groups[j]["area"]}
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    return [g["bbox"] for g in groups if g["area"] >= cw * ch]


def _to_z(b):                      # [x, y, w, h] -> [cx, cy, area, aspect]
    x, y, w, h = b
    return np.array([x + w / 2.0, y + h / 2.0, w * h, w / float(h)], dtype=float)


def _to_box(xs):                   # [cx, cy, area, aspect, ...] -> [x, y, w, h]
    s, r = max(xs[2], 1e-6), max(xs[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return [int(round(xs[0] - w / 2)), int(round(xs[1] - h / 2)), int(round(w)), int(round(h))]


def _iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


class _KalmanBox:
    """Constant-velocity Kalman filter of SORT's reference implementation."""

    def __init__(self, box):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)
        self.R = np.eye(4); self.R[2:, 2:] *= 10.0
        self.P = np.eye(7); self.P[4:, 4:] *= 1000.0; self.P *= 10.0
        self.Q = np.eye(7); self.Q[-1, -1] *= 0.01; self.Q[4:, 4:] *= 0.01
        self.x = np.zeros(7); self.x[:4] = _to_z(box)

    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _to_box(self.x)

    def update(self, box):
        y = _to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P


class Track:
    def __init__(self, box, tid):
        self.kf = _KalmanBox(box)
        self.id = tid
        self.bbox = list(box)
        self.hit_streak = 1
        self.time_since_update = 0
        self.uid = 0               # display number, assigned when the track is counted


class DamageTracker:
    def __init__(self, iou_threshold=IOU_THRESHOLD, min_hits=MIN_HITS, max_age=MAX_AGE):
        self.iou_threshold, self.min_hits, self.max_age = iou_threshold, min_hits, max_age
        self.tracks: list[Track] = []
        self.next_id = 1
        self.unique_count = 0

    def update(self, boxes: list[list[int]]) -> list[Track]:
        """boxes: [x, y, w, h] per merged damage region this frame.
        Returns the counted tracks matched in this frame."""
        preds = [t.kf.predict() for t in self.tracks]
        for t in self.tracks:
            if t.time_since_update > 0:
                t.hit_streak = 0
            t.time_since_update += 1
        matched = set()
        if preds and boxes:
            iou = np.array([[_iou(p, b) for b in boxes] for p in preds])
            for ti, di in zip(*linear_sum_assignment(-iou)):
                if iou[ti, di] >= self.iou_threshold:
                    t = self.tracks[ti]
                    t.kf.update(boxes[di])
                    t.bbox = list(boxes[di])
                    t.time_since_update = 0
                    t.hit_streak += 1
                    matched.add(di)
        for di, b in enumerate(boxes):
            if di not in matched:
                self.tracks.append(Track(b, self.next_id))
                self.next_id += 1
        out = []
        for t in self.tracks:
            if t.time_since_update == 0 and t.hit_streak >= self.min_hits:
                if t.uid == 0:
                    self.unique_count += 1
                    t.uid = self.unique_count
                out.append(t)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return out
