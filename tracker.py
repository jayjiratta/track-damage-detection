"""
Link damage regions across video frames so each physical damage is counted once.

Per frame, fragments of the same damage are first merged (merge_regions) and
small blobs are left out of counting. As the robot drives forward, a damage
region slides down the frame and grows (and sideways while turning). Each
merged box is matched to an existing track by box IoU, or failing that by
centre distance, as long as it hasn't jumped upward (things only move down /
toward the camera). A track is counted once it has been matched in
MIN_HITS frames, which filters flickers, and is dropped after MAX_MISSED
consecutive frames without a match.

This is a simple tracker for the demo; it has not been evaluated against
ground-truth damage identities.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MIN_HITS = 4
MAX_MISSED = 8
IOU_MATCH = 0.05
MAX_UPWARD_PX = 40
MERGE_GAP_PX = 40         # fragments of one crack closer than this are one region
COUNT_MIN_AREA_PX = 800   # smaller blobs (texture, leaves) are shown but never counted


def merge_regions(regions: list[dict]) -> list[list[int]]:
    """Union the boxes of regions whose boxes lie within MERGE_GAP_PX of each
    other (one crack often splits into several components), then keep merged
    boxes whose total damage area reaches COUNT_MIN_AREA_PX."""
    groups = [{"bbox": list(r["bbox"]), "area": r["area_px"]} for r in regions]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i]["bbox"], groups[j]["bbox"]
                gx = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
                gy = max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])
                if gx <= MERGE_GAP_PX and gy <= MERGE_GAP_PX:
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    groups[i] = {"bbox": [x0, y0, x1 - x0, y1 - y0], "area": groups[i]["area"] + groups[j]["area"]}
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    return [g["bbox"] for g in groups if g["area"] >= COUNT_MIN_AREA_PX]


@dataclass
class Track:
    id: int
    bbox: list[int]
    hits: int = 1
    missed: int = 0
    confirmed: bool = False
    uid: int = 0          # display number, assigned in order of confirmation
    history: list[list[int]] = field(default_factory=list)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-9)


def _centre(b):
    return b[0] + b[2] / 2.0, b[1] + b[3] / 2.0


class DamageTracker:
    def __init__(self):
        self.tracks: list[Track] = []
        self.next_id = 1
        self.unique_count = 0

    def update(self, boxes: list[list[int]]) -> list[Track]:
        """boxes: [x, y, w, h] per detected damage region this frame.
        Returns the confirmed tracks matched in this frame."""
        pairs = []
        for ti, t in enumerate(self.tracks):
            tcx, tcy = _centre(t.bbox)
            reach = max(60.0, 0.75 * (t.bbox[2] ** 2 + t.bbox[3] ** 2) ** 0.5)
            for di, b in enumerate(boxes):
                cx, cy = _centre(b)
                if cy < tcy - MAX_UPWARD_PX:
                    continue
                iou = _iou(t.bbox, b)
                dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
                if iou >= IOU_MATCH or dist <= reach:
                    pairs.append((-iou, dist, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self.tracks[ti]
            t.bbox = list(boxes[di])
            t.history.append(t.bbox)
            t.hits += 1
            t.missed = 0
            if not t.confirmed and t.hits >= MIN_HITS:
                t.confirmed = True
                self.unique_count += 1
                t.uid = self.unique_count
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
        for di, b in enumerate(boxes):
            if di not in used_d:
                self.tracks.append(Track(self.next_id, list(b), history=[list(b)]))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.missed <= MAX_MISSED]
        return [t for t in self.tracks if t.confirmed and t.missed == 0]
