"""
Demo video on a few segments of the recording, each introduced by a title card.
The unique count restarts per segment (segments are different places on the track).
Also writes the unique count of each segment to results/demo/counts_method<n>.csv.

  default        method 1 vs method 2 on the same frames: per method, the PaDiM
                 anomaly map (inside its track mask) and the damage mask with
                 tracked regions + unique count -> results/demo/demo_pipeline.mp4
  --method 1|2   one method, full 2x2 panel (track mask / anomaly / damage / tracking)
                 -> results/demo/demo_method<n>.mp4

  method 1  separate backbones: YOLOv26n track mask + ResNet-18 PaDiM
  method 2  shared frozen ResNet-18: one backbone for the track mask and PaDiM

Usage (from track-damage-detection/):
    python evaluation/make_demo_video.py <video>
    python evaluation/make_demo_video.py <video> --method 2
    python evaluation/make_demo_video.py <video> --method 1 --counts-only
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import METHODS, DamagePipeline, render_summary
from tracker import DamageTracker, merge_regions

LABELS = {1: "Method 1: separate backbones (YOLOv26n + ResNet-18)",
          2: "Method 2: shared frozen ResNet-18"}
# (start frame, processed frames, title) in capture_20260806_121427.mp4
SEGMENTS = [
    (1900, 120, "1/4  Normal track surface"),
    (22350, 120, "2/4  Lane line on a curve"),
    (21760, 150, "3/4  Real crack: detection + tracking + unique count"),
    (4100, 120, "4/4  Road markings (rare at each image position)"),
]
STRIDE = 2
TITLE_SECONDS = 2.0
PANEL_W, PANEL_H = 640, 360
METHOD_COLORS = {1: (230, 160, 60), 2: (60, 110, 240)}    # BGR: blue-ish / orange-ish


def title_card(size, text, sub):
    card = np.full((size[1], size[0], 3), (25, 25, 35), np.uint8)
    for i, (t, s, c) in enumerate([(text, 1.2, (255, 255, 255)), (sub, 0.7, (180, 180, 180))]):
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
        cv2.putText(card, t, ((size[0] - tw) // 2, size[1] // 2 - 10 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, s, c, 2)
    return card


def _tag(img, text, color):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (20, 20, 30), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return img


def compare_row(img, res, tracks, unique_count, method):
    """[anomaly map inside the track mask | damage mask + tracked regions] for one method."""
    heat = img.copy()
    u8 = np.clip(res.anomaly / (res.threshold * 1.5) * 255, 0, 255).astype(np.uint8)
    hm = cv2.addWeighted(img, 0.45, cv2.applyColorMap(u8, cv2.COLORMAP_JET), 0.55, 0)
    heat[res.roi > 0] = hm[res.roi > 0]

    dmg = img.copy()
    cnts, _ = cv2.findContours(res.roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dmg, cnts, -1, (0, 200, 0), 3)
    dmg[res.damage > 0] = (0, 0, 255)
    for t in tracks:
        x, y, w, h = t.bbox
        cv2.rectangle(dmg, (x, y), (x + w, y + h), (0, 165, 255), 3)
        cv2.putText(dmg, f"ID {t.uid}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

    color = METHOD_COLORS[method]
    heat = _tag(cv2.resize(heat, (PANEL_W, PANEL_H)), f"Method {method}  |  PaDiM anomaly map (inside track mask)", color)
    dmg = _tag(cv2.resize(dmg, (PANEL_W, PANEL_H)),
               f"Method {method}  |  damage mask + tracking   unique = {unique_count}", color)
    row = cv2.hconcat([heat, dmg])
    cv2.rectangle(row, (0, 0), (row.shape[1] - 1, row.shape[0] - 1), color, 2)
    return row


def compare_frame(img, title, idx, rows):
    header = np.full((40, PANEL_W * 2, 3), (20, 20, 30), np.uint8)
    cv2.putText(header, f"{title}   |   frame {idx}", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return cv2.vconcat([header] + rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--method", type=int, choices=[1, 2], help="one method only (default: compare 1 vs 2)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--counts-only", action="store_true", help="write the counts CSV, no video")
    args = ap.parse_args()
    methods = [args.method] if args.method else [1, 2]
    out = args.out or ROOT / "results" / "demo" / (f"demo_method{args.method}.mp4" if args.method else "demo_pipeline.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    pipes = {m: DamagePipeline(*METHODS[m]) for m in methods}
    cap = cv2.VideoCapture(str(args.video))
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) / STRIDE
    writer = None
    rows = {m: [] for m in methods}
    for start, n, title in SEGMENTS:
        trackers = {m: DamageTracker() for m in methods}
        flagged = {m: 0 for m in methods}
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx, done, frames = start, 0, []
        while done < n:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start) % STRIDE == 0:
                parts = []
                for m in methods:
                    res = pipes[m](frame)
                    tracks = trackers[m].update(merge_regions(res.regions, res.coarse_cell))
                    flagged[m] += bool(res.regions)
                    if args.counts_only:
                        continue
                    if args.method:
                        panel = render_summary(frame, res, f"{title}   |   frame {idx}", tracks=tracks,
                                               unique_count=trackers[m].unique_count)
                        parts.append(cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2)))
                    else:
                        parts.append(compare_row(frame, res, tracks, trackers[m].unique_count, m))
                if parts:
                    frames.append(parts[0] if args.method else compare_frame(frame, title, idx, parts))
                done += 1
            idx += 1
        for m in methods:
            rows[m].append({"segment": title, "start_frame": start, "end_frame": idx - 1, "processed_frames": done,
                            "frames_with_damage_regions": flagged[m], "unique_count": trackers[m].unique_count})
        print(f"{title}: {done} frames, unique damage = " + ", ".join(f"method {m}: {trackers[m].unique_count}" for m in methods))
        if args.counts_only or not frames:
            continue
        size = frames[0].shape[1::-1]
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        sub = LABELS[args.method] if args.method else "Method 1 (separate backbones) vs Method 2 (shared frozen ResNet-18)"
        card = title_card(size, title, f"{sub}   |   frames {start}-{idx}")
        for _ in range(int(TITLE_SECONDS * fps)):
            writer.write(card)
        for f in frames:
            writer.write(f)
    cap.release()
    if writer is not None:
        writer.release()
        print(f"saved: {out}")
    for m in methods:
        out_csv = out.parent / f"counts_method{m}.csv"
        with open(out_csv, "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[m][0])); w.writeheader(); w.writerows(rows[m])
        print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
