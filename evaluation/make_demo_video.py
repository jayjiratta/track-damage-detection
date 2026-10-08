"""
Demo video: track mask + PaDiM anomaly map + damage mask + tracking / unique count,
on a few segments of the recording, each introduced by a title card. The unique
count restarts per segment (segments are different places on the track).
Also writes the unique count of each segment to results/demo/counts_<method>.csv.

  --method 2 (default)  shared frozen ResNet-18: one backbone for the track mask and PaDiM
  --method 1            separate backbones: YOLOv26n track mask + ResNet-18 PaDiM

Usage (from track-damage-detection/):
    python evaluation/make_demo_video.py <video> [--method 2] [--out results/demo/demo_pipeline.mp4]
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
from pipeline import DamagePipeline, render_summary
from tracker import DamageTracker, merge_regions

METHODS = {
    1: ("yolo", "resnet18", "Method 1: YOLOv26n track mask + separate ResNet-18 PaDiM"),
    2: ("yolo_r18", "shared_r18", "Method 2: shared frozen ResNet-18 (track mask + PaDiM)"),
}
# (start frame, processed frames, title) in capture_20260806_121427.mp4
SEGMENTS = [
    (1900, 120, "1/4  Normal track surface"),
    (22350, 120, "2/4  Lane line on a curve"),
    (21760, 150, "3/4  Real crack: detection + tracking + unique count"),
    (4100, 120, "4/4  Road markings (rare at each image position)"),
]
STRIDE = 2
TITLE_SECONDS = 2.0


def title_card(size, text, sub):
    card = np.full((size[1], size[0], 3), (25, 25, 35), np.uint8)
    for i, (t, s, c) in enumerate([(text, 1.2, (255, 255, 255)), (sub, 0.7, (180, 180, 180))]):
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
        cv2.putText(card, t, ((size[0] - tw) // 2, size[1] // 2 - 10 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, s, c, 2)
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--method", type=int, default=2, choices=[1, 2])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "demo" / "demo_pipeline.mp4")
    ap.add_argument("--counts-only", action="store_true", help="write the counts CSV, no video")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    roi, backbone, label = METHODS[args.method]
    pipe = DamagePipeline(roi, backbone)
    cap = cv2.VideoCapture(str(args.video))
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) / STRIDE
    writer = None
    rows = []
    for start, n, title in SEGMENTS:
        tracker = DamageTracker()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx, done, frames, flagged = start, 0, [], 0
        while done < n:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start) % STRIDE == 0:
                res = pipe(frame)
                tracks = tracker.update(merge_regions(res.regions))
                flagged += bool(res.regions)
                if not args.counts_only:
                    panel = render_summary(frame, res, f"{title}   |   frame {idx}", tracks=tracks,
                                           unique_count=tracker.unique_count)
                    frames.append(cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2)))
                done += 1
            idx += 1
        rows.append({"segment": title, "start_frame": start, "end_frame": idx - 1, "processed_frames": done,
                     "frames_with_damage_regions": flagged, "unique_count": tracker.unique_count})
        print(f"{title}: {done} frames, unique damage = {tracker.unique_count}")
        if args.counts_only or not frames:
            continue
        size = frames[0].shape[1::-1]
        if writer is None:
            writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        card = title_card(size, title, f"{label}   |   frames {start}-{idx}")
        for _ in range(int(TITLE_SECONDS * fps)):
            writer.write(card)
        for f in frames:
            writer.write(f)
    cap.release()
    if writer is not None:
        writer.release()
        print(f"saved: {args.out}")
    out_csv = args.out.parent / f"counts_method{args.method}.csv"
    with open(out_csv, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
