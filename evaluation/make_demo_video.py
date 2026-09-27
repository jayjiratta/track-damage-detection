"""
Demo video: track ROI (YOLOv26) + PaDiM damage map + tracking / unique count,
on a few segments of a recording, each introduced by a title card. The unique
count restarts per segment (segments are different places on the track).

Usage (from track-damage-pipeline/):
    python evaluation/make_demo_video.py <video> [--out results/demo/demo_pipeline.mp4]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import DamagePipeline, render_summary
from tracker import DamageTracker, merge_regions

# (start frame, processed frames, title) in capture_20260806_121427.mp4
SEGMENTS = [
    (1900, 120, "1/4  Normal track surface"),
    (4100, 120, "2/4  Road markings (paint filter)"),
    (21760, 150, "3/4  Real crack: detection + tracking + unique count"),
    (16150, 120, "4/4  Limitation: shaded section not in PaDiM training data"),
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
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "demo" / "demo_pipeline.mp4")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    pipe = DamagePipeline("yolo")
    cap = cv2.VideoCapture(str(args.video))
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) / STRIDE
    writer = None
    for start, n, title in SEGMENTS:
        tracker = DamageTracker()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx, done, frames = start, 0, []
        while done < n:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start) % STRIDE == 0:
                res = pipe(frame)
                tracks = tracker.update(merge_regions(res.regions))
                panel = render_summary(frame, res, f"{title}   |   frame {idx}", tracks=tracks,
                                       unique_count=tracker.unique_count)
                frames.append(cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2)))
                done += 1
            idx += 1
        if not frames:
            continue
        size = frames[0].shape[1::-1]
        if writer is None:
            writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        card = title_card(size, title, f"YOLOv26 track ROI + PaDiM (ResNet18) + tracking   |   frames {start}-{idx}")
        for _ in range(int(TITLE_SECONDS * fps)):
            writer.write(card)
        for f in frames:
            writer.write(f)
        print(f"{title}: {len(frames)} frames, unique damage = {tracker.unique_count}")
    cap.release()
    if writer is not None:
        writer.release()
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
