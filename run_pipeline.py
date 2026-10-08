"""
Run track damage detection on an image, a folder of images, or a video.

Usage:
    python run_pipeline.py <image|folder|video> [--method 2|1] [--out results]
    python run_pipeline.py <image|folder> --method 1 --roi traditional   # method 1 with the traditional ROI
    python run_pipeline.py video.mp4 --start 0 --stride 2 --max-frames 900

Per image (folder/single): results/<stem>/summary_4panel.jpg + regions.json
Video: results/<stem>_<start>_damage.mp4 (2x2 panel per processed frame, damage
tracked across frames with a unique count) + <stem>_<start>_regions.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pipeline import METHODS, DamagePipeline, render_summary
from tracker import DamageTracker, merge_regions

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def run_images(pipe, paths, out_dir):
    for p in paths:
        img = cv2.imread(str(p))
        res = pipe(img)
        d = out_dir / p.stem
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "summary_4panel.jpg"), render_summary(img, res, f"{p.name}  |  ROI: {pipe.roi_method}"))
        (d / "regions.json").write_text(json.dumps(
            {"image": p.name, "roi_method": pipe.roi_method, "threshold": res.threshold,
             "roi_px": int((res.roi > 0).sum()), "damage_px": int((res.damage > 0).sum()),
             "regions": res.regions}, indent=2))
        print(f"{p.name}: {len(res.regions)} regions, {int((res.damage > 0).sum()):,} damage px")


def run_video(pipe, path, out_dir, max_frames, start=0, stride=1):
    """Process every `stride`-th frame from `start`; regions are linked across
    processed frames by DamageTracker to count each damage once."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{path.stem}_{start}"
    tracker = DamageTracker()
    writer = None
    with open(out_dir / f"{tag}_regions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "n_regions", "damage_px", "roi_px", "track_ids", "unique_count"])
        done, idx = 0, start
        while done < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start) % stride == 0:
                res = pipe(frame)
                tracks = tracker.update(merge_regions(res.regions))
                panel = render_summary(frame, res, f"{path.name}  frame {idx}  |  ROI: {pipe.roi_method}",
                                       tracks=tracks, unique_count=tracker.unique_count)
                panel = cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2))
                if writer is None:
                    writer = cv2.VideoWriter(str(out_dir / f"{tag}_damage.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, panel.shape[1::-1])
                writer.write(panel)
                w.writerow([idx, len(res.regions), int((res.damage > 0).sum()), int((res.roi > 0).sum()),
                            " ".join(str(t.uid) for t in tracks), tracker.unique_count])
                done += 1
                if done % 100 == 0:
                    print(f"  {done} frames")
            idx += 1
    cap.release()
    if writer is not None:
        writer.release()
    print(f"saved: {out_dir / (tag + '_damage.mp4')}  (unique damage regions: {tracker.unique_count})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--method", type=int, default=2, choices=[1, 2],
                    help="1: separate backbones (YOLOv26n + ResNet-18), 2: shared frozen ResNet-18")
    ap.add_argument("--roi", choices=["yolo", "traditional"], help="method 1 only: override the track ROI")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--max-frames", type=int, default=900, help="video only: frames to process")
    ap.add_argument("--start", type=int, default=0, help="video only: first frame")
    ap.add_argument("--stride", type=int, default=1, help="video only: process every N-th frame")
    args = ap.parse_args()

    roi, backbone = METHODS[args.method]
    if args.roi:
        if args.method != 1:
            ap.error("--roi only applies to --method 1")
        roi = args.roi
    pipe = DamagePipeline(roi, backbone)
    if args.input.is_dir():
        run_images(pipe, sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMG_EXT), args.out)
    elif args.input.suffix.lower() in VID_EXT:
        run_video(pipe, args.input, args.out, args.max_frames, args.start, args.stride)
    else:
        run_images(pipe, [args.input], args.out)


if __name__ == "__main__":
    main()
