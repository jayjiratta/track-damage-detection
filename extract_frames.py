"""
Build PaDiM's normal-frame set: every STEP-th frame of the inspection video,
skipping frame ranges that contain known damage (so it is not learned as
normal). Frames are named frame_<video index>.jpg.

The whole route is sampled on purpose: an earlier set taken from the first
5 minutes only (frames 0-9000 of a 21.6-minute run) left the later sections
-- a shaded stretch and the section with the real crack -- outside PaDiM's
normal distribution.

Usage:
    python extract_frames.py <video> [--step 20] [--exclude 21780-22000] [--out data/normal_frames]
"""
import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--exclude", nargs="*", default=["21780-22000"],
                    help="frame ranges with known damage, e.g. 21500-22300")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "normal_frames")
    args = ap.parse_args()

    excluded = [tuple(int(v) for v in r.split("-")) for r in args.exclude]
    args.out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    i = n = 0
    while cap.grab():
        if i % args.step == 0 and not any(lo <= i <= hi for lo, hi in excluded):
            _, img = cap.retrieve()
            cv2.imwrite(str(args.out / f"frame_{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
        i += 1
    print(f"{n} frames (of {i}) -> {args.out}")


if __name__ == "__main__":
    main()
