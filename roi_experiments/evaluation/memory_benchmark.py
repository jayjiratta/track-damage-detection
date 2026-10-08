"""
Memory Footprint Benchmark — YOLOv26n vs Traditional CV
==========================================================
Measures whole-process RSS memory (via psutil, same approach as
track-segmentation/benchmark/run_benchmark.py) while each method runs
inference over the test-set images, plus per-image latency.

This does NOT read ground-truth labels and does not affect model
selection/tuning -- it is a pure resource-usage measurement, run after
the accuracy evaluation (run_final_evaluation.py) as part of the final
test-set evaluation.

Each method is run in its own subprocess (like the original project) so
one method's resident memory (e.g. YOLO's loaded PyTorch/CUDA context)
doesn't contaminate the other's reading.

Output:
  report/memory_yolo_log.csv
  report/memory_traditional_log.csv
  columns: frame_idx, image, latency_ms, memory_mb

Usage:
    py evaluation/memory_benchmark.py --method all
    py evaluation/memory_benchmark.py --method yolo
    py evaluation/memory_benchmark.py --method traditional
    py evaluation/memory_benchmark.py --method yolo --yolo-device cpu   # -> memory_yolo_cpu_log.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "Lane Segmentation ver2.v1i.yolo26"
TEST_IMG_DIR = DATASET_DIR / "test" / "images"
REPORT_DIR = PROJECT_DIR / "report"
WEIGHTS_PATH = PROJECT_DIR.parent / "models" / "yolo26n_track_seg.pt"

WARMUP_FRAMES = 5   # skip first N images (model/JIT warmup), same convention as run_benchmark.py


def get_rss_mb() -> float:
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def benchmark_yolo(image_paths, out_csv: Path, device: str | None = None):
    from ultralytics import YOLO
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(WEIGHTS_PATH))
    model.to(device)

    rows = []
    for idx, img_path in enumerate(image_paths):
        frame = cv2.imread(str(img_path))
        t0 = time.perf_counter()
        model.predict(frame, imgsz=640, conf=0.25, verbose=False, device=device, retina_masks=True)
        latency_ms = (time.perf_counter() - t0) * 1000
        mem_mb = get_rss_mb()
        if idx >= WARMUP_FRAMES:
            rows.append({"frame_idx": idx, "image": img_path.stem,
                          "latency_ms": round(latency_ms, 3), "memory_mb": round(mem_mb, 2)})

    _write_csv(out_csv, rows)


def benchmark_traditional(image_paths, out_csv: Path):
    sys.path.insert(0, str(PROJECT_DIR.parent / "roi"))
    from traditional_segment import segment_track_traditional

    rows = []
    for idx, img_path in enumerate(image_paths):
        frame = cv2.imread(str(img_path))
        t0 = time.perf_counter()
        segment_track_traditional(frame)
        latency_ms = (time.perf_counter() - t0) * 1000
        mem_mb = get_rss_mb()
        if idx >= WARMUP_FRAMES:
            rows.append({"frame_idx": idx, "image": img_path.stem,
                          "latency_ms": round(latency_ms, 3), "memory_mb": round(mem_mb, 2)})

    _write_csv(out_csv, rows)


def _write_csv(out_csv: Path, rows: list[dict]):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "image", "latency_ms", "memory_mb"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out_csv}  ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", nargs="+", choices=["yolo", "traditional", "all"], default=["all"])
    parser.add_argument("--yolo-device", choices=["cuda", "cpu"], default=None,
                        help="YOLO device (default: cuda if available); cpu compares with the CPU-only traditional method")
    args = parser.parse_args()
    methods = ["yolo", "traditional"] if "all" in args.method else args.method

    if len(methods) > 1:
        # Separate subprocess per method -- see module docstring.
        for method in methods:
            print(f"\n{'='*55}\n  Launching subprocess for: {method.upper()}\n{'='*55}")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--method", method])
            if result.returncode != 0:
                sys.exit(result.returncode)
        print("\nAll memory benchmarks done.")
        return

    image_paths = sorted(TEST_IMG_DIR.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {TEST_IMG_DIR}")

    method = methods[0]
    print(f"\n{'='*55}\n  Benchmarking memory: {method.upper()}\n{'='*55}")
    out_csv = REPORT_DIR / (f"memory_{method}_cpu_log.csv" if method == "yolo" and args.yolo_device == "cpu"
                            else f"memory_{method}_log.csv")

    if method == "yolo":
        if not WEIGHTS_PATH.exists():
            print(f"  SKIP -- weights not found: {WEIGHTS_PATH}")
            return
        benchmark_yolo(image_paths, out_csv, args.yolo_device)
    else:
        benchmark_traditional(image_paths, out_csv)


if __name__ == "__main__":
    main()
