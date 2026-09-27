"""
Per-frame latency breakdown (track ROI / PaDiM / post-processing) and process
memory (RSS) of the full pipeline, on PaDiM's held-out frames.

Usage (from track-damage-pipeline/):
    python evaluation/benchmark_pipeline.py [--roi yolo|traditional]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import DamagePipeline, erode_roi

DATA_DIR = ROOT.parent / "damage-detection" / "data"
WARMUP = 5


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", default="yolo", choices=["yolo", "traditional"])
    args = ap.parse_args()

    names = json.loads((ROOT / "models" / "padim_split.json").read_text())["holdout"]
    imgs = [cv2.imread(str(DATA_DIR / n)) for n in names]
    pipe = DamagePipeline(args.roi)
    t_roi, t_padim, t_total = [], [], []
    for i, img in enumerate(imgs):
        sync(); t0 = time.perf_counter()
        erode_roi(pipe.roi(img)); sync(); t1 = time.perf_counter()
        pipe.padim.score(img); sync(); t2 = time.perf_counter()
        pipe(img); sync(); t3 = time.perf_counter()
        if i >= WARMUP:
            t_roi.append((t1 - t0) * 1e3)
            t_padim.append((t2 - t1) * 1e3)
            t_total.append((t3 - t2) * 1e3)
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
    gpu = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    res = {"roi_method": args.roi, "n_frames": len(t_total),
           "roi_ms_mean": float(np.mean(t_roi)), "padim_ms_mean": float(np.mean(t_padim)),
           "total_ms_mean": float(np.mean(t_total)), "total_ms_std": float(np.std(t_total)),
           "fps": 1000.0 / float(np.mean(t_total)), "rss_mb": rss, "torch_gpu_peak_mb": gpu,
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    out = ROOT / "results" / "evaluation" / f"benchmark_{args.roi}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
