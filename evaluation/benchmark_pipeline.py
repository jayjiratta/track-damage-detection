"""
Per-frame inference time (track ROI, PaDiM, whole pipeline) and memory of
one system, on PaDiM's held-out frames. Run each system in its own process
so the memory figures don't mix:

    python evaluation/benchmark_pipeline.py --backbone resnet18     # method 1
    python evaluation/benchmark_pipeline.py --backbone shared_r18   # method 2

For method 2 the "PaDiM" time excludes the CNN forward pass, which is
shared with (and counted in) the ROI step.
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

WARMUP = 5


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def pipe_file_mb(backbone):
    return (ROOT / "models" / f"padim_{backbone}.pt").stat().st_size / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", default="yolo", choices=["yolo", "yolo_r18", "traditional"])
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "shared_r18"])
    args = ap.parse_args()
    if args.backbone == "shared_r18":
        args.roi = "yolo_r18"

    split = json.loads((ROOT / "models" / "padim_split.json").read_text())
    imgs = [cv2.imread(str(ROOT / split["data"] / n)) for n in split["holdout"]]
    rss0 = psutil.Process(os.getpid()).memory_info().rss / 1e6
    pipe = DamagePipeline(args.roi, args.backbone)
    t_roi, t_padim, t_total = [], [], []
    for i, img in enumerate(imgs):
        sync(); t0 = time.perf_counter()
        erode_roi(pipe.roi(img)); sync(); t1 = time.perf_counter()
        if args.backbone == "shared_r18":
            pipe.padim.score(img, feats=pipe.padim.hook.get())
        else:
            pipe.padim.score(img)
        sync(); t2 = time.perf_counter()
        pipe(img); sync(); t3 = time.perf_counter()
        if i >= WARMUP:
            t_roi.append((t1 - t0) * 1e3)
            t_padim.append((t2 - t1) * 1e3)
            t_total.append((t3 - t2) * 1e3)
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
    gpu = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    res = {"roi_method": args.roi, "backbone": args.backbone, "n_frames": len(t_total),
           "roi_ms_mean": float(np.mean(t_roi)), "padim_ms_mean": float(np.mean(t_padim)),
           "total_ms_mean": float(np.mean(t_total)), "total_ms_std": float(np.std(t_total)),
           "fps": 1000.0 / float(np.mean(t_total)), "rss_mb": rss, "rss_models_mb": rss - rss0,
           "torch_gpu_peak_mb": gpu, "padim_model_file_mb": pipe_file_mb(args.backbone),
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    out = ROOT / "results" / "evaluation" / f"benchmark_{args.roi}_{args.backbone}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
