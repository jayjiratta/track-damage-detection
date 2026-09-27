"""
Fit PaDiM on normal track frames -> models/padim_resnet18.pt (+ padim_split.json).

The fitted model (~290 MB of per-patch Gaussians) is over GitHub's file size
limit, so it is not committed; fitting takes a few seconds on a GPU.

Split: every 10th frame (sorted by name = frame number, so spread evenly
across the video) is held out and never fitted. calibrate.py and
evaluation/evaluate.py use the held-out frames.

Usage:
    python train_padim.py [--data ../damage-detection/data]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "padim"))
from padim import PaDiM

HOLDOUT_EVERY = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT.parent / "damage-detection" / "data")
    args = ap.parse_args()

    images = sorted(p for p in args.data.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        raise FileNotFoundError(f"no images in {args.data}")
    holdout = images[::HOLDOUT_EVERY]
    held = set(holdout)
    fit = [p for p in images if p not in held]
    print(f"images: {len(images)}  fit: {len(fit)}  held-out: {len(holdout)}")

    model = PaDiM()
    t0 = time.time()
    model.fit(fit)
    print(f"fit done in {time.time() - t0:.1f}s  (patch grid {model.grid_hw}, n={model.n_train})")
    model.save(ROOT / "models" / "padim_resnet18.pt")
    (ROOT / "models" / "padim_split.json").write_text(json.dumps(
        {"fit": [p.name for p in fit], "holdout": [p.name for p in holdout]}, indent=1))
    print("saved: models/padim_resnet18.pt, models/padim_split.json")


if __name__ == "__main__":
    main()
