"""
Fit PaDiM on normal track frames -> models/padim_<backbone>.pt (+ padim_split.json).

  --backbone resnet18   method 1: separate ResNet-18
  --backbone shared_r18 method 2: shared frozen ResNet-18 backbone of the yolo_r18 track model

The fitted models (per-patch Gaussians, 150-300 MB) are over GitHub's file
size limit, so they are not committed; fitting takes about a minute on a GPU.

Split: every 10th frame (sorted by name = frame number, so spread evenly
along the route) is held out and never fitted. calibrate.py and the
evaluation scripts use the held-out frames. Both backbones use the same split.

Usage:
    python extract_frames.py <video>          # once: data/normal_frames
    python train_padim.py --backbone resnet18
    python train_padim.py --backbone shared_r18
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "padim"))
sys.path.insert(0, str(ROOT / "roi"))
from padim import PaDiM

HOLDOUT_EVERY = 10
DEFAULT_DATA = ROOT / "data" / "normal_frames"


def split_frames(data: Path):
    images = sorted(p for p in data.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        raise FileNotFoundError(f"no images in {data} -- run extract_frames.py first")
    holdout = images[::HOLDOUT_EVERY]
    held = set(holdout)
    return [p for p in images if p not in held], holdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "shared_r18"])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--grid-stride", type=int, default=None,
                    help="patch grid stride for shared_r18 (default 8 = 48x80 grid)")
    ap.add_argument("--out", type=Path, default=None, help="model file (default models/padim_<backbone>.pt)")
    args = ap.parse_args()

    fit, holdout = split_frames(args.data)
    print(f"images: {len(fit) + len(holdout)}  fit: {len(fit)}  held-out: {len(holdout)}")

    if args.backbone == "shared_r18":
        from roi import TrackROI
        model = PaDiM(args.backbone, yolo=TrackROI("yolo_r18").model, grid_stride=args.grid_stride)
    else:
        model = PaDiM("resnet18")
    t0 = time.time()
    model.fit(fit)
    print(f"fit done in {time.time() - t0:.1f}s  (patch grid {model.grid_hw}, n={model.n_train})")
    out = (args.out or ROOT / "models" / f"padim_{args.backbone}.pt").resolve()
    model.save(out)
    (ROOT / "models" / "padim_split.json").write_text(json.dumps(
        {"data": str(args.data.relative_to(ROOT)) if args.data.is_relative_to(ROOT) else str(args.data),
         "fit": [p.name for p in fit], "holdout": [p.name for p in holdout]}, indent=1))
    print(f"saved: {out.relative_to(ROOT)}, models/padim_split.json")


if __name__ == "__main__":
    main()
