"""
Train the method-2 track model: YOLO26-seg neck + Segment26 head on a FROZEN
ImageNet ResNet-18 backbone (train/yolo26-seg-resnet18.yaml).

Same dataset, split rule (test/ never referenced), image size and
augmentation as train_yolo26n_seg.py; only the model differs. freeze=[0]
freezes the whole TorchVision backbone module (ultralytics also keeps its
BatchNorm layers in eval mode), so the backbone stays exactly ImageNet
ResNet-18 and its layer1-3 can be shared with PaDiM (padim/padim.py,
backbone="shared_r18").
Best weights: runs/yolo26_resnet18_frozen_seg/weights/best.pt -> copy to
../models/yolo26_r18frozen_track_seg.pt for the pipeline.

Usage (from roi_experiments/):
    python train/train_yolo26_resnet18_frozen_seg.py
"""
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "Lane Segmentation ver2.v1i.yolo26"
DATA_YAML   = PROJECT_DIR / "runs" / "data_train_valid.yaml"
RUNS_DIR    = PROJECT_DIR / "runs"
MODEL_YAML  = PROJECT_DIR / "train" / "yolo26-seg-resnet18.yaml"

CFG = dict(name="yolo26_resnet18_frozen_seg", epochs=150, imgsz=640, batch=16, patience=30)


def main():
    from ultralytics import YOLO

    if not (DATASET_DIR / "train" / "images").exists():
        raise FileNotFoundError(f"dataset not found: {DATASET_DIR}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # train + valid only; the test split is deliberately not referenced here
    DATA_YAML.write_text(f"train: {(DATASET_DIR / 'train' / 'images').as_posix()}\n"
                         f"val: {(DATASET_DIR / 'valid' / 'images').as_posix()}\n"
                         "nc: 1\nnames: ['track']\n")

    model = YOLO(str(MODEL_YAML))
    t0 = time.time()
    results = model.train(
        data=str(DATA_YAML), epochs=CFG["epochs"], imgsz=CFG["imgsz"], batch=CFG["batch"],
        patience=CFG["patience"], name=CFG["name"], project=str(RUNS_DIR), device=0, workers=4,
        exist_ok=True, plots=True, save=True, val=True, amp=True,
        freeze=[0],
        degrees=5.0, translate=0.05, scale=0.3, fliplr=0.5, mosaic=0.5,
    )
    print(f"\nTraining done in {(time.time() - t0) / 60:.1f} min")
    print(f"Best weights: {RUNS_DIR / CFG['name'] / 'weights' / 'best.pt'}")
    print(f"seg_map50   : {results.results_dict.get('metrics/mAP50(M)', 'N/A')}")
    print(f"seg_map50-95: {results.results_dict.get('metrics/mAP50-95(M)', 'N/A')}")


if __name__ == "__main__":
    main()
