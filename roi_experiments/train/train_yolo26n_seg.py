"""
Train YOLOv26n (nano) segmentation for WHOLE running-track area segmentation
=============================================================================
Dataset : "Lane Segmentation ver2.v1i.yolo26" Roboflow export (1 class),
          placed in roi_experiments/ (not committed to git)
Split rule (STRICTLY ENFORCED):
    - train/  -> training only
    - valid/  -> validation only (used by ultralytics during training for
                 early stopping / best-checkpoint selection)
    - test/   -> NEVER referenced here. It is only read by
                 evaluation/run_final_evaluation.py, after training/model
                 selection is complete.

Model: YOLO26n-seg (C3k2/C2PSA backbone, Segment26 head), fine-tuned from
the COCO-pretrained yolo26n-seg.pt (auto-downloaded). Requires ultralytics >= 8.4.
Best weights: runs/yolo26n_track_seg/weights/best.pt -> copy to
../models/yolo26n_track_seg.pt for the pipeline.

Usage (from roi_experiments/):
    python train/train_yolo26n_seg.py
"""
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "Lane Segmentation ver2.v1i.yolo26"
DATA_YAML   = PROJECT_DIR / "runs" / "data_train_valid.yaml"
RUNS_DIR    = PROJECT_DIR / "runs"

CFG = dict(
    model     = "yolo26n-seg.pt",
    name      = "yolo26n_track_seg",
    epochs    = 150,
    imgsz     = 640,
    batch     = 16,
    patience  = 30,
)


def main():
    from ultralytics import YOLO

    if not (DATASET_DIR / "train" / "images").exists():
        raise FileNotFoundError(f"dataset not found: {DATASET_DIR}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # train + valid only; the test split is deliberately not referenced here
    DATA_YAML.write_text(f"train: {(DATASET_DIR / 'train' / 'images').as_posix()}\n"
                         f"val: {(DATASET_DIR / 'valid' / 'images').as_posix()}\n"
                         "nc: 1\nnames: ['track']\n")

    print("=" * 60)
    print(f"Training YOLOv26n whole-track segmentation")
    print(f"model={CFG['model']}  epochs={CFG['epochs']}  imgsz={CFG['imgsz']}  batch={CFG['batch']}")
    print(f"data={DATA_YAML}")
    print("=" * 60)

    model = YOLO(CFG["model"])

    t0 = time.time()
    results = model.train(
        data      = str(DATA_YAML),
        epochs    = CFG["epochs"],
        imgsz     = CFG["imgsz"],
        batch     = CFG["batch"],
        patience  = CFG["patience"],
        name      = CFG["name"],
        project   = str(RUNS_DIR),
        device    = 0,
        workers   = 4,
        exist_ok  = True,
        plots     = True,
        save      = True,
        val       = True,
        amp       = True,
        # Whole-track masks are large & roughly symmetric in perspective view;
        # a horizontal flip is geometrically valid here (unlike per-lane
        # left/right classes in the original project), so fliplr is allowed.
        degrees   = 5.0,
        translate = 0.05,
        scale     = 0.3,
        fliplr    = 0.5,
        mosaic    = 0.5,
    )

    elapsed = time.time() - t0
    best_pt = RUNS_DIR / CFG["name"] / "weights" / "best.pt"

    print(f"\nTraining done in {elapsed/60:.1f} min")
    print(f"Best weights: {best_pt}")
    print(f"seg_map50   : {results.results_dict.get('metrics/mAP50(M)', 'N/A')}")
    print(f"seg_map50-95: {results.results_dict.get('metrics/mAP50-95(M)', 'N/A')}")


if __name__ == "__main__":
    main()
