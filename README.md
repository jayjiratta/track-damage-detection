# track-damage-pipeline

Running-track surface damage detection: **YOLOv26 segmentation** restricts the search to the track area
(ROI), **PaDiM** (Patch Distribution Modeling) flags surface regions that differ from normal track surface,
and a simple tracker links detections across video frames to count each damage once.

Demo video: [`results/demo/demo_pipeline.mp4`](results/demo/demo_pipeline.mp4). It covers a normal surface, road markings, a real crack with tracking and counting, and a known failure case.

| Stage | What |
|---|---|
| 1. Track ROI | YOLOv26n-seg (default), or a traditional CV baseline (LAB color + Otsu + field-of-view prior + edge snap) |
| 2. Anomaly map | PaDiM, ResNet-18 features, fitted on normal frames only (no labels) |
| 3. Damage mask | anomaly > calibrated threshold, inside the eroded ROI; small blobs and mostly-painted-marking blobs dropped |
| 4. Count | video: nearby fragments merged, regions linked across frames, counted once after 4 consecutive matches |

## Layout

```
pipeline.py            ROI + PaDiM + post-processing (DamagePipeline)
tracker.py             cross-frame linking + unique-damage count
run_pipeline.py        run on an image, a folder, or a video
train_padim.py         fit PaDiM -> models/padim_resnet18.pt
calibrate.py           damage threshold from held-out normal frames -> models/threshold.json
roi/                   ROI methods (YOLO wrapper, traditional CV, FOV prior derivation)
padim/                 PaDiM implementation
evaluation/            LBP baseline, LBP vs PaDiM evaluation, benchmark, demo video, report figures
roi_experiments/       YOLOv26 vs traditional CV track segmentation (training + test-set evaluation)
models/                YOLOv26n weights, PaDiM split, thresholds
results/evaluation/    figures + metrics used in the report
results/demo/          demo video
```

## Setup

Python 3.10, a CUDA build of PyTorch, then `pip install -r requirements.txt` (YOLO26 needs `ultralytics>=8.4`).

Data is not in this repo:
- PaDiM frames: 450 normal track frames. Default path is `../damage-detection/data`; pass `--data` to override.
- ROI dataset: the Roboflow export `Lane Segmentation ver2.v1i.yolo26`, placed in `roi_experiments/`.
- Demo recording: the full-loop capture video (`capture_20260806_121427.mp4`).

`models/padim_resnet18.pt` (~290 MB) is over GitHub's file size limit, so it isn't committed. Regenerate it with `python train_padim.py` (a few seconds on a GPU).

## Usage

```
python train_padim.py                       # once
python calibrate.py                         # once
python run_pipeline.py <image|folder>       [--roi yolo|traditional]
python run_pipeline.py <video> --start 21760 --stride 2 --max-frames 150
```

Reproduce the report figures and numbers:

```
python evaluation/figure_lbp_vs_padim_real.py <capture video>   # LBP vs PaDiM on real frames
python evaluation/make_demo_video.py <capture video>            # demo video
python evaluation/benchmark_pipeline.py                         # latency / memory
python evaluation/evaluate.py                                   # synthetic-defect sanity check (see below)
cd roi_experiments
python train/train_yolo26n_seg.py                               # optional: retrain YOLOv26n
python evaluation/run_final_evaluation.py                       # YOLOv26n vs traditional on the test split
python evaluation/memory_benchmark.py && python evaluation/visualize.py && python evaluation/generate_summary.py
```

## Results

### Track ROI: YOLOv26n vs traditional CV (45 test images, never used for training or tuning)

| | IoU | Boundary error (px) | Area error | Memory (MB) | Latency (ms) |
|---|---|---|---|---|---|
| YOLOv26n-seg | **0.989 ± 0.005** | **4.3 ± 3.3** | **0.004 ± 0.001** | 1294 | 13.9 |
| Traditional CV | 0.909 ± 0.045 | 33.3 ± 16.1 | 0.049 ± 0.028 | **52** | 48.8 |

Boundary error is the average symmetric distance between predicted and ground-truth track boundaries, with the image frame edge excluded. Details are in `roi_experiments/report/summary.md`.

### Damage: LBP (earlier demo) vs PaDiM

There is no expert-labeled damage ground truth yet, so the comparison is qualitative
(`results/evaluation/qualitative_lbp_vs_padim_real.jpg`). On real frames, LBP flags lane lines, road markings and
surface texture, while PaDiM with the YOLOv26 ROI stays quiet on normal surface and follows a real crack.

`evaluation/evaluate.py` also runs a synthetic-defect sanity check (dark cracks painted onto held-out frames). It
confirms the mechanism works, but it is not a measure of real-world accuracy.

Full pipeline on an RTX 3050 laptop GPU: 56.8 ± 6.1 ms per frame (17.6 FPS), 1.83 GB process RSS.

## Known limitations

- **No damage ground truth.** Quantitative damage metrics (precision/recall, counting accuracy) need expert-labeled masks. The tracker's counts have not been checked against labeled damage identities.
- **Painted markings** (chevrons, distance text) are rare and appear at different image positions, so PaDiM flags them. The paint filter removes most of these, but it also drops real damage lying mostly on paint.
- **Sections not in the training frames.** In the shaded tree-lined section, PaDiM flags shadows and debris (see the last demo segment). PaDiM's training frames need to cover the whole route and its lighting.
- **Training frames were not screened for damage.** PaDiM treats all fitted frames as normal.
- **Same camera framing assumed.** PaDiM's per-position model and the traditional method's FOV prior both assume the training camera mount.
- Sharing YOLOv26 backbone features with PaDiM and Jetson Orin Nano deployment are not implemented yet.
