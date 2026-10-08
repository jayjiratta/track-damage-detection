# track-damage-detection

Running-track surface damage detection: **YOLOv26 segmentation** restricts the search to the track area
(ROI), **PaDiM** (Patch Distribution Modeling) flags surface regions that differ from normal track surface,
and a simple tracker links detections across video frames to count each damage once.

Demo video (method 2): [`results/demo/demo_pipeline.mp4`](results/demo/demo_pipeline.mp4). It has four segments:
a normal surface, a lane line on a curve, a real crack with tracking and a unique count, and road markings.

| Stage | What |
|---|---|
| 1. Track ROI | YOLOv26 segmentation (or a traditional CV baseline: LAB a* + Otsu + field-of-view prior + edge snap) |
| 2. Anomaly map | PaDiM on ResNet-18 layer1-3 features, fitted on normal frames only (no labels) |
| 3. Damage mask | anomaly > calibrated threshold inside the eroded ROI; components < 100 px dropped |
| 4. Count | regions merged into boxes, linked across frames, counted once after 4 consecutive matches |

## Two methods

| | Method 1: separate backbones | Method 2: shared backbone |
|---|---|---|
| Track ROI | YOLOv26n-seg (`models/yolo26n_track_seg.pt`) | YOLO26 neck + Segment26 head on a **frozen** ImageNet ResNet-18 (`models/yolo26_r18frozen_track_seg.pt`) |
| PaDiM features | a second ResNet-18, image resized to 448x256, 64x112 grid | layer1-3 of the same frozen ResNet-18, taken with forward hooks during the ROI pass, 48x80 grid |
| CNN passes per frame | 2 | 1 |
| Code | `DamagePipeline("yolo", "resnet18")` | `DamagePipeline("yolo_r18", "shared_r18")` |

Because the shared ResNet-18 is never fine-tuned, its features stay generic, which is what PaDiM needs to see
surface damage. Only the YOLO neck/head (2.35M of 13.5M parameters) is trained on the track masks
(`roi_experiments/train/yolo26-seg-resnet18.yaml`, `freeze=[0]`).

## Layout

```
pipeline.py            ROI + PaDiM + post-processing (DamagePipeline, METHODS)
tracker.py             cross-frame linking + unique-damage count
run_pipeline.py        run on an image, a folder, or a video (--method 1|2)
extract_frames.py      normal frames for PaDiM from the capture video -> data/normal_frames/
train_padim.py         fit PaDiM -> models/padim_<backbone>.pt
calibrate.py           damage threshold from held-out normal frames -> models/threshold.json
roi/                   ROI methods (YOLO wrappers, traditional CV, FOV prior derivation)
padim/                 PaDiM implementation (resnet18 and shared_r18 feature sources)
evaluation/            benchmark, demo video, report figure
roi_experiments/       track segmentation: training (both track models) + test-set evaluation
models/                track-model weights, PaDiM split, thresholds
results/evaluation/    benchmarks + figure used in the report
results/demo/          demo video + per-segment unique counts
```

## Setup

Python 3.10, a CUDA build of PyTorch, then `pip install -r requirements.txt` (YOLO26 needs `ultralytics>=8.4`).

Data is not in this repo:
- Capture video: the full-loop recording `capture_20260806_121427.mp4` (1280x720, 30 fps, 21.6 min).
- PaDiM frames: `python extract_frames.py <video>` writes every 20th frame, minus the frames where the real
  crack is visible (21780-22000), to `data/normal_frames/` (1,932 frames; every 10th is held out, 1,738 are fitted).
- ROI dataset: the Roboflow export `Lane Segmentation ver2.v1i.yolo26` (450 images, 315/90/45 split), placed in `roi_experiments/`.

The fitted PaDiM models (`models/padim_resnet18.pt` ~290 MB, `models/padim_shared_r18.pt` ~155 MB) are over
GitHub's file size limit, so they are not committed. Each takes about a minute to fit on a GPU.

## Usage

```
python extract_frames.py <video>                       # once
python train_padim.py --backbone resnet18              # method 1
python train_padim.py --backbone shared_r18            # method 2
python calibrate.py                                    # thresholds for both methods
python run_pipeline.py <image|folder>                  [--method 1|2]   (default 2)
python run_pipeline.py <video> --start 21760 --stride 2 --max-frames 150
```

Reproduce the report numbers:

```
python evaluation/benchmark_pipeline.py --backbone resnet18     # method 1: time / memory
python evaluation/benchmark_pipeline.py --backbone shared_r18   # method 2
python evaluation/make_demo_video.py <video>                    # demo video + counts_method2.csv
python evaluation/make_demo_video.py <video> --method 1 --counts-only
python evaluation/figure_traditional_steps.py <image>
cd roi_experiments
python train/train_yolo26n_seg.py                               # optional: retrain method 1 track model
python train/train_yolo26_resnet18_frozen_seg.py                # optional: retrain method 2 track model
python evaluation/run_final_evaluation.py                       # YOLOv26n vs traditional on the test split
```

## Results (RTX 3050 laptop GPU)

### Track ROI: YOLOv26n vs traditional CV (45 test images)

| | IoU | Min IoU | Time per image (ms) | Memory (MB) |
|---|---|---|---|---|
| YOLOv26n-seg | **0.989 ± 0.005** | **0.965** | 13.9 (GPU) | 1294 |
| Traditional CV | 0.909 ± 0.045 | 0.829 | 48.8 (CPU) | **52** |

### Method 1 vs method 2

| | Method 1 | Method 2 |
|---|---|---|
| Track IoU (test set) | 0.989 ± 0.005 | 0.985 ± 0.003 |
| Time per frame (ms) | 56.9 ± 8.1 | **47.2 ± 7.0** |
| FPS | 17.6 | **21.2** |
| Peak GPU memory (MB) | 418 | **313** |
| Process memory (MB) | 2,254 | **1,897** |
| PaDiM model file (MB) | 290 | **155** |
| Crack-line pixels flagged* | 35.4% | 36.7% |
| Normal-track pixels flagged | 0.42% | 0.33% |

\* Proxy on the real-crack frames; there is no expert damage ground truth yet, so this only compares the two methods.
Benchmarks: `results/evaluation/benchmark_*.json`.

### Unique damage count (demo segments, every 2nd frame)

| Segment (frames) | Method 1 | Method 2 |
|---|---|---|
| Normal surface (1900-2138) | 1 | 1 |
| Lane line on a curve (22350-22588) | 0 | 2 |
| Real crack (21760-22058) | 15 | 8 |
| Road markings (4100-4338) | 12 | 13 |

From `results/demo/counts_method{1,2}.csv`. The counts are inflated by false alarms (next section).

## Known limitations

- **No damage ground truth.** Precision/recall and counting accuracy need expert-labeled damage masks and counts.
- **Patterns rare at an image position.** PaDiM models each patch position separately, so a normal pattern that
  appears at a position in only a few training frames (arrows, distance numbers) scores as anomalous. Lane lines
  that appear at a position in 20-50% of the frames are flagged ~1% of the time; positions where paint appears in
  <1% of the frames are flagged ~29%. The fix is more normal frames of those markings, not a rule.
- **Training frames were not screened for damage** apart from the known crack segment; PaDiM treats them as normal.
- **Same camera framing assumed.** PaDiM's per-position model and the traditional method's FOV prior both assume the training camera mount.
- Jetson Orin Nano deployment (ONNX + TensorRT FP16) is not implemented yet.
