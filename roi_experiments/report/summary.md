# track-segmentation-ver2 — Final Comparison Report

Test images evaluated: **45** (from the untouched `test/` split)

## A. Accuracy (IoU)

| Method | Mean IoU | Std IoU | Median | Min | Max |
|---|---|---|---|---|---|
| YOLOv26n | 0.9895 | 0.0046 | 0.9908 | 0.9654 | 0.9934 |
| Traditional CV | 0.9094 | 0.0448 | 0.9101 | 0.8293 | 0.9797 |

## B. Stability / Consistency (Geometric Errors vs. Ground Truth)

Lower mean = more accurate geometry; lower std = more consistent across frames.

| Feature | YOLOv26n mean | YOLOv26n std | Traditional mean | Traditional std |
|---|---|---|---|---|
| area_error | 0.0044 | 0.0014 | 0.0487 | 0.0275 |
| centroid_error | 0.0011 | 0.0025 | 0.0263 | 0.0189 |
| width_error | 0.0034 | 0.0113 | 0.0000 | 0.0000 |
| height_error | 0.0012 | 0.0011 | 0.0177 | 0.0159 |
| perimeter_error | 0.0445 | 0.0335 | 0.3101 | 0.2275 |
| boundary_error_px | 4.2711 | 3.2849 | 33.3397 | 16.1331 |

## C. Resource Usage (Memory Footprint)

| Method | Mean RSS Memory (MB) | Std (MB) | Mean Latency (ms) |
|---|---|---|---|
| YOLOv26n | 1294.2 | 0.0 | 13.9 |
| Traditional CV | 52.4 | 0.3 | 48.8 |

## Conclusions

1. **More accurate (IoU):** YOLOv26n (mean IoU 0.9895 vs 0.9094).
2. **More consistent results:** YOLOv26n (wins on 5/6 geometric-error features by mean; see std columns above for variability).
3. **Lower geometric error vs. GT:** YOLOv26n (by the same feature-wise mean comparison).
4. **Suitability for a future damage-detection ROI:** prefer the method that is both accurate (high IoU, so the ROI doesn't clip real track area or include background) and low-boundary-error (so the ROI edge closely tracks the true track edge, where damage near the track border must not be missed or where the ROI must not falsely include surrounding grass/curb). See the boundary_error_px row above. Note the resource-usage trade-off in section C: YOLOv26n's higher accuracy/consistency comes with a substantially larger memory footprint (PyTorch/CUDA runtime) than the traditional method, which matters if the ROI step must run on constrained hardware.

_Note: Normal-distribution assumptions for IoU are checked in `figures/normality_qq.png` (Shapiro-Wilk test) — do not assume normality if that test rejects it._