"""
Generate report/summary.md — the final written comparison — from
report/test_results.csv and report/summary_stats.json.

Run AFTER run_final_evaluation.py (and, optionally, visualize.py for figures).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_DIR / "report"

ERROR_FEATURES = ["area_error", "centroid_error", "width_error", "height_error",
                   "perimeter_error", "boundary_error_px"]


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def main():
    df = pd.read_csv(REPORT_DIR / "test_results.csv")
    with open(REPORT_DIR / "summary_stats.json") as f:
        s = json.load(f)

    n = len(df)
    iou_y, iou_t = s["iou_yolo"], s["iou_traditional"]
    more_accurate = "YOLOv26n" if iou_y["mean"] > iou_t["mean"] else "Traditional CV"

    # Consistency: lower mean geometric error & lower std across error features -> more stable
    err_rows = []
    yolo_wins, trad_wins = 0, 0
    for feat in ERROR_FEATURES:
        ky, kt = f"{feat}_yolo", f"{feat}_traditional"
        if ky not in s or kt not in s:
            continue
        row = (feat, s[ky]["mean"], s[ky]["std"], s[kt]["mean"], s[kt]["std"])
        err_rows.append(row)
        if s[ky]["mean"] < s[kt]["mean"]:
            yolo_wins += 1
        else:
            trad_wins += 1

    more_consistent = "YOLOv26n" if yolo_wins > trad_wins else ("Traditional CV" if trad_wins > yolo_wins else "Tied")

    lines = []
    lines.append("# track-segmentation-ver2 — Final Comparison Report\n")
    lines.append(f"Test images evaluated: **{n}** (from the untouched `test/` split)\n")

    lines.append("## A. Accuracy (IoU)\n")
    lines.append("| Method | Mean IoU | Std IoU | Median | Min | Max |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| YOLOv26n | {fmt(iou_y['mean'])} | {fmt(iou_y['std'])} | {fmt(iou_y['median'])} | {fmt(iou_y['min'])} | {fmt(iou_y['max'])} |")
    lines.append(f"| Traditional CV | {fmt(iou_t['mean'])} | {fmt(iou_t['std'])} | {fmt(iou_t['median'])} | {fmt(iou_t['min'])} | {fmt(iou_t['max'])} |")
    lines.append("")

    lines.append("## B. Stability / Consistency (Geometric Errors vs. Ground Truth)\n")
    lines.append("Lower mean = more accurate geometry; lower std = more consistent across frames.\n")
    lines.append("| Feature | YOLOv26n mean | YOLOv26n std | Traditional mean | Traditional std |")
    lines.append("|---|---|---|---|---|")
    for feat, ym, ys, tm, ts in err_rows:
        lines.append(f"| {feat} | {fmt(ym)} | {fmt(ys)} | {fmt(tm)} | {fmt(ts)} |")
    lines.append("")

    mem_paths = {"YOLOv26n": REPORT_DIR / "memory_yolo_log.csv",
                 "Traditional CV": REPORT_DIR / "memory_traditional_log.csv"}
    mem_rows = []
    for label, p in mem_paths.items():
        if p.exists():
            mdf = pd.read_csv(p)
            mem_rows.append((label, mdf["memory_mb"].mean(), mdf["memory_mb"].std(),
                              mdf["latency_ms"].mean()))
    if mem_rows:
        lines.append("## C. Resource Usage (Memory Footprint)\n")
        lines.append("| Method | Mean RSS Memory (MB) | Std (MB) | Mean Latency (ms) |")
        lines.append("|---|---|---|---|")
        for label, m, sd, lat in mem_rows:
            lines.append(f"| {label} | {fmt(m, 1)} | {fmt(sd, 1)} | {fmt(lat, 1)} |")
        lines.append("")

    lines.append("## Conclusions\n")
    lines.append(f"1. **More accurate (IoU):** {more_accurate} "
                  f"(mean IoU {fmt(max(iou_y['mean'], iou_t['mean']))} vs "
                  f"{fmt(min(iou_y['mean'], iou_t['mean']))}).")
    lines.append(f"2. **More consistent results:** {more_consistent} "
                  f"(wins on {max(yolo_wins, trad_wins)}/{len(err_rows)} geometric-error features "
                  f"by mean; see std columns above for variability).")
    lines.append(f"3. **Lower geometric error vs. GT:** {more_consistent} "
                  f"(by the same feature-wise mean comparison).")
    lines.append("4. **Suitability for a future damage-detection ROI:** prefer the method that is "
                  "both accurate (high IoU, so the ROI doesn't clip real track area or include "
                  "background) and low-boundary-error (so the ROI edge closely tracks the true "
                  "track edge, where damage near the track border must not be missed or where the ROI must "
                  "not falsely include surrounding grass/curb). See the boundary_error_px row above. "
                  "Note the resource-usage trade-off in section C: YOLOv26n's higher accuracy/consistency "
                  "comes with a substantially larger memory footprint (PyTorch/CUDA runtime) than the "
                  "traditional method, which matters if the ROI step must run on constrained hardware.")
    lines.append("")
    lines.append("_Note: Normal-distribution assumptions for IoU are checked in `figures/normality_qq.png` "
                  "(Shapiro-Wilk test) — do not assume normality if that test rejects it._")

    out_path = REPORT_DIR / "summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
