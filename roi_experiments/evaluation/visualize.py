"""
Visualization of the YOLOv26n vs Traditional-CV whole-track segmentation
comparison, built from evaluation/run_final_evaluation.py's outputs.

Run AFTER run_final_evaluation.py.

Outputs (into ../report/figures/):
  - iou_boxplot.png             IoU distribution comparison (box+strip)
  - iou_dist.png                 IoU histogram + fitted Normal curve, both methods
  - dist_<feature>.png           one histogram + fitted Normal curve per geometric
                                  error feature (area/centroid/width/height/
                                  perimeter/boundary), both methods overlaid --
                                  same style as track-segmentation/benchmark's
                                  a_dist.png / b_dist.png / c_dist.png
  - all_distributions.png        grid of all the dist_<feature> + iou_dist plots
  - mean_std_comparison.png      bar chart: mean +/- std per feature, per method
  - sample_overlays.png           GT vs YOLO vs Traditional mask overlays, a few samples
  - normality_qq.png             Q-Q plots for IoU (checks the Normal-distribution assumption)
  - memory_vs_time.png           RSS memory footprint per test image, both methods
                                  (same style as track-segmentation/benchmark's
                                  memory_vs_time.png) -- run
                                  evaluation/memory_benchmark.py first
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_DIR / "report"
FIG_DIR = REPORT_DIR / "figures"
MASKS_DIR = REPORT_DIR / "masks"
DATASET_IMG_DIR = PROJECT_DIR / "Lane Segmentation ver2.v1i.yolo26" / "test" / "images"

METHOD_COLORS = {"yolo": "#2E86AB", "traditional": "#C1440E"}
METHOD_LABELS = {"yolo": "YOLOv26n", "traditional": "Traditional CV"}

ERROR_FEATURES = ["area_error", "centroid_error", "width_error", "height_error",
                   "perimeter_error", "boundary_error_px"]


def load_data():
    df = pd.read_csv(REPORT_DIR / "test_results.csv")
    with open(REPORT_DIR / "summary_stats.json") as f:
        summary = json.load(f)
    return df, summary


def plot_iou_boxplot(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    data = [df["iou_yolo"].dropna(), df["iou_traditional"].dropna()]
    bp = ax.boxplot(data, tick_labels=[METHOD_LABELS["yolo"], METHOD_LABELS["traditional"]],
                     patch_artist=True, widths=0.5)
    for patch, key in zip(bp["boxes"], ["yolo", "traditional"]):
        patch.set_facecolor(METHOD_COLORS[key])
        patch.set_alpha(0.6)
    for i, key in enumerate(["yolo", "traditional"]):
        y = df[f"iou_{key}"].dropna().values
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, color=METHOD_COLORS[key], alpha=0.6, s=18, zorder=3)
    ax.set_ylabel("IoU")
    ax.set_title("Test-set IoU: YOLOv26n vs Traditional CV")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "iou_boxplot.png", dpi=150)
    plt.close(fig)


DIST_CONFIGS = [
    ("iou_{}", "iou_dist.png", "IoU Normal Distribution", "IoU"),
    ("area_error_{}", "dist_area_error.png", "Area Error Normal Distribution", "Area Error (|pred - GT| ratio)"),
    ("centroid_error_{}", "dist_centroid_error.png", "Centroid Error Normal Distribution", "Centroid Error (normalized distance)"),
    ("width_error_{}", "dist_width_error.png", "Width Error Normal Distribution", "Width Error (ratio)"),
    ("height_error_{}", "dist_height_error.png", "Height Error Normal Distribution", "Height Error (ratio)"),
    ("perimeter_error_{}", "dist_perimeter_error.png", "Perimeter Error Normal Distribution", "Perimeter Error (normalized)"),
    ("boundary_error_px_{}", "dist_boundary_error.png", "Boundary Error Normal Distribution", "Boundary Error (px)"),
]


def plot_all_normal_distributions(df):
    """One saved PNG per metric (IoU + each geometric error feature), each in
    the hist+fitted-Normal-curve style of a_dist.png/b_dist.png/c_dist.png,
    plus one combined grid figure."""
    n = len(DIST_CONFIGS)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig_all, axes_all = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
    axes_flat = axes_all.flat

    for (col_tpl, filename, title, xlabel), ax_combined in zip(DIST_CONFIGS, axes_flat):
        base_col = col_tpl[:-3]  # strip trailing "_{}" -> "<feature>" (joined with "_yolo"/"_traditional")
        fmt = "{:.2f}" if xlabel.endswith("(px)") else "{:.4f}"

        # individual figure
        fig, ax = plt.subplots(figsize=(9, 5))
        _draw_normal_dist_by_base(ax, df, base_col, title, xlabel, fmt)
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=150)
        plt.close(fig)

        # combined grid panel
        _draw_normal_dist_by_base(ax_combined, df, base_col, title, xlabel, fmt)

    for ax in list(axes_flat)[n:]:
        ax.axis("off")

    fig_all.suptitle("Normal-Distribution Comparison: YOLOv26n vs Traditional CV",
                      fontsize=14, fontweight="bold", y=1.01)
    fig_all.tight_layout()
    fig_all.savefig(FIG_DIR / "all_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig_all)


def _draw_normal_dist_by_base(ax, df, base_col: str, title: str, xlabel: str, fmt: str):
    any_plotted = False
    for key in ["yolo", "traditional"]:
        col = f"{base_col}_{key}"
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        if len(vals) == 0:
            continue
        any_plotted = True
        mean_val, std_val = np.mean(vals), np.std(vals)
        if std_val == 0:
            std_val = 1e-6
        style_color = METHOD_COLORS[key]
        ax.hist(vals, bins=min(20, max(5, len(vals) // 3)), density=True,
                alpha=0.3, color=style_color)
        x = np.linspace(mean_val - 4 * std_val, mean_val + 4 * std_val, 200)
        p = np.exp(-0.5 * ((x - mean_val) / std_val) ** 2) / (std_val * np.sqrt(2 * np.pi))
        ax.plot(x, p, color=style_color, linewidth=2.5,
                label=f"{METHOD_LABELS[key]}  (μ={fmt.format(mean_val)}, σ={fmt.format(std_val)})")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    if any_plotted:
        ax.legend(fontsize=8)


def plot_mean_std_comparison(summary):
    metrics = ["iou"] + ERROR_FEATURES
    labels_present = [m for m in metrics if f"{m}_yolo" in summary or m == "iou"]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels_present))
    width = 0.35

    def get(metric, method):
        key = metric if metric == "iou" else f"{metric}_{method}"
        full_key = f"{key}_{method}" if metric == "iou" else key
        return summary.get(full_key, {"mean": np.nan, "std": np.nan})

    for i, method in enumerate(["yolo", "traditional"]):
        means, stds = [], []
        for m in labels_present:
            k = f"iou_{method}" if m == "iou" else f"{m}_{method}"
            s = summary.get(k, {"mean": np.nan, "std": np.nan})
            means.append(s["mean"])
            stds.append(s["std"])
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               color=METHOD_COLORS[method], alpha=0.75, label=METHOD_LABELS[method])

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in labels_present], rotation=30, ha="right")
    ax.set_ylabel("Value (mean +/- std)")
    ax.set_title("Mean +/- Std Comparison: Accuracy (IoU) and Geometric Errors")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mean_std_comparison.png", dpi=150)
    plt.close(fig)


def plot_normality_qq(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key in zip(axes, ["yolo", "traditional"]):
        vals = df[f"iou_{key}"].dropna().values
        stats.probplot(vals, dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot: IoU ({METHOD_LABELS[key]})")
        # Shapiro-Wilk normality test annotation
        if len(vals) >= 3:
            w, p = stats.shapiro(vals)
            ax.text(0.05, 0.95, f"Shapiro-Wilk p={p:.3f}\n({'looks normal' if p > 0.05 else 'NOT normal'})",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "normality_qq.png", dpi=150)
    plt.close(fig)


def plot_sample_overlays(df, n_samples=6):
    rng = np.random.default_rng(0)
    sample_stems = rng.choice(df["image"].values, size=min(n_samples, len(df)), replace=False)

    fig, axes = plt.subplots(len(sample_stems), 4, figsize=(16, 4 * len(sample_stems)))
    if len(sample_stems) == 1:
        axes = axes[None, :]

    for row_i, stem in enumerate(sample_stems):
        img_path = DATASET_IMG_DIR / f"{stem}.jpg"
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(MASKS_DIR / "gt" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        yolo = cv2.imread(str(MASKS_DIR / "yolo" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        trad = cv2.imread(str(MASKS_DIR / "traditional" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)

        def overlay(base, mask, color):
            out = base.copy()
            m = mask > 0
            out[m] = (0.5 * out[m] + 0.5 * np.array(color)).astype(np.uint8)
            return out

        axes[row_i, 0].imshow(img)
        axes[row_i, 0].set_title(f"{stem}\nOriginal" if row_i == 0 else stem)
        axes[row_i, 1].imshow(overlay(img, gt, [0, 255, 0]))
        axes[row_i, 1].set_title("Ground Truth" if row_i == 0 else "")
        axes[row_i, 2].imshow(overlay(img, yolo, [255, 0, 0]))
        axes[row_i, 2].set_title("YOLOv26n" if row_i == 0 else "")
        axes[row_i, 3].imshow(overlay(img, trad, [0, 128, 255]))
        axes[row_i, 3].set_title("Traditional CV" if row_i == 0 else "")
        for ax in axes[row_i]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "sample_overlays.png", dpi=130)
    plt.close(fig)


def plot_memory_footprint():
    """
    RSS memory footprint per test image, both methods -- same style as
    track-segmentation/benchmark/plots/memory_vs_time.png (plain line plot,
    one color per method, memory in MB). x-axis is image index within the
    test set (no video timestamps here, unlike the original project).
    Skips silently if evaluation/memory_benchmark.py hasn't been run yet.
    """
    paths = {"yolo": REPORT_DIR / "memory_yolo_log.csv",
             "traditional": REPORT_DIR / "memory_traditional_log.csv"}
    dfs = {k: pd.read_csv(p) for k, p in paths.items() if p.exists()}
    if not dfs:
        print("  [SKIP] memory_vs_time.png -- run evaluation/memory_benchmark.py first")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for key, mdf in dfs.items():
        ax.plot(mdf["frame_idx"], mdf["memory_mb"], color=METHOD_COLORS[key],
                linewidth=1.5, alpha=0.85, label=METHOD_LABELS[key])
    ax.set_title("Memory Footprint vs Test Image Index", fontweight="bold")
    ax.set_xlabel("Test Image Index")
    ax.set_ylabel("Memory (MB)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "memory_vs_time.png", dpi=150)
    plt.close(fig)

    # Mean footprint bar chart for a quick at-a-glance comparison.
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    means = [dfs[k]["memory_mb"].mean() for k in dfs]
    stds = [dfs[k]["memory_mb"].std() for k in dfs]
    labels = [METHOD_LABELS[k] for k in dfs]
    colors = [METHOD_COLORS[k] for k in dfs]
    ax2.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.75)
    ax2.set_ylabel("Memory (MB)")
    ax2.set_title("Mean Memory Footprint", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "memory_mean_comparison.png", dpi=150)
    plt.close(fig2)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df, summary = load_data()

    plot_iou_boxplot(df)
    plot_all_normal_distributions(df)
    plot_mean_std_comparison(summary)
    plot_normality_qq(df)
    plot_sample_overlays(df)
    plot_memory_footprint()

    print(f"Figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
