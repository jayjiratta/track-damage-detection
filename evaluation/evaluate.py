"""
LBP (earlier demo) vs PaDiM -- evaluation for the report.

No real damage labels exist, so defects are synthetic: dark cracks and
pothole-like blotches painted into a probe box that is always track surface
(bottom-center of the frame). Same defects, same frames, same probe box for
every method. Frames are PaDiM's held-out split (never used to fit PaDiM):
  calibration half  holdout[::2]   -> thresholds (calibrate.py uses the same half)
  evaluation half   holdout[1::2]  -> every threshold-dependent number below

Part A  anomaly scoring only (no ROI, probe box): pixel/image AUROC on all
        held-out frames (threshold-free), and recall/FPR at a p99.5 threshold
        (from the calibration half) on the evaluation half.
Part B  full systems on the evaluation half:
          LBP demo          trapezoid ROI + LBP/STD + per-image p82 threshold
          PaDiM             YOLO26 track ROI + PaDiM + calibrated threshold
          PaDiM + paint     same, plus the paint filter (pipeline.py)
        false regions per clean frame, synthetic-defect detection rate.

Usage (from track-damage-pipeline/):
    python evaluation/evaluate.py [--part A|B|all]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
import lbp_baseline as lbp

DATA_DIR = ROOT.parent / "damage-detection" / "data"
OUT_DIR = ROOT / "results" / "evaluation"
PROBE = (0.30, 0.70, 0.62, 0.95)   # x0, x1, y0, y1 fractions -- always track surface
OP_PERCENTILE = 99.5
COLORS = {"LBP": "#C1440E", "PaDiM": "#2E86AB", "PaDiM + paint filter": "#1B998B"}


def probe_slice(h, w):
    x0, x1, y0, y1 = PROBE
    return slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w))


def add_synthetic_defects(img, rng):
    """1-2 dark random-walk cracks + (70%) one dark blotch inside the probe box."""
    out = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    ys, xs = probe_slice(h, w)
    base = np.median(img[ys, xs].reshape(-1, 3), axis=0)
    for _ in range(rng.integers(1, 3)):
        x, y = rng.uniform(xs.start + 20, xs.stop - 20), rng.uniform(ys.start + 20, ys.stop - 20)
        ang, pts = rng.uniform(0, np.pi), []
        for _ in range(rng.integers(8, 16)):
            pts.append((x, y))
            ang += rng.normal(0, 0.5)
            step = rng.uniform(8, 18)
            x = np.clip(x + step * np.cos(ang), xs.start, xs.stop - 1)
            y = np.clip(y + step * np.sin(ang), ys.start, ys.stop - 1)
        pts = np.array(pts, np.int32)
        thick = int(rng.integers(3, 7))
        cv2.polylines(out, [pts], False, (base * 0.35).tolist(), thick)
        cv2.polylines(mask, [pts], False, 255, thick)
    if rng.random() < 0.7:
        c = (int(rng.uniform(xs.start + 40, xs.stop - 40)), int(rng.uniform(ys.start + 25, ys.stop - 25)))
        axes = (int(rng.uniform(15, 35)), int(rng.uniform(8, 20)))
        blob = np.zeros((h, w), np.uint8)
        cv2.ellipse(blob, c, axes, rng.uniform(0, 180), 0, 360, 255, -1)
        dark = np.clip(base * 0.45 + rng.normal(0, 12, img.shape), 0, 255)
        out[blob > 0] = dark[blob > 0]
        mask |= blob
    return out.astype(np.uint8), mask


def load_frames():
    holdout = json.loads((ROOT / "models" / "padim_split.json").read_text())["holdout"]
    frames = []
    for i, name in enumerate(holdout):
        img = cv2.imread(str(DATA_DIR / name))
        bad, gt = add_synthetic_defects(img, np.random.default_rng(1000 + i))
        frames.append({"name": name, "img": img, "bad": bad, "gt": gt, "calib": i % 2 == 0})
    return frames


# ------------------------------------------------------------------ Part A
def part_a(frames):
    sys.path.insert(0, str(ROOT / "padim"))
    from padim import PaDiM
    padim = PaDiM.load(ROOT / "models" / "padim_resnet18.pt")
    scorers = {"LBP": lambda im: lbp.anomaly_score(im), "PaDiM": padim.score}

    res = {}
    fig_roc, ax_roc = plt.subplots(figsize=(6.5, 5.5))
    fig_img, axes_img = plt.subplots(1, 2, figsize=(12, 4.5))
    for (method, score), ax_img in zip(scorers.items(), axes_img):
        clean_px = {True: [], False: []}
        bad_px, bad_lbl, eval_bad_px, eval_bad_lbl = [], [], [], []
        img_clean, img_bad = [], []
        for f in frames:
            ys, xs = probe_slice(*f["img"].shape[:2])
            c = score(f["img"])[ys, xs]
            b = score(f["bad"])[ys, xs]
            lbl = f["gt"][ys, xs] > 0
            clean_px[f["calib"]].append(c.ravel())
            bad_px.append(b.ravel())
            bad_lbl.append(lbl.ravel())
            if not f["calib"]:
                eval_bad_px.append(b.ravel())
                eval_bad_lbl.append(lbl.ravel())
            img_clean.append(float(c.max()))
            img_bad.append(float(b.max()))

        s, y = np.concatenate(bad_px), np.concatenate(bad_lbl)
        fpr, tpr, _ = roc_curve(y, s)
        pix_auc = float(roc_auc_score(y, s))
        img_auc = float(roc_auc_score([0] * len(img_clean) + [1] * len(img_bad), img_clean + img_bad))
        thr = float(np.percentile(np.concatenate(clean_px[True]), OP_PERCENTILE))
        ev_s, ev_y = np.concatenate(eval_bad_px), np.concatenate(eval_bad_lbl)
        res[method] = {
            "pixel_auroc": pix_auc, "image_auroc": img_auc,
            "threshold_p99.5_calib": thr,
            "eval_recall_at_threshold": float((ev_s[ev_y] > thr).mean()),
            "eval_clean_fpr_at_threshold": float((np.concatenate(clean_px[False]) > thr).mean()),
        }
        ax_roc.plot(fpr, tpr, color=COLORS[method], lw=2.5, label=f"{method} (AUROC = {pix_auc:.3f})")
        bins = np.linspace(min(img_clean + img_bad), max(img_clean + img_bad), 20)
        ax_img.hist(img_clean, bins=bins, alpha=0.55, color="#888888", label="normal")
        ax_img.hist(img_bad, bins=bins, alpha=0.55, color=COLORS[method], label="with synthetic defect")
        ax_img.set_title(f"{method}: image-level max score (AUROC = {img_auc:.3f})", fontweight="bold")
        ax_img.set_xlabel("max anomaly score in probe box")
        ax_img.set_ylabel("number of images")
        ax_img.legend()

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Pixel-level ROC: LBP vs PaDiM (synthetic defects)", fontweight="bold")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig(OUT_DIR / "roc_pixel_lbp_vs_padim.png", dpi=170)
    fig_img.tight_layout()
    fig_img.savefig(OUT_DIR / "image_scores_lbp_vs_padim.png", dpi=170)
    plt.close("all")
    return res


# ------------------------------------------------------------------ Part B
def components(mask):
    n, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, n)]


def system_stats(detect, frames):
    clean_regions, clean_ratio, det_hits, det_total, bad_false = [], [], 0, 0, []
    for f in frames:
        dmg, area = detect(f["img"])
        clean_regions.append(len(components(dmg)))
        clean_ratio.append(float((dmg > 0).sum()) / max(1, area))
        dmg_b, _ = detect(f["bad"])
        gt_c, det_c = components(f["gt"]), components(dmg_b)
        det_hits += sum(any((g & d).any() for d in det_c) for g in gt_c)
        det_total += len(gt_c)
        bad_false.append(sum(not (d & (f["gt"] > 0)).any() for d in det_c))
    return {
        "clean_regions_per_frame_mean": float(np.mean(clean_regions)),
        "clean_frames_with_zero_regions": float(np.mean(np.array(clean_regions) == 0)),
        "clean_flagged_fraction_of_roi_mean": float(np.mean(clean_ratio)),
        "synthetic_defect_detection_rate": det_hits / max(1, det_total),
        "synthetic_defects_total": det_total,
        "false_regions_per_defect_frame_mean": float(np.mean(bad_false)),
        "clean_regions_per_frame": clean_regions,
    }


def part_b(frames):
    from pipeline import DamagePipeline
    ev = [f for f in frames if not f["calib"]]
    pipe = DamagePipeline("yolo")

    def run_lbp(im):
        tm, _, dmg = lbp.run(im)
        return dmg, int((tm > 0).sum())

    def run_padim(paint):
        def f(im):
            pipe.paint_filter = paint
            r = pipe(im)
            return r.damage, int((r.roi > 0).sum())
        return f

    systems = {"LBP": run_lbp, "PaDiM": run_padim(False), "PaDiM + paint filter": run_padim(True)}
    res = {k: system_stats(v, ev) for k, v in systems.items()}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    names = list(res)
    panels = [("clean_regions_per_frame_mean", "False regions per normal frame\n(lower is better)", "{:.2f}"),
              ("synthetic_defect_detection_rate", "Synthetic defect detection rate\n(higher is better)", "{:.0%}"),
              ("false_regions_per_defect_frame_mean", "False regions per defect frame\n(lower is better)", "{:.2f}")]
    for ax, (key, title, fmt) in zip(axes, panels):
        vals = [res[n][key] for n in names]
        bars = ax.bar(names, vals, color=[COLORS[n] for n in names], alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=11)
        ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"System-level comparison ({len(ev)} held-out evaluation frames)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "system_lbp_vs_padim.png", dpi=170)
    plt.close(fig)

    # qualitative: a clean frame, the frame with the most PaDiM (no filter) false regions, a defect frame
    worst = int(np.argmax(res["PaDiM"]["clean_regions_per_frame"]))
    clean0 = int(np.argmin(res["PaDiM"]["clean_regions_per_frame"]))
    picks = [(ev[clean0]["img"], None), (ev[worst]["img"], None), (ev[1]["bad"], ev[1]["gt"])]
    rows = []
    for im, gt in picks:
        cells = [im]
        for name, fn in systems.items():
            dmg, _ = fn(im)
            vis = im.copy()
            vis[dmg > 0] = (0.35 * vis[dmg > 0] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
            cnts, _ = cv2.findContours(dmg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)
            if gt is not None:
                g, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, g, -1, (0, 255, 0), 2)
            cells.append(vis)
        rows.append(cv2.hconcat([cv2.resize(c, (480, 270)) for c in cells]))
    header = np.zeros((36, 480 * 4, 3), np.uint8)
    for j, t in enumerate(["Original", "LBP (demo)", "PaDiM", "PaDiM + paint filter"]):
        cv2.putText(header, t, (10 + 480 * j, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(OUT_DIR / "qualitative_lbp_vs_padim.jpg"), cv2.vconcat([header] + rows))
    for r in res.values():
        r.pop("clean_regions_per_frame")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["A", "B", "all"])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = load_frames()
    metrics_path = OUT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    if args.part in ("A", "all"):
        metrics["A_scoring"] = part_a(frames)
    if args.part in ("B", "all"):
        metrics["B_system"] = part_b(frames)
    metrics["protocol"] = {"n_holdout": len(frames), "n_calib": sum(f["calib"] for f in frames),
                           "n_eval": sum(not f["calib"] for f in frames), "probe_box": PROBE,
                           "operating_percentile": OP_PERCENTILE}
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
