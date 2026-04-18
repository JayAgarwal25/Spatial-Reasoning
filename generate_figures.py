"""
generate_figures.py  —  PDF only, no PNG
"""

import json, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FIGURES_DIR = Path("submission/05_results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "figure.dpi":        150,
})

C = {
    "baseline": "#e07b54",
    "full":     "#4c72b0",
    "no_geom":  "#55a868",
    "no_epi":   "#c44e52",
    "plain":    "#8172b2",
    "feedback": "#937860",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_scenes(path="results_20k/eval_qwen_feedback_plain.json"):
    with open(path) as f:
        d = json.load(f)
    return d["qwen2-vl-7b"]["per_scene"]

def flatten(scenes, key):
    out = []
    for s in scenes: out.extend(s[key])
    return np.array(out, dtype=float)

def parse_log(path):
    train, val = [], []
    pat = re.compile(r"Epoch\s+(\d+)/\d+\s+train\s+loss=([\d.]+).*val\s+loss=([\d.]+)")
    for line in open(path):
        m = pat.search(line)
        if m:
            e = int(m.group(1))
            train.append((e, float(m.group(2))))
            val.append((e, float(m.group(3))))
    return train, val

def save(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {name}.png")


# ---------------------------------------------------------------------------
# Fig 1 — Main results bar chart
# ---------------------------------------------------------------------------
def fig1_main():
    labels = ["Qwen2-VL-7B\n(baseline)", "GNN only\n(plain)", "GNN +\nfeedback loop"]
    maes   = [6.1924, 2.8885, 2.8868]
    cols   = [C["baseline"], C["plain"], C["feedback"]]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(labels, maes, color=cols, width=0.45, edgecolor="white", zorder=3)
    for bar, v in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.08,
                f"{v:.2f} m", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # reduction arrow
    x0, x1 = bars[0].get_x()+bars[0].get_width()/2, bars[1].get_x()+bars[1].get_width()/2
    ymid = (maes[0]+maes[1])/2
    ax.annotate("", xy=(x1, maes[1]+0.15), xytext=(x0, maes[0]-0.15),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.5))
    ax.text((x0+x1)/2, ymid, "−53%", ha="center", va="center", fontsize=9, color="#444",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ccc", lw=0.8))

    ax.set_ylabel("MAE (m) ↓")
    ax.set_ylim(0, 8.0)
    ax.set_title("Distance Prediction MAE — Spatial457 (999 scenes)")
    fig.tight_layout()
    save(fig, "fig1_main_results")


# ---------------------------------------------------------------------------
# Fig 2 — Ablation bar chart
# ---------------------------------------------------------------------------
def fig2_ablation():
    labels = ["QuantEpiGNN\n(full)", "No geom\nconstraint", "No epistemic\nσ", "Plain GNN"]
    maes   = [3.4048, 3.2050, 2.8885, 2.8885]
    cols   = [C["full"], C["no_geom"], C["no_epi"], C["plain"]]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(labels, maes, color=cols, width=0.45, edgecolor="white", zorder=3)
    for bar, v in zip(bars, maes):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.03, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.axhline(6.1924, color=C["baseline"], ls="--", lw=1.3, alpha=0.85, zorder=2)
    ax.text(3.42, 6.30, "Qwen2-VL-7B (6.19 m)",
            color=C["baseline"], fontsize=8.5, ha="right")
    ax.set_ylabel("MAE (m) ↓")
    ax.set_ylim(0, 7.2)
    ax.set_title("Ablation Study — Geometric Constraint and Epistemic Uncertainty")
    fig.tight_layout()
    save(fig, "fig2_ablation")


# ---------------------------------------------------------------------------
# Fig 3 — Scatter: VLM vs GT  |  GNN vs GT   (clipped, log scale)
# ---------------------------------------------------------------------------
def fig3_scatter(scenes):
    gt  = flatten(scenes, "gt")
    vlm = flatten(scenes, "pred_baseline")
    gnn = flatten(scenes, "pred_gnn")

    # Clip VLM outliers at p95 for readability; keep finite only
    finite = np.isfinite(vlm) & np.isfinite(gt) & np.isfinite(gnn) & (vlm > 0) & (gnn > 0) & (gt > 0)
    gt, vlm, gnn = gt[finite], vlm[finite], gnn[finite]

    clip = np.percentile(vlm, 95)   # ~6m
    mask = vlm <= clip
    gt_c, vlm_c, gnn_c = gt[mask], vlm[mask], gnn[mask]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(gt_c), min(4000, len(gt_c)), replace=False)

    lim = (0.5, 10)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, preds, col, title, mae_all in [
        (axes[0], vlm_c, C["baseline"], "Qwen2-VL-7B Predictions vs GT",
         np.mean(np.abs(vlm - gt))),
        (axes[1], gnn_c, C["plain"],    "GNN Predictions vs GT",
         np.mean(np.abs(gnn - gt))),
    ]:
        ax.scatter(gt_c[idx], preds[idx], alpha=0.12, s=5, color=col, rasterized=True)
        ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel("GT distance (m)"); ax.set_ylabel("Predicted distance (m)")
        ax.set_title(f"{title}\nMAE = {mae_all:.2f} m (full dataset)")
        note = "(VLM p95 shown)" if ax is axes[0] else ""
        if note: ax.text(0.97, 0.04, note, transform=ax.transAxes,
                         fontsize=7.5, ha="right", color="#888")

    fig.suptitle("Predicted vs GT Distances — Spatial457 Test Set", fontsize=12)
    fig.tight_layout()
    save(fig, "fig3_scatter_predictions")


# ---------------------------------------------------------------------------
# Fig 4 — Residual distribution  (clipped at p98, log y)
# ---------------------------------------------------------------------------
def fig4_residuals(scenes):
    res = flatten(scenes, "residuals")
    res = res[np.isfinite(res) & (res >= 0)]
    clip = np.percentile(res, 98)   # ~10m
    res_c = res[res <= clip]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(res_c, bins=50, color=C["plain"], edgecolor="white", lw=0.4, alpha=0.85)
    ax.axvline(0.3, color=C["baseline"], ls="--", lw=1.8,
               label=f"Trigger threshold ε = 0.3\n(flags {100*(res>0.3).mean():.0f}% of edges)")
    ax.set_xlabel("Geometric consistency residual (m)")
    ax.set_ylabel("Edge count")
    ax.set_title("Per-Edge Residual Distribution (GNN output, 999 scenes)")
    ax.text(0.98, 0.97, f"p98 clip at {clip:.1f} m\nmedian = {np.median(res):.2f} m",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#555")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "fig4_residual_distribution")


# ---------------------------------------------------------------------------
# Fig 5 — Error distributions as overlapping histograms  (clipped at p99)
# ---------------------------------------------------------------------------
def fig5_errors(scenes):
    gt  = flatten(scenes, "gt")
    vlm = flatten(scenes, "pred_baseline")
    gnn = flatten(scenes, "pred_gnn")
    finite = np.isfinite(vlm) & np.isfinite(gt) & np.isfinite(gnn)
    err_v = np.abs(vlm[finite] - gt[finite])
    err_g = np.abs(gnn[finite] - gt[finite])

    clip = np.percentile(err_v, 99)   # ~22m
    bins = np.linspace(0, min(clip, 15), 60)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(err_v[err_v <= clip], bins=bins, color=C["baseline"], alpha=0.5,
            label=f"Qwen2-VL-7B  (MAE={err_v.mean():.2f} m)", edgecolor="white", lw=0.3)
    ax.hist(err_g[err_g <= clip], bins=bins, color=C["plain"], alpha=0.7,
            label=f"GNN plain  (MAE={err_g.mean():.2f} m)", edgecolor="white", lw=0.3)
    ax.set_xlabel("Absolute error (m)")
    ax.set_ylabel("Edge count")
    ax.set_title("Error Distribution — VLM vs GNN (clipped at 15 m)")
    ax.legend(fontsize=9)
    ax.text(0.98, 0.97, f"VLM p99 = {np.percentile(err_v,99):.1f} m",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#888")
    fig.tight_layout()
    save(fig, "fig5_error_distribution")


# ---------------------------------------------------------------------------
# Fig 6 — Training curves
# ---------------------------------------------------------------------------
def fig6_training():
    logs = [
        ("Plain GNN",          "logs_20k/train_plain.log",   C["plain"]),
        ("No epistemic σ",     "logs_20k/train_no_epi.log",  C["no_epi"]),
        ("No geom constraint", "logs_20k/train_no_geom.log", C["no_geom"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, path, col in logs:
        try:
            tr, va = parse_log(path)
            if not tr: continue
            te, tl = zip(*tr); ve, vl = zip(*va)
            axes[0].plot(te, tl, color=col, lw=1.5, label=label)
            axes[1].plot(ve, vl, color=col, lw=1.5, label=label)
        except FileNotFoundError:
            pass
    for ax, title in zip(axes, ["Train Loss", "Validation Loss"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(title); ax.legend(fontsize=8.5)
    fig.suptitle("Training Curves — Spatial457-20k (100 epochs)", fontsize=12)
    fig.tight_layout()
    save(fig, "fig6_training_curves")


# ---------------------------------------------------------------------------
# Fig 7 — Qualitative feedback (4 scenes, 2×2 grid)
# ---------------------------------------------------------------------------
def fig7_qualitative(scene_indices, tag="fig7_qualitative_feedback"):
    import cv2, torch
    from itertools import combinations as combs
    from step4_evaluation.spatialqa_eval import load_spatial457
    from step1_scene_graph.schemas import ObjectNode, RelationEdge
    from step3_visual_agent.actions import draw_bbox, draw_line

    with open("results_20k/eval_qwen_feedback_plain.json") as f:
        eval_data = json.load(f)["qwen2-vl-7b"]["per_scene"]

    scenes = load_spatial457("data/spatial457")

    rows = []
    for si in scene_indices:
        scene = scenes[si]
        sr    = eval_data[si]
        img   = cv2.imread(scene["image_path"])
        if img is None: continue

        objects  = scene["objects"]
        edges    = list(combs(range(len(objects)), 2))
        residuals = sr["residuals"]
        pred_gnn  = sr["pred_gnn"]

        annotated = img.copy()
        # top-3 highest residual edges
        top = sorted(enumerate(residuals),
                     key=lambda x: x[1] if np.isfinite(x[1]) else 0, reverse=True)[:3]

        for edge_idx, res in top:
            if edge_idx >= len(edges): continue
            i, j = edges[edge_idx]
            def node(obj, idx):
                bb = obj.get("bbox", [0,0,10,10])
                cx, cy = obj.get("pixel_center", [0,0])
                return ObjectNode(id=idx, label=obj["id"], bbox=bb, confidence=1.0,
                                  center=[cx,cy], width=float(bb[2]), height=float(bb[3]),
                                  area=float(bb[2])*float(bb[3]))
            ni, nj = node(objects[i], i), node(objects[j], j)
            rel = RelationEdge(subject_id=i, object_id=j, predicate="near", confidence=0.5)
            r1 = draw_bbox(annotated, ni, edge_idx, res)
            r2 = draw_bbox(r1.image, nj, edge_idx, res)
            d  = pred_gnn[edge_idx] if edge_idx < len(pred_gnn) else 0.0
            r3 = draw_line(r2.image, ni, nj, rel, edge_idx, res, d)
            annotated = r3.image

        orig = cv2.cvtColor(img,       cv2.COLOR_BGR2RGB)
        ann  = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        rows.append((orig, ann, scene["scene_id"]))

    # Individual figures per scene
    for orig, ann, sid in rows:
        short = sid.replace("superCLEVR_new_0", "scene_")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(orig); axes[0].axis("off"); axes[0].set_title("Original image")
        axes[1].imshow(ann);  axes[1].axis("off")
        axes[1].set_title("Step 3: high-residual edges annotated")
        fig.suptitle(f"Visual Grounding Feedback — {sid}", fontsize=11)
        fig.tight_layout()
        save(fig, f"{tag}_{short}")

    # Combined grid figure
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4.5 * n))
    if n == 1: axes = [axes]
    for ax_row, (orig, ann, sid) in zip(axes, rows):
        ax_row[0].imshow(orig); ax_row[0].axis("off"); ax_row[0].set_title("Original")
        ax_row[1].imshow(ann);  ax_row[1].axis("off")
        ax_row[1].set_title("Step 3: annotated")
        ax_row[0].set_ylabel(sid.replace("superCLEVR_new_0", "Scene "), fontsize=9)
    fig.suptitle("Visual Grounding Feedback — Top-3 High-Residual Edges Per Scene",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, tag)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading eval results...")
    scenes = load_scenes()

    print("Generating figures 1–6 (no GPU needed)...")
    fig1_main()
    fig2_ablation()
    fig3_scatter(scenes)
    fig4_residuals(scenes)
    fig5_errors(scenes)
    fig6_training()

    print("Generating qualitative figures 7a + 7b (CPU only, no VLM)...")
    sys.path.insert(0, ".")
    fig7_qualitative([0, 1, 2, 3],  tag="fig7a_qualitative_4scenes")
    fig7_qualitative([10, 25, 50, 99], tag="fig7b_qualitative_4scenes_more")

    print(f"\nAll figures saved to {FIGURES_DIR}/")
