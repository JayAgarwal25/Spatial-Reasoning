"""
print_results.py
----------------
Prints a side-by-side comparison table of all ablation eval results.

Usage:
    python print_results.py
    python print_results.py --results_dir results/
"""

import argparse
import json
from pathlib import Path


VARIANT_ORDER = ["full", "no_geom", "no_epi", "no_geom_no_epi"]
VARIANT_LABELS = {
    "full":            "QuantEpiGNN (full)",
    "no_geom":         "No geom constraint",
    "no_epi":          "No epistemic σ",
    "no_geom_no_epi":  "Plain GNN",
}

METRICS = [
    ("mae",                           "GNN MAE (m)"),
    ("baseline_mae",                  "Baseline MAE (m)"),
    ("mra",                           "MRA"),
    ("triangle_violation_rate",        "Tri-viol rate"),
    ("mean_residual",                  "Mean residual"),
    ("residual_hallucination_pearson", "Resid-Hall Pearson r"),
    ("residual_hallucination_auroc",   "Resid-Hall AUROC"),
    ("trigger_f1",                     "Trigger F1"),
]


def _get(d, key):
    if "." in key:
        outer, inner = key.split(".", 1)
        return d.get(outer, {}).get(inner, float("nan"))
    return d.get(key, float("nan"))


def load_results(results_dir: str):
    rd = Path(results_dir)
    data = {}
    for variant in VARIANT_ORDER:
        # Support both eval_{variant}.json and eval_{baseline}_{variant}.json
        candidates = list(rd.glob(f"eval_*_{variant}.json")) + [rd / f"eval_{variant}.json"]
        fpath = next((p for p in candidates if p.exists()), None)
        if fpath is None:
            continue
        with open(fpath) as f:  # type: ignore[arg-type]
            raw = json.load(f)
        # Pick the first baseline's summary
        for bl_tag, bl_data in raw.items():
            if bl_tag != "mock":
                data[variant] = bl_data.get("summary", bl_data)
                break
        else:
            # fall back to mock
            first = next(iter(raw.values()))
            data[variant] = first.get("summary", first)
    return data


def print_table(results: dict):
    variants = [v for v in VARIANT_ORDER if v in results]
    if not variants:
        print("No results found.")
        return

    col_w = 22
    label_w = 30

    header = f"{'Metric':<{label_w}}" + "".join(
        f"{VARIANT_LABELS.get(v, v):>{col_w}}" for v in variants
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print("  ABLATION RESULTS — Spatial457 Benchmark")
    print(sep)
    print(header)
    print(sep)

    for metric_key, metric_label in METRICS:
        row = f"{metric_label:<{label_w}}"
        for v in variants:
            val = _get(results[v], metric_key)
            if isinstance(val, float):
                cell = f"{val:.4f}" if not (val != val) else "   —"
            else:
                cell = str(val)
            row += f"{cell:>{col_w}}"
        print(row)

    print(sep)
    print()

    # Highlight best MAE
    maes = {v: _get(results[v], "mae") for v in variants}
    best = min(maes, key=lambda v: maes[v] if maes[v] == maes[v] else float("inf"))
    print(f"  Best MAE: {VARIANT_LABELS.get(best, best)}  ({maes[best]:.4f} m)\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    results = load_results(args.results_dir)
    print_table(results)


if __name__ == "__main__":
    main()
