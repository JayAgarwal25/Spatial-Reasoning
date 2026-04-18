"""
metrics.py
----------
Shared evaluation metrics for the quantitative spatial hallucination
detection pipeline (Step 4).

Covers:
  - MAE / MRA (mean relative accuracy)
  - CircularEval / FlipEval accuracy (3DSRBench protocols)
  - Triangle-inequality violation rate
  - Residual–hallucination correlation (Pearson r, AUROC)
  - Precision / recall / F1 for re-grounding trigger decisions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scipy.stats import pearsonr
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """
    Aggregated evaluation result for one model on one dataset split.
    All numeric fields are Python floats so they serialise cleanly to JSON/YAML.
    """
    dataset: str = ""
    model_tag: str = ""

    # Distance quality
    mae: float = float("nan")
    mra: float = float("nan")           # mean relative accuracy (SpatialBench)

    # Classification accuracy (3DSRBench)
    circular_acc: float = float("nan")
    flip_acc: float = float("nan")

    # Triangle-inequality diagnostics
    triangle_violation_rate: float = float("nan")   # fraction of triples that violate TI
    mean_residual: float = float("nan")
    residual_hallucination_pearson: float = float("nan")
    residual_hallucination_auroc: float = float("nan")

    # Re-grounding trigger quality (when hallucination labels are available)
    trigger_precision: float = float("nan")
    trigger_recall: float = float("nan")
    trigger_f1: float = float("nan")

    # Extra per-ablation fields
    extras: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = {k: v for k, v in self.__dict__.items() if k != "extras"}
        d.update(self.extras)
        return d


# ---------------------------------------------------------------------------
# Distance quality
# ---------------------------------------------------------------------------

def compute_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Mean Absolute Error between predicted and ground-truth metric distances.

    Parameters
    ----------
    pred : (N,) array of predicted distances (metres or cm — caller must be consistent)
    gt   : (N,) array of ground-truth distances
    """
    pred, gt = np.asarray(pred, dtype=float), np.asarray(gt, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(gt)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - gt[mask])))


def compute_mra(
    pred: np.ndarray,
    gt: np.ndarray,
    thresholds: Tuple[float, ...] = (0.25, 0.50, 1.00),
) -> float:
    """
    Mean Relative Accuracy (SpatialBench protocol).

    For each threshold τ, MRA@τ = fraction of samples where
      |pred - gt| / gt  ≤  τ
    MRA = mean over all τ.

    Parameters
    ----------
    pred       : (N,) predicted distances
    gt         : (N,) ground-truth distances (must be > 0)
    thresholds : relative error thresholds to average over
    """
    pred, gt = np.asarray(pred, dtype=float), np.asarray(gt, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    if mask.sum() == 0:
        return float("nan")
    rel_err = np.abs(pred[mask] - gt[mask]) / gt[mask]
    accs = [float(np.mean(rel_err <= t)) for t in thresholds]
    return float(np.mean(accs))


# ---------------------------------------------------------------------------
# 3DSRBench protocols
# ---------------------------------------------------------------------------

def circular_eval(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
) -> float:
    """
    CircularEval accuracy.

    For each question the model is evaluated on both the original phrasing
    and a semantically equivalent circular rephrasing.  A sample is counted
    correct only if the model answers *both* correctly.

    Here we assume the caller has interleaved pairs: rows 2i and 2i+1 are
    a (original, rephrase) pair for sample i.

    Parameters
    ----------
    pred_labels : (2N,) integer class predictions
    gt_labels   : (2N,) integer ground-truth classes
    """
    pred_labels = np.asarray(pred_labels, dtype=int)
    gt_labels   = np.asarray(gt_labels,   dtype=int)
    assert len(pred_labels) % 2 == 0, "CircularEval requires even-length arrays (pairs)."
    n = len(pred_labels) // 2
    correct_orig   = pred_labels[0::2] == gt_labels[0::2]
    correct_rephr  = pred_labels[1::2] == gt_labels[1::2]
    return float(np.mean(correct_orig & correct_rephr))


def flip_eval(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
) -> float:
    """
    FlipEval accuracy.

    For each question, a semantically negated (flipped) version is also
    presented.  The model must answer *both* the original and the flipped
    version correctly (they have opposite ground-truth labels).

    Same layout assumption as CircularEval: rows 2i / 2i+1 are a pair.
    """
    pred_labels = np.asarray(pred_labels, dtype=int)
    gt_labels   = np.asarray(gt_labels,   dtype=int)
    assert len(pred_labels) % 2 == 0, "FlipEval requires even-length arrays (pairs)."
    correct_orig  = pred_labels[0::2] == gt_labels[0::2]
    correct_flip  = pred_labels[1::2] == gt_labels[1::2]
    return float(np.mean(correct_orig & correct_flip))


# ---------------------------------------------------------------------------
# Triangle-inequality diagnostics
# ---------------------------------------------------------------------------

def triangle_violation_rate(
    distances: Dict[Tuple[int, int], float],
    node_ids: List[int],
    epsilon: float = 1e-6,
) -> Tuple[float, float]:
    """
    Given a dict of pairwise distances (keyed by sorted node-id pairs),
    enumerate all triples (A, B, C) and check whether the triangle inequality
    holds for every permutation:
        d(A,C) ≤ d(A,B) + d(B,C)

    Returns
    -------
    violation_rate : fraction of (A,B,C) triples that violate the inequality
    mean_violation_magnitude : mean |d(A,C) - (d(A,B) + d(B,C))| over violations
    """
    from itertools import combinations

    def d(u, v):
        key = (min(u, v), max(u, v))
        return distances.get(key, None)

    violations = 0
    total = 0
    magnitudes = []

    for a, b, c in combinations(node_ids, 3):
        dac = d(a, c)
        dab = d(a, b)
        dbc = d(b, c)
        if dac is None or dab is None or dbc is None:
            continue
        total += 1
        surplus = dac - (dab + dbc)
        if surplus > epsilon:
            violations += 1
            magnitudes.append(surplus)

    if total == 0:
        return float("nan"), float("nan")
    vr = violations / total
    mv = float(np.mean(magnitudes)) if magnitudes else 0.0
    return float(vr), mv


def compute_residuals_from_distances(
    edge_index: np.ndarray,          # (2, E) — directed edge list
    edge_dist: np.ndarray,           # (E,)   — VLM-predicted distances
) -> np.ndarray:
    """
    Standalone residual computation (mirrors Step 2 logic) for use in
    post-hoc analysis and ablation A1 (residual as post-hoc filter).

    For each direct edge (A→C), finds all two-hop paths A→B→C and returns:
        r_AC = | d_AC_direct − mean_B(d_AB + d_BC) |

    Parameters
    ----------
    edge_index : (2, E) int array — edge_index[0] = source, edge_index[1] = target
    edge_dist  : (E,) float array — VLM predicted distance per edge

    Returns
    -------
    residuals : (E,) float array — 0.0 for edges with no two-hop path
    """
    src, tgt = edge_index[0], edge_index[1]
    E = len(edge_dist)

    # Build adjacency: node → {neighbour: distance}
    adj: Dict[int, Dict[int, float]] = {}
    for i in range(E):
        a, c = int(src[i]), int(tgt[i])
        adj.setdefault(a, {})[c] = float(edge_dist[i])

    residuals = np.zeros(E, dtype=float)
    for i in range(E):
        a, c = int(src[i]), int(tgt[i])
        d_ac = float(edge_dist[i])
        # two-hop paths: a→b→c
        two_hop_sums = []
        for b, d_ab in adj.get(a, {}).items():
            if b == c:
                continue
            d_bc = adj.get(b, {}).get(c, None)
            if d_bc is not None:
                two_hop_sums.append(d_ab + d_bc)
        if two_hop_sums:
            residuals[i] = abs(d_ac - float(np.mean(two_hop_sums)))
    return residuals


# ---------------------------------------------------------------------------
# Residual–hallucination correlation
# ---------------------------------------------------------------------------

def residual_hallucination_correlation(
    residuals: np.ndarray,
    hallucination_labels: np.ndarray,   # 1 = hallucinated, 0 = correct
) -> Dict[str, float]:
    """
    Measures whether the geometric consistency residual is a good detector
    of VLM hallucinations (the key publication-strength experiment from the
    README's known limitations section).

    Returns Pearson r and AUROC.  Requires sklearn.
    """
    residuals = np.asarray(residuals, dtype=float)
    labels    = np.asarray(hallucination_labels, dtype=int)

    if not SKLEARN_AVAILABLE:
        return {"pearson_r": float("nan"), "auroc": float("nan"),
                "avg_precision": float("nan"),
                "note": "sklearn not installed"}

    mask = np.isfinite(residuals) & np.isin(labels, [0, 1])
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return {"pearson_r": float("nan"), "auroc": float("nan"),
                "avg_precision": float("nan")}

    r, p = pearsonr(residuals[mask], labels[mask].astype(float))
    auroc = roc_auc_score(labels[mask], residuals[mask])
    ap    = average_precision_score(labels[mask], residuals[mask])
    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "auroc": float(auroc),
        "avg_precision": float(ap),
    }


# ---------------------------------------------------------------------------
# Re-grounding trigger quality
# ---------------------------------------------------------------------------

def trigger_precision_recall_f1(
    residuals: np.ndarray,
    hallucination_labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Treats `residual > threshold` as a positive prediction (edge is a
    hallucination).  Computes precision, recall, F1 against ground-truth
    hallucination labels.

    Parameters
    ----------
    residuals           : (E,) float
    hallucination_labels: (E,) int  — 1 = hallucinated, 0 = correct
    threshold           : float     — ε from trigger.py
    """
    residuals = np.asarray(residuals, dtype=float)
    labels    = np.asarray(hallucination_labels, dtype=int)

    pred_pos = (residuals > threshold).astype(int)
    tp = int(np.sum((pred_pos == 1) & (labels == 1)))
    fp = int(np.sum((pred_pos == 1) & (labels == 0)))
    fn = int(np.sum((pred_pos == 0) & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "tp": tp, "fp": fp, "fn": fn,
    }


def calibrate_threshold(
    residuals: np.ndarray,
    hallucination_labels: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, Dict[str, float]]:
    """
    Grid-search over candidate thresholds on a *validation* split and
    return the one maximising `metric` (one of 'f1', 'precision', 'recall').

    Usage: call on val set; apply the returned threshold to the test set.
    """
    candidates = np.linspace(
        float(np.nanmin(residuals)), float(np.nanmax(residuals)), 200
    )
    best_thresh = candidates[0]
    best_score  = -1.0
    best_stats  = {}

    for t in candidates:
        stats = trigger_precision_recall_f1(residuals, hallucination_labels, t)
        score = stats[metric]
        if score > best_score:
            best_score  = score
            best_thresh = float(t)
            best_stats  = stats

    return best_thresh, best_stats


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_result(result: EvalResult, indent: int = 2) -> None:
    pad = " " * indent
    print(f"\n{'='*60}")
    print(f"  Dataset : {result.dataset}")
    print(f"  Model   : {result.model_tag}")
    print(f"{'='*60}")
    for k, v in result.to_dict().items():
        if k in ("dataset", "model_tag"):
            continue
        if isinstance(v, float):
            print(f"{pad}{k:<40s}: {v:.4f}")
        else:
            print(f"{pad}{k:<40s}: {v}")
    print()
