"""Survey-weighted classification metrics and calibration diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .weights import kish_effective_n

_EPS = 1e-12


def _arrays(y, p, w):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    if not (len(y) == len(p) == len(w)):
        raise ValueError("y, p, w must have equal length")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y must be binary 0/1")
    if np.isnan(p).any() or np.isnan(w).any() or (w < 0).any():
        raise ValueError("p and w must be non-missing; w non-negative")
    return y, p, w


def weighted_prevalence(y, w=None) -> float:
    y, _, w = _arrays(y, np.zeros(len(y)), w)
    return float(np.sum(w * y) / np.sum(w))


def roc_auc(y, p, w=None) -> float:
    y, p, w = _arrays(y, p, w)
    return float(roc_auc_score(y, p, sample_weight=w))


def pr_auc(y, p, w=None) -> float:
    """Weighted average precision; compare against weighted prevalence."""
    y, p, w = _arrays(y, p, w)
    return float(average_precision_score(y, p, sample_weight=w))


def brier(y, p, w=None) -> float:
    y, p, w = _arrays(y, p, w)
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def log_loss(y, p, w=None) -> float:
    y, p, w = _arrays(y, p, w)
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def top_share_recall(y, p, w=None, share: float = 0.10) -> dict:
    """Recall and lift among the top `share` of predicted risk, measured by survey weight.

    The family straddling the cutoff contributes fractionally, so the selected
    weight is exactly `share` of the total. Ties in `p` are broken by averaging
    over the tied block.
    """
    y, p, w = _arrays(y, p, w)
    if not 0 < share <= 1:
        raise ValueError("share must be in (0, 1]")
    total_w, pos_w = w.sum(), np.sum(w * y)
    budget = share * total_w
    df = pd.DataFrame({"p": p, "wy": w * y, "w": w}).groupby("p", sort=True)[["wy", "w"]].sum()[::-1]
    cum_before = df["w"].cumsum().shift(fill_value=0.0)
    frac = ((budget - cum_before) / df["w"]).clip(0, 1)
    captured = float((frac * df["wy"]).sum())
    recall = captured / pos_w if pos_w > 0 else np.nan
    precision = captured / budget
    prev = pos_w / total_w
    return {"recall": recall, "precision": precision,
            "lift": precision / prev if prev > 0 else np.nan, "share": share}


def threshold_metrics(y, p, threshold: float, w=None) -> dict:
    """Precision/recall at a fixed threshold (chosen beforehand on validation data)."""
    y, p, w = _arrays(y, p, w)
    flag = p >= threshold
    tp = np.sum(w * y * flag)
    fp = np.sum(w * (1 - y) * flag)
    fn = np.sum(w * y * ~flag)
    return {"threshold": threshold,
            "precision": tp / (tp + fp) if tp + fp > 0 else np.nan,
            "recall": tp / (tp + fn) if tp + fn > 0 else np.nan,
            "flagged_share": float(np.sum(w * flag) / np.sum(w))}


def threshold_for_share(p, w=None, share: float = 0.10) -> float:
    """Score cutoff that flags `share` of weight (use on validation, then carry forward)."""
    _, p, w = _arrays(np.zeros(len(p)), p, w)
    order = np.argsort(-p, kind="stable")
    cum = np.cumsum(w[order]) / w.sum()
    idx = int(np.searchsorted(cum, share, side="left"))
    return float(p[order][min(idx, len(p) - 1)])


def calibration_table(y, p, w=None, n_bins: int = 10) -> pd.DataFrame:
    """Bins at weighted quantiles of p: mean predicted vs observed rate, counts and Kish n."""
    y, p, w = _arrays(y, p, w)
    order = np.argsort(p, kind="stable")
    cum = np.cumsum(w[order]) / w.sum()
    bins = np.empty(len(p), dtype=int)
    bins[order] = np.minimum((cum * n_bins - 1e-12).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        rows.append({"bin": b, "n": int(m.sum()), "n_eff": kish_effective_n(w[m]),
                     "weight_share": float(w[m].sum() / w.sum()),
                     "mean_pred": float(np.average(p[m], weights=w[m])),
                     "obs_rate": float(np.average(y[m], weights=w[m]))})
    return pd.DataFrame(rows)


def calibration_intercept_slope(y, p, w=None, max_iter: int = 100) -> dict:
    """Weighted logistic recalibration: logit P(y=1) = a + b * logit(p).

    Returns the slope b, the intercept a from the joint fit, and the
    calibration-in-the-large intercept (slope fixed at 1).
    """
    y, p, w = _arrays(y, p, w)
    lp = np.log(np.clip(p, _EPS, 1 - _EPS) / np.clip(1 - p, _EPS, 1))
    X = np.column_stack([np.ones_like(lp), lp])

    def irls(X, offset):
        beta = np.zeros(X.shape[1])
        for _ in range(max_iter):
            mu = 1 / (1 + np.exp(-(X @ beta + offset)))
            v = w * mu * (1 - mu) + _EPS
            step = np.linalg.solve(X.T @ (v[:, None] * X), X.T @ (w * (y - mu)))
            beta += step
            if np.max(np.abs(step)) < 1e-10:
                break
        return beta

    (a_citl,) = irls(X[:, :1], lp)
    if np.ptp(lp) < 1e-9:  # constant predictions: slope is not identified
        return {"intercept": np.nan, "slope": np.nan, "citl_intercept": float(a_citl)}
    a, b = irls(X, np.zeros_like(lp))
    return {"intercept": float(a), "slope": float(b), "citl_intercept": float(a_citl)}


def all_metrics(y, p, w=None, top_share: float = 0.10) -> dict:
    top = top_share_recall(y, p, w, top_share)
    cal = calibration_intercept_slope(y, p, w)
    return {"prevalence": weighted_prevalence(y, w), "roc_auc": roc_auc(y, p, w),
            "pr_auc": pr_auc(y, p, w), "brier": brier(y, p, w), "log_loss": log_loss(y, p, w),
            f"recall_top{int(top_share * 100)}": top["recall"], f"lift_top{int(top_share * 100)}": top["lift"],
            "cal_intercept": cal["intercept"], "cal_slope": cal["slope"], "citl": cal["citl_intercept"]}
