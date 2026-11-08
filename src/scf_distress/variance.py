"""Design-based standard errors with SCF replicate weights (Fed codebook method).

From the SCF codebook (MACRO MEANIT):
* replicate weight r = max(0, MM_r) * max(0, WT1B_r), r = 1..999, defined for the
  first implicate of each case and merged on Y1 (missing = not selected = 0);
* sampling variance  = sum_r (theta_r - mean(theta_r))^2 / 998, computed on implicate 1;
* imputation variance = sum_m (theta_m - mean(theta_m))^2 / 4 over the 5 implicates;
* total SE = sqrt((6/5) * imputation variance + sampling variance); point estimate = mean over implicates.

For model metrics on the 2025 test wave, the fitted models (trained on 2010-2019)
are held fixed and the metric is recomputed under each replicate weight. The
resulting SE reflects sampling and imputation uncertainty of the 2025 evaluation,
not variability from re-drawing the training sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import features as F

ROOT = Path(__file__).resolve().parents[2]
N_REP = 999
Z95 = 1.959964


def load_replicates(rep_dir: str | Path, year: int) -> pd.DataFrame:
    """Return a frame indexed by family_id (= YY1) with 999 replicate weight columns R1..R999."""
    return load_replicate_file(Path(rep_dir) / str(year) / f"p{year % 100:02d}_rw1.dta", year)


def load_replicate_file(path: str | Path, year: int | str = "") -> pd.DataFrame:
    d = pd.read_stata(path, convert_categoricals=False)
    d.columns = [c.lower() for c in d.columns]
    imp = d["y1"].astype(np.int64) - 10 * d["yy1"].astype(np.int64)
    if not (imp == 1).all():
        raise ValueError(f"{year}: replicate file should hold implicate 1 only")
    mm = np.nan_to_num(d[[f"mm{i}" for i in range(1, N_REP + 1)]].to_numpy(float), nan=0.0)
    wt = np.nan_to_num(d[[f"wt1b{i}" for i in range(1, N_REP + 1)]].to_numpy(float), nan=0.0)
    rep = np.maximum(0, mm) * np.maximum(0, wt)
    return pd.DataFrame(rep, index=d["yy1"].astype(np.int64).to_numpy(), columns=[f"R{i}" for i in range(1, N_REP + 1)])


def combine(theta_imps, theta_reps) -> dict:
    """Fed combination of implicate and replicate estimates."""
    ti = np.asarray(theta_imps, dtype=float)
    tr = np.asarray(theta_reps, dtype=float)
    tr = tr[np.isfinite(tr)]
    v_imp = float(np.sum((ti - ti.mean()) ** 2) / (len(ti) - 1))
    v_samp = float(np.sum((tr - tr.mean()) ** 2) / (len(tr) - 1))
    se = float(np.sqrt(6 / 5 * v_imp + v_samp))
    est = float(ti.mean())
    return {"estimate": est, "se": se, "se_sampling": float(np.sqrt(v_samp)), "se_imputation": float(np.sqrt(v_imp)),
            "ci_lo": est - Z95 * se, "ci_hi": est + Z95 * se, "n_replicates_used": int(len(tr))}


# --------------------------------------------------------------------------- prevalence
def prevalence_se(df: pd.DataFrame, reps: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (year, pop, oc) in [(y, p, o) for y in sorted(df["year"].unique())
                            for p in ("A_all", "B_indebted", "B2_payments_due") for o in F.OUTCOMES]:
        g = df[df["year"] == year]
        g = g[F.population_mask(g, pop)]
        ti = g.groupby("implicate").apply(lambda s: np.average(s[oc], weights=s["WGT"]), include_groups=False)
        g1 = g[g["implicate"] == 1]
        R = reps[year].loc[g1["family_id"].to_numpy()].to_numpy()
        y = g1[oc].to_numpy(float)
        tr = (R * y[:, None]).sum(0) / R.sum(0)
        rows.append({"year": year, "population": pop, "outcome": oc, **combine(ti.to_numpy(), tr)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2025 metrics
METRICS = {
    "roc_auc": ev.roc_auc,
    "pr_auc": ev.pr_auc,
    "brier": ev.brier,
    "log_loss": ev.log_loss,
    "citl": lambda y, p, w: ev.calibration_intercept_slope(y, p, w)["citl_intercept"],
}


def _metric_reps(fn, y, p, R):
    out = np.empty(R.shape[1])
    for r in range(R.shape[1]):
        w = R[:, r]
        m = w > 0
        yy = y[m]
        out[r] = fn(yy, p[m], w[m]) if yy.min() != yy.max() else np.nan
    return out


def cell_metric_se(preds: pd.DataFrame, R1: pd.DataFrame, key: tuple, other: tuple | None = None) -> list[dict]:
    """SEs of 2025 metrics for one cell, or of the difference cell(key) - cell(other) when `other` is given."""
    def sel(k):
        o, p, b, m = k
        return preds[(preds.outcome == o) & (preds.population == p) & (preds.block == b) & (preds.model == m)]

    a = sel(key)
    c = sel(other) if other is not None else None
    rows = []
    for name, fn in METRICS.items():
        ti = []
        for m, g in a.groupby("implicate"):
            v = fn(g.y.to_numpy(), g.p.to_numpy(), g.wgt.to_numpy())
            if c is not None:
                h = c[c.implicate == m]
                assert (h.family_id.to_numpy() == g.family_id.to_numpy()).all()
                v -= fn(h.y.to_numpy(), h.p.to_numpy(), h.wgt.to_numpy())
            ti.append(v)
        g1 = a[a.implicate == 1]
        R = R1.loc[g1.family_id.to_numpy()].to_numpy()
        tr = _metric_reps(fn, g1.y.to_numpy(), g1.p.to_numpy(), R)
        if c is not None:
            h1 = c[c.implicate == 1]
            tr = tr - _metric_reps(fn, h1.y.to_numpy(), h1.p.to_numpy(), R)
        rows.append({"metric": name, **combine(ti, tr)})
    return rows


def run_all(cfg, jobs: int = 6) -> None:
    from joblib import Parallel, delayed

    tables = ROOT / "results/tables"
    rep_dir = ROOT / "data/external/replicate_weights"
    years = cfg["data"]["waves"]
    reps = {y: load_replicates(rep_dir, y) for y in years}
    for y, R in reps.items():
        print(f"[variance] {y}: {R.shape[0]} families, replicate totals {R.sum(axis=0).min() / 1e6:.1f}-"
              f"{R.sum(axis=0).max() / 1e6:.1f}M households", flush=True)

    df = F.load_processed(ROOT / cfg["data"]["processed_dir"], years)
    df, _ = F.make_labels(df)
    prevalence_se(df, reps).to_csv(tables / "outcome_prevalence_se.csv", index=False)
    print("[variance] prevalence done", flush=True)

    preds = pd.read_parquet(ROOT / "results/predictions/test_2025.parquet")
    R1 = reps[int(cfg["splits"]["test"])]
    cells = sorted({tuple(r) for r in preds[["outcome", "population", "block", "model"]].drop_duplicates().to_numpy()
                    if r[3] != "constant"})
    diffs = []
    for o in F.OUTCOMES:
        for p in ("A_all", "B_indebted"):
            for m in ("logit", "xgb"):
                diffs += [((o, p, "M2", m), (o, p, "M1", m)), ((o, p, "M3", m), (o, p, "M2", m)),
                          ((o, p, "M4", m), (o, p, "M3", m)), ((o, p, "M4", m), (o, p, "M1", m))]
            for b in ("M1", "M2", "M3", "M4"):
                diffs.append(((o, p, b, "xgb"), (o, p, b, "logit")))

    def job(k, other):
        rows = cell_metric_se(preds, R1, k, other)
        lab = dict(zip(["outcome", "population", "block", "model"], k))
        if other is not None:
            lab["minus"] = f"{other[2]} {other[3]}"
        return [{**lab, **r} for r in rows]

    out = Parallel(n_jobs=jobs)(delayed(job)(k, None) for k in cells)
    pd.DataFrame([r for rs in out for r in rs]).to_csv(tables / "test_metrics_ci.csv", index=False)
    print("[variance] cell metrics done", flush=True)
    out = Parallel(n_jobs=jobs)(delayed(job)(k, o) for k, o in diffs)
    d = pd.DataFrame([r for rs in out for r in rs])
    d["excludes_zero"] = (d.ci_lo > 0) | (d.ci_hi < 0)
    d.to_csv(tables / "test_differences_ci.csv", index=False)
    print("[variance] differences done", flush=True)


if __name__ == "__main__":
    from .config import load_config
    run_all(load_config(ROOT / "config.yaml"))
