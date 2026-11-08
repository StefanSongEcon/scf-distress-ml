"""Prospective delinquency prediction on the SCF 2007-2009 panel.

Question: does a family's **2007** balance sheet predict delinquency reported in
the **2009** re-interview (payments late in the 12 months before the 2009
interview, i.e. during the financial crisis)?

Design
------
* Outcomes: ``LATE`` / ``LATE60`` from the 2009 interview (``P3004``/``P3005``).
* Feature sets (2007 values unless noted; same transforms as the cross-sections):
    - ``P_M1``..``P_M4``: nested 2007 blocks;
    - ``P_LAG``: 2007 delinquency only (``LATE07``, ``LATE6007``);
    - ``P_M4LAG``: 2007 M4 plus 2007 delinquency;
    - ``R_M4``: retrospective benchmark, 2009 M4 features for the 2009 outcome.
* Samples: all families; families with debt in 2007; onset (families not late in 2007).
* Validation: one panel, so no later wave exists. Family-grouped 5-fold
  cross-validation; the same folds for every implicate, feature set and model, so
  comparisons are paired. Tuning is nested (3 inner folds inside each training fold;
  hyperparameters chosen by mean inner log loss over inner folds and implicates).
* Weights: 2009 panel weight ``P42001`` (Fed ``WGT09``) for fitting and evaluation.
* Every model is fit per implicate; each family receives exactly one out-of-fold
  prediction per implicate.
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import features as F
from . import models as M
from . import panel

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed/panel_2007_2009.parquet"
OUT_TABLES = ROOT / "results/panel/tables"
OUT_PRED = ROOT / "results/panel/predictions"
SEED = 20261108
N_OUTER, N_INNER = 5, 3
XGB_GRID_PANEL = [{"max_depth": d} for d in (2, 3, 4)]
LAG_FEATURES = ["late07", "late6007"]
RAW_MAP = {"P3004": "X3004", "P3005": "X3005", "P5729": "X5729", "P7650": "X7650"}


# --------------------------------------------------------------------------- data
def _fed_ratios(d: pd.DataFrame) -> pd.DataFrame:
    """LEVRATIO and DEBT2INC exactly as in the Fed's main macro (not computed by the panel macro)."""
    debt, asset, inc = d["DEBT"], d["ASSET"], d["INCOME"]
    d["LEVRATIO"] = np.where((debt > 0) & (asset > 0), debt / asset.where(asset > 0),
                             np.where((debt > 0) & (asset == 0), 1.0, 0.0))
    d["DEBT2INC"] = np.where((debt > 0) & (inc > 0), debt / inc.where(inc > 0),
                             np.where((debt > 0) & (inc == 0), 10.0, 0.0))
    return d


def _prepare_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    extra = ["X3004", "X3005", "X5729", "X7650"] if year == 2007 else list(RAW_MAP)
    b, _ = panel.build_panel_year(year, extra_raw=extra)
    if year == 2009:
        b = b.drop(columns=[c for c in RAW_MAP.values() if c in b.columns]).rename(columns=RAW_MAP)
    b["LATE"] = (b["X3004"] == 5).astype(int)
    b["LATE60"] = ((b["X3005"] == 1) & (b["LATE"] == 1)).astype(int)
    b = _fed_ratios(b)
    feats = F.build_features(b)
    return b, feats


def build_panel_frame() -> tuple[pd.DataFrame, dict]:
    b07, f07 = _prepare_year(2007)
    b09, f09 = _prepare_year(2009)
    if not (b07["Y1"].to_numpy() == b09["Y1"].to_numpy()).all():
        raise ValueError("2007 and 2009 records are not aligned")
    if not np.allclose(b07["WGT"], b09["WGT"]):
        raise ValueError("panel weight differs between years")
    df = pd.DataFrame({"family_id": b09["family_id"], "implicate": b09["implicate"], "Y1": b09["Y1"],
                       "WGT": b09["WGT"], "LATE": b09["LATE"], "LATE60": b09["LATE60"],
                       "late07": b07["LATE"].astype(float), "late6007": b07["LATE60"].astype(float),
                       "hdebt07": b07["HDEBT"], "AGE07": b07["AGE"], "INCOME07": b07["INCOME"],
                       "X3004_09": b09["X3004"]})
    df = pd.concat([df, f07.add_prefix("f07_"), f09.add_prefix("f09_")], axis=1)
    info = {"families": int(df.family_id.nunique()), "records": len(df),
            "late60_gt_late_2009_fixed": int(((b09["X3005"] == 1) & (b09["LATE"] == 0)).sum()),
            "late60_gt_late_2007_fixed": int(((b07["X3005"] == 1) & (b07["LATE"] == 0)).sum())}
    return df, info


def feature_columns(fset: str) -> list[str]:
    if fset == "P_LAG":
        return list(LAG_FEATURES)
    if fset == "P_M4LAG":
        return [f"f07_{c}" for c in F.block_features("M4")] + LAG_FEATURES
    if fset.startswith("P_M"):
        return [f"f07_{c}" for c in F.block_features(fset[2:])]
    if fset == "R_M4":
        return [f"f09_{c}" for c in F.block_features("M4")]
    raise ValueError(fset)


def population_mask(df: pd.DataFrame, pop: str) -> pd.Series:
    if pop == "A_all":
        return pd.Series(True, index=df.index)
    if pop == "B_indebted07":
        return df["hdebt07"] == 1
    if pop == "C_onset":
        return df["late07"] == 0
    raise ValueError(pop)


def assign_folds(family_ids: np.ndarray, k: int, seed: int) -> dict:
    """Random family -> fold map (identical for all implicates, feature sets and models)."""
    fams = np.unique(family_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(fams))
    return dict(zip(fams[perm], np.arange(len(fams)) % k))


# --------------------------------------------------------------------------- modelling
def _norm(w):
    return w / w.mean()


def run_cell(df: pd.DataFrame, outcome: str, pop: str, fset: str, model: str, outer: dict) -> dict:
    t0 = time.time()
    cols = feature_columns(fset)
    sub = df[population_mask(df, pop)]
    fold = sub["family_id"].map(outer).to_numpy()
    grid = XGB_GRID_PANEL if model == "xgb" else M.GRIDS[model]
    fit = M.FITTERS[model]
    preds, chosen = [], []
    for k in range(N_OUTER):
        tr_all, te_all = sub[fold != k], sub[fold == k]
        inner = assign_folds(tr_all["family_id"].to_numpy(), N_INNER, SEED + 100 + k)
        # ---- inner tuning (pooled over implicates)
        scores = {g: [] for g in range(len(grid))}
        iters = {g: [] for g in range(len(grid))}
        if len(grid) > 1:
            ifold = tr_all["family_id"].map(inner).to_numpy()
            for m, j in product(range(1, 6), range(N_INNER)):
                imp = (tr_all["implicate"] == m).to_numpy()
                a, b = tr_all[imp & (ifold != j)], tr_all[imp & (ifold == j)]
                Xa, ya, wa = a[cols].to_numpy(), a[outcome].to_numpy(), _norm(a["WGT"].to_numpy())
                Xb, yb, wb = b[cols].to_numpy(), b[outcome].to_numpy(), b["WGT"].to_numpy()
                for g, hp in enumerate(grid):
                    f = fit(Xa, ya, wa, **hp, X_val=Xb, y_val=yb, w_val=_norm(wb), seed=SEED + m, n_jobs=2)
                    scores[g].append(ev.log_loss(yb, f.predict(Xb), wb))
                    if model == "xgb":
                        iters[g].append(f.params["n_estimators"])
            gbest = min(scores, key=lambda g: np.mean(scores[g]))
        else:
            gbest = 0
        hp = dict(grid[gbest])
        if model == "xgb":
            hp["n_estimators"] = int(round(np.mean(iters[gbest])))
        chosen.append({"fold": k, **hp})
        # ---- refit on the training fold per implicate, predict the held-out families
        for m in range(1, 6):
            a = tr_all[tr_all["implicate"] == m]
            b = te_all[te_all["implicate"] == m]
            f = fit(a[cols].to_numpy(), a[outcome].to_numpy(), _norm(a["WGT"].to_numpy()), **hp, seed=SEED + m,
                    n_jobs=2)
            preds.append(pd.DataFrame({"outcome": outcome, "population": pop, "fset": fset, "model": model,
                                       "implicate": m, "fold": k, "family_id": b["family_id"].to_numpy(),
                                       "Y1": b["Y1"].to_numpy(), "y": b[outcome].to_numpy(),
                                       "p": f.predict(b[cols].to_numpy()), "wgt": b["WGT"].to_numpy()}))
    pr = pd.concat(preds, ignore_index=True).sort_values(["implicate", "family_id"]).reset_index(drop=True)
    rows = []
    for m, g in pr.groupby("implicate"):
        rows.append({"outcome": outcome, "population": pop, "fset": fset, "model": model, "implicate": m,
                     "n_families": len(g), "events": int(g.y.sum()),
                     **ev.all_metrics(g.y.to_numpy(), g.p.to_numpy(), g.wgt.to_numpy())})
    return {"preds": pr, "metrics": pd.DataFrame(rows), "chosen": chosen, "seconds": time.time() - t0,
            "key": (outcome, pop, fset, model)}


def cells() -> list[tuple]:
    out = []
    for oc in F.OUTCOMES:
        for pop in ("A_all", "B_indebted07", "C_onset"):
            fsets = ["P_M1", "P_M2", "P_M3", "P_M4", "R_M4"]
            if pop != "C_onset":  # in the onset sample the lagged outcome is constant (0)
                fsets += ["P_LAG", "P_M4LAG"]
            for fs in fsets:
                for mdl in ("constant", "logit", "xgb"):
                    if mdl == "constant" and fs != "P_M1":
                        continue
                    if fs == "P_LAG" and mdl == "xgb":
                        continue  # two binary inputs: the logit is saturated enough
                    out.append((oc, pop, fs, mdl))
    return out


def run_all(jobs: int = 6) -> None:
    from joblib import Parallel, delayed

    t0 = time.time()
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_PRED.mkdir(parents=True, exist_ok=True)
    df, info = build_panel_frame()
    PROC.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROC, index=False)
    outer = assign_folds(df["family_id"].to_numpy(), N_OUTER, SEED)
    cs = cells()
    print(f"[panel] {info['families']} families, {len(cs)} cells, {jobs} jobs", flush=True)

    def one(c):
        r = run_cell(df, *c, outer)
        print(f"[panel] {'/'.join(c)} {r['seconds']:.0f}s hp={[{k: v for k, v in h.items() if k != 'fold'} for h in r['chosen']][:1]}",
              flush=True)
        return r

    res = Parallel(n_jobs=jobs, backend="loky")(delayed(one)(c) for c in cs)
    pd.concat([r["preds"] for r in res]).to_parquet(OUT_PRED / "oof_predictions.parquet", index=False)
    met = pd.concat([r["metrics"] for r in res], ignore_index=True)
    met.to_csv(OUT_TABLES / "cv_by_implicate.csv", index=False)
    keys = ["outcome", "population", "fset", "model"]
    vals = ["prevalence", "roc_auc", "pr_auc", "brier", "log_loss", "recall_top10", "lift_top10",
            "cal_intercept", "cal_slope", "citl", "events"]
    s = met.groupby(keys)[vals].agg(["mean", "min", "max"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s.reset_index().to_csv(OUT_TABLES / "cv_performance.csv", index=False)

    # descriptive: rates and persistence
    rows = []
    for m, g in df.groupby("implicate"):
        w = g["WGT"]
        rows.append({"implicate": m, "late07": np.average(g.late07, weights=w), "late6007": np.average(g.late6007, weights=w),
                     "late09": np.average(g.LATE, weights=w), "late6009": np.average(g.LATE60, weights=w),
                     "p_late09_given_late07": np.average(g.LATE[g.late07 == 1], weights=w[g.late07 == 1]),
                     "p_late09_given_not07": np.average(g.LATE[g.late07 == 0], weights=w[g.late07 == 0]),
                     "hdebt07": np.average(g.hdebt07, weights=w)})
    pd.DataFrame(rows).to_csv(OUT_TABLES / "panel_descriptives_by_implicate.csv", index=False)
    meta = {**info, "outer_folds": N_OUTER, "inner_folds": N_INNER, "seed": SEED,
            "chosen_hyperparameters": {"/".join(r["key"]): r["chosen"] for r in res},
            "cell_seconds": {"/".join(r["key"]): round(r["seconds"], 1) for r in res},
            "total_seconds": round(time.time() - t0, 1)}
    (ROOT / "results/panel/run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[panel] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run_all()
