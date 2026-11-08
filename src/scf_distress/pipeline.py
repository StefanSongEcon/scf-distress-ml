"""Command-line pipeline.

    python -m scf_distress.pipeline --config config.yaml --build   # Fed variables from raw data (+ check)
    python -m scf_distress.pipeline --config config.yaml --run     # audit tables, tuning, locked 2025 test
    python -m scf_distress.pipeline --config config.yaml --explain --variance --figures   # interpretation, SEs, figures

Design (see config.yaml): each model is fit separately on each of the 5
implicates. Hyperparameters are chosen by mean validation log loss over the
forward-chaining tuning folds and all implicates. The final model is refit on all
training waves and scored once on the test wave.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import features as F
from . import models as M
from .config import config_hash, load_config
from .splits import partition, splits_from_config
from .weights import normalize_training_weights

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- build
def cmd_build(cfg: dict) -> None:
    from . import prepare

    raw, proc = ROOT / cfg["data"]["raw_dir"], ROOT / cfg["data"]["processed_dir"]
    proc.mkdir(parents=True, exist_ok=True)
    checks = []
    for y in cfg["data"]["waves"]:
        t = time.time()
        b, info = prepare.build_wave(raw, y, extra_raw=["x3004", "x3005", "x5729", "x7650"])
        b.to_parquet(proc / f"bullit_{y}.parquet", index=False)
        ext_dir = ROOT / "data/external/summary_extract"
        if (ext_dir / str(y) / f"SCFP{y}.csv").exists():
            r = prepare.compare_to_extract(b, prepare.read_extract(ext_dir, y))
            r.insert(0, "year", y)
            checks.append(r)
            status = r["status"].value_counts().to_dict()
        else:
            status = "no extract to compare"
        print(f"[build] {y}: {len(b)} records, {time.time() - t:.1f}s, check={status}", flush=True)
    if checks:
        pd.concat(checks).to_csv(ROOT / "results/tables/macro_validation.csv", index=False)


# --------------------------------------------------------------------------- audit
def audit_tables(df: pd.DataFrame, label_info: dict, out: Path) -> None:
    rows, prev = [], []
    for y, g in df.groupby("year"):
        rows.append({"year": y, "records": len(g), "families": g["family_id"].nunique(),
                     "implicates_per_family": g.groupby("family_id")["implicate"].nunique().unique().tolist(),
                     "population_millions": g.groupby("implicate")["WGT"].sum().mean() * 5 / 1e6,
                     "families_label_varies_LATE": int((g.groupby("family_id")["LATE"].nunique() > 1).sum()),
                     "families_label_varies_LATE60": int((g.groupby("family_id")["LATE60"].nunique() > 1).sum()),
                     "late60_without_late_fixed": sum(1 for yy, _ in label_info["late60_without_late_families"]
                                                      if yy == y)})
        for pop in ("A_all", "B_indebted", "B2_payments_due"):
            gp = g[F.population_mask(g, pop)]
            for oc in F.OUTCOMES:
                per_imp = gp.groupby("implicate").apply(lambda s: np.average(s[oc], weights=s["WGT"]),
                                                        include_groups=False)
                share = g.groupby("implicate").apply(
                    lambda s: s.loc[F.population_mask(s, pop), "WGT"].sum() / s["WGT"].sum(), include_groups=False)
                prev.append({"year": y, "population": pop, "outcome": oc, "weighted_rate": per_imp.mean(),
                             "rate_min_imp": per_imp.min(), "rate_max_imp": per_imp.max(),
                             "population_share": share.mean(),
                             "families": gp["family_id"].nunique(),
                             "events_per_implicate": gp.groupby("implicate")[oc].sum().mean()})
    pd.DataFrame(rows).to_csv(out / "data_audit_by_year.csv", index=False)
    pd.DataFrame(prev).to_csv(out / "outcome_prevalence.csv", index=False)


# --------------------------------------------------------------------------- modelling
def run_cell(df: pd.DataFrame, X_all: np.ndarray, feat_names: list[str], outcome: str, population: str,
             block: str, model: str, cfg: dict, n_jobs_inner: int = 2) -> dict:
    """Tune and test one (outcome, population, block, model) cell across all implicates."""
    t0 = time.time()
    seed = int(cfg["seed"])
    cols = [feat_names.index(c) for c in F.block_features(block)]
    pop = F.population_mask(df, population)
    splits = splits_from_config(cfg)
    tuning = [s for s in splits if s.role == "tuning"]
    test = [s for s in splits if s.role == "test"][0]
    scheme = cfg["weights"]["train_scheme"]
    imps = range(1, int(cfg["data"]["n_implicates"]) + 1)
    grid = M.GRIDS[model]
    fit = M.FITTERS[model]
    key = dict(outcome=outcome, population=population, block=block, model=model)

    def arrays(sub_df, sub_X, train: bool):
        y = sub_df[outcome].to_numpy()
        w_raw = sub_df["WGT"].to_numpy()
        w = normalize_training_weights(pd.Series(w_raw), pd.Series(sub_df["year"].to_numpy()), scheme).to_numpy() \
            if train else w_raw / w_raw.mean()
        return sub_X[:, cols], y, w, w_raw

    # ---- tuning
    tune_rows, best_iters = [], {}
    val_cache = {}
    for sp, m in product(tuning, imps):
        base = pop & (df["implicate"] == m)
        tr_df, ev_df = partition(df[base], sp)
        Xtr, ytr, wtr, _ = arrays(tr_df, X_all[tr_df.index.to_numpy()], True)
        Xva, yva, wva, wva_raw = arrays(ev_df, X_all[ev_df.index.to_numpy()], False)
        for gi, hp in enumerate(grid):
            f = fit(Xtr, ytr, wtr, **hp, X_val=Xva, y_val=yva, w_val=wva, seed=seed + m, n_jobs=n_jobs_inner)
            p = f.predict(Xva)
            met = ev.all_metrics(yva, p, wva_raw, cfg["evaluation"]["top_k_share"])
            tune_rows.append({**key, "split": sp.name, "eval_year": sp.evaluate, "implicate": m, "grid_id": gi,
                              **{f"hp_{k}": v for k, v in f.params.items()},
                              "converged": f.extra.get("converged", True), **met})
            if model == "xgb":
                best_iters.setdefault(gi, []).append(f.params["n_estimators"])
            val_cache[(sp.name, m, gi)] = (p, wva_raw)
    tune = pd.DataFrame(tune_rows)
    score = tune.groupby("grid_id")["log_loss"].mean()
    gbest = int(score.idxmin())
    hp_final = dict(grid[gbest])
    if model == "xgb":
        hp_final["n_estimators"] = int(round(np.mean(best_iters[gbest])))
    tune["selected"] = tune["grid_id"] == gbest

    # ---- locked test: refit on all training waves, score the test wave once
    last_fold = max(tuning, key=lambda s: s.evaluate).name
    test_rows, preds, coefs = [], [], []
    for m in imps:
        base = pop & (df["implicate"] == m)
        tr_df, te_df = partition(df[base], test)
        Xtr, ytr, wtr, _ = arrays(tr_df, X_all[tr_df.index.to_numpy()], True)
        Xte, yte, _, wte_raw = arrays(te_df, X_all[te_df.index.to_numpy()], False)
        f = fit(Xtr, ytr, wtr, **hp_final, seed=seed + m, n_jobs=n_jobs_inner)
        p = f.predict(Xte)
        met = ev.all_metrics(yte, p, wte_raw, cfg["evaluation"]["top_k_share"])
        pv, wv = val_cache[(last_fold, m, gbest)]
        thr = ev.threshold_for_share(pv, wv, cfg["evaluation"]["top_k_share"])
        tm = ev.threshold_metrics(yte, p, thr, wte_raw)
        test_rows.append({**key, "split": test.name, "eval_year": test.evaluate, "implicate": m,
                          **{f"hp_{k}": v for k, v in f.params.items()}, **met,
                          "thr_from_val": thr, "thr_precision": tm["precision"], "thr_recall": tm["recall"],
                          "thr_flagged_share": tm["flagged_share"], "n_train": len(ytr), "n_test": len(yte),
                          "test_events": int(yte.sum())})
        preds.append(pd.DataFrame({**key, "implicate": m, "family_id": te_df["family_id"].to_numpy(),
                                   "y": yte, "p": p, "wgt": wte_raw}))
        if model == "logit":
            coefs.append(pd.DataFrame({**key, "implicate": m, "feature": F.block_features(block),
                                       "coef_per_unit": f.extra["coef"], "coef_standardized": f.extra["coef_std"]}))
    return {"tune": tune, "test": pd.DataFrame(test_rows), "preds": pd.concat(preds),
            "coefs": pd.concat(coefs) if coefs else None, "hp": hp_final, "seconds": time.time() - t0, "key": key}


def summarize(results: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    g = results.groupby(by)
    out = g[metrics].agg(["mean", "min", "max"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


def cmd_run(cfg: dict, cfg_path: Path, jobs: int) -> None:
    from joblib import Parallel, delayed

    t_start = time.time()
    tables, preds_dir = ROOT / "results/tables", ROOT / "results/predictions"
    tables.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    df = F.load_processed(ROOT / cfg["data"]["processed_dir"], cfg["data"]["waves"])
    df, label_info = F.make_labels(df)
    audit_tables(df, label_info, tables)
    Xdf = F.build_features(df)
    feat_names = list(Xdf.columns)
    X_all = Xdf.to_numpy(dtype=float)
    leak = set(feat_names) & ({b.lower() for b in F.BANNED} | F.BANNED)
    assert not leak, leak

    cells = [(o, p, b, m) for o in F.OUTCOMES for p in ("A_all", "B_indebted") for b in F.BLOCKS
             for m in ("constant", "logit", "xgb")]
    cells += [(o, "B2_payments_due", "M4", m) for o in F.OUTCOMES for m in ("constant", "logit", "xgb")]
    print(f"[run] {len(cells)} cells, {jobs} parallel jobs, config sha256={config_hash(cfg_path)[:12]}", flush=True)

    def one(c):
        r = run_cell(df, X_all, feat_names, *c, cfg)
        print(f"[cell] {'/'.join(c)} done in {r['seconds']:.1f}s, hp={r['hp']}", flush=True)
        return r

    res = Parallel(n_jobs=jobs, backend="loky", verbose=0)(delayed(one)(c) for c in cells)

    tune = pd.concat([r["tune"] for r in res], ignore_index=True)
    test = pd.concat([r["test"] for r in res], ignore_index=True)
    tune.to_csv(tables / "tuning_results.csv", index=False)
    test.to_csv(tables / "test_by_implicate.csv", index=False)
    pd.concat([r["preds"] for r in res], ignore_index=True).to_parquet(preds_dir / "test_2025.parquet", index=False)
    coefs = [r["coefs"] for r in res if r["coefs"] is not None]
    if coefs:
        c = pd.concat(coefs)
        c.groupby(["outcome", "population", "block", "model", "feature"])[["coef_per_unit", "coef_standardized"]] \
            .agg(["mean", "min", "max"]).pipe(lambda d: d.set_axis([f"{a}_{b}" for a, b in d.columns], axis=1)) \
            .reset_index().to_csv(tables / "logit_coefficients.csv", index=False)

    metrics = ["prevalence", "roc_auc", "pr_auc", "brier", "log_loss", "recall_top10", "lift_top10",
               "cal_intercept", "cal_slope", "citl"]
    keys = ["outcome", "population", "block", "model"]
    perf_test = summarize(test, keys + ["eval_year"], metrics + ["thr_precision", "thr_recall", "thr_flagged_share"])
    perf_test.insert(4, "role", "locked_test")
    sel = tune[tune["selected"]]
    perf_val = summarize(sel, keys + ["eval_year"], metrics)
    perf_val.insert(4, "role", "tuning_validation (selected hp; optimistic)")
    pd.concat([perf_test, perf_val], ignore_index=True).to_csv(tables / "model_performance.csv", index=False)

    import sklearn
    import xgboost
    meta = {"config_sha256": config_hash(cfg_path), "seed": cfg["seed"], "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__, "label_fixes": label_info, "features": feat_names,
            "selected_hyperparameters": {"/".join(r["key"].values()): r["hp"] for r in res},
            "cell_seconds": {"/".join(r["key"].values()): round(r["seconds"], 1) for r in res},
            "total_seconds": round(time.time() - t_start, 1)}
    (ROOT / "results/run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[run] finished in {time.time() - t_start:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--build", action="store_true", help="build Fed summary variables from raw data")
    ap.add_argument("--run", action="store_true", help="audit, tune, and run the locked test")
    ap.add_argument("--explain", action="store_true", help="importance, SHAP, subgroups, BNPL sensitivity (2025)")
    ap.add_argument("--variance", action="store_true", help="replicate-weight SEs for prevalence and 2025 metrics")
    ap.add_argument("--figures", action="store_true", help="render figures from saved results")
    ap.add_argument("--panel", action="store_true", help="2007-2009 panel: prospective models, CIs, importance")
    ap.add_argument("--deploy", action="store_true", help="build the parsimonious deployable risk score")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    cfg_path = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    cfg = load_config(cfg_path)
    if not (a.build or a.run or a.explain or a.variance or a.figures or a.panel or a.deploy):
        ap.error("choose at least one of --build, --run, --explain, --variance, --figures, --panel, --deploy")
    if a.build:
        cmd_build(cfg)
    if a.run:
        cmd_run(cfg, cfg_path, a.jobs)
    if a.explain:
        from . import explain
        explain.run_all(cfg)
    if a.variance:
        from . import variance
        variance.run_all(cfg, a.jobs)
    if a.figures:
        from . import figures
        figures.make_all()
    if a.panel:
        from . import panel_analysis, panel_model
        panel_model.run_all(a.jobs)
        panel_analysis.run_all(a.jobs)
    if a.deploy:
        from . import deploy
        deploy.run(a.jobs)


if __name__ == "__main__":
    main()
