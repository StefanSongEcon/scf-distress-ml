"""Interpretation and robustness analyses on the locked 2025 test wave.

Nothing here changes model specifications: the final models are refit exactly as
in the pipeline (same data, hyperparameters and seeds) and then *described*.
None of these quantities are causal effects.

* permutation importance (weighted, held-out 2025): single features and whole blocks;
* TreeSHAP contributions for the boosted M4 models (xgboost ``pred_contribs``);
* conditional associations: observed vs predicted delinquency by bins of key variables;
* partial dependence (synthetic: one input changed, all others held at observed values);
* subgroup performance by age class and income quintile;
* sensitivity: 2025 features with buy-now-pay-later (BNPL) removed from Fed debt and payment
  totals, since BNPL is measured only from 2025 on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import features as F
from . import models as M
from .config import load_config
from .splits import partition, splits_from_config
from .weights import kish_effective_n, normalize_training_weights

ROOT = Path(__file__).resolve().parents[2]
EXTRA_COLS = ["BNPL", "REVPAY", "ASSET"]  # BNPL sensitivity needs these
PDP_VARS = {"age": "AGE", "asinh_liq": "LIQ", "debt2inc": "DEBT2INC", "pirtotal": "PIRTOTAL",
            "asinh_networth": "NETWORTH", "asinh_ccbal": "CCBAL"}


# --------------------------------------------------------------------------- data + final models
def load(cfg):
    cols = list(dict.fromkeys(F.SOURCE_COLS + EXTRA_COLS))
    frames = [pd.read_parquet(ROOT / cfg["data"]["processed_dir"] / f"bullit_{y}.parquet", columns=cols)
              for y in cfg["data"]["waves"]]
    df, info = F.make_labels(pd.concat(frames, ignore_index=True))
    X = F.build_features(df)
    return df, X


def fit_final(df, X, cfg, outcome, population, block, model, hp, implicate):
    """Refit one final model on all training waves for one implicate; return model and test rows."""
    test = [s for s in splits_from_config(cfg) if s.role == "test"][0]
    cols = F.block_features(block)
    base = F.population_mask(df, population) & (df["implicate"] == implicate)
    tr, te = partition(df[base], test)
    w = normalize_training_weights(pd.Series(tr["WGT"].to_numpy()), pd.Series(tr["year"].to_numpy()),
                                   cfg["weights"]["train_scheme"]).to_numpy()
    f = M.FITTERS[model](X.loc[tr.index, cols].to_numpy(), tr[outcome].to_numpy(), w, **hp,
                         seed=int(cfg["seed"]) + implicate, n_jobs=2)
    return f, te, cols


def _hp(meta, key):
    return meta["selected_hyperparameters"]["/".join(key)]


# --------------------------------------------------------------------------- analyses
def permutation_importance(f, Xte, y, w, names, groups, n_repeats, rng):
    base_p = f.predict(Xte)
    base_auc, base_ll = ev.roc_auc(y, base_p, w), ev.log_loss(y, base_p, w)
    rows = []
    items = [(n, [i]) for i, n in enumerate(names)] + [(g, [names.index(c) for c in cs]) for g, cs in groups.items()]
    for name, idx in items:
        d_auc, d_ll = [], []
        for _ in range(n_repeats):
            Xp = Xte.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, idx] = Xte[perm][:, idx]  # permute the group jointly (keeps within-group structure)
            p = f.predict(Xp)
            d_auc.append(base_auc - ev.roc_auc(y, p, w))
            d_ll.append(ev.log_loss(y, p, w) - base_ll)
        rows.append({"feature": name, "kind": "group" if len(idx) > 1 or name in groups else "feature",
                     "auc_drop": float(np.mean(d_auc)), "logloss_increase": float(np.mean(d_ll))})
    return pd.DataFrame(rows)


def binned_association(te, X_te, p, var_raw, w, n_bins=10):
    """Observed vs mean predicted rate by weighted-quantile bins of a raw variable (2025)."""
    x = te[var_raw].to_numpy(float)
    y = te["_y"].to_numpy(float)
    order = np.argsort(x, kind="stable")
    cum = np.cumsum(w[order]) / w.sum()
    b = np.empty(len(x), int)
    b[order] = np.minimum((cum * n_bins - 1e-12).astype(int), n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        if m.sum() == 0:
            continue
        rows.append({"bin": k, "x_median": float(np.median(x[m])), "x_lo": float(x[m].min()), "x_hi": float(x[m].max()),
                     "obs_rate": float(np.average(y[m], weights=w[m])), "pred_rate": float(np.average(p[m], weights=w[m])),
                     "n": int(m.sum()), "n_eff": kish_effective_n(w[m])})
    return pd.DataFrame(rows)


def partial_dependence(f, Xte, names, feat, w, grid):
    """Weighted mean prediction when `feat` is set to each grid value for everyone (age_sq tied to age)."""
    j = names.index(feat)
    out = []
    for g in grid:
        Xg = Xte.copy()
        Xg[:, j] = g
        if feat == "age" and "age_sq" in names:
            Xg[:, names.index("age_sq")] = (g - 50.0) ** 2 / 100.0
        out.append(float(np.average(f.predict(Xg), weights=w)))
    return np.array(out)


def _weighted_quantiles(x, w, qs):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return np.interp(qs, cw, x[o])


def subgroups(te, p, y, w):
    """Performance by age class and income quintile (weighted quintiles within the test wave/implicate)."""
    agecl = pd.cut(te["AGE"], [0, 34, 44, 54, 64, 74, 200], labels=["<35", "35-44", "45-54", "55-64", "65-74", "75+"])
    cuts = _weighted_quantiles(te["INCOME"].to_numpy(float), w, [0.2, 0.4, 0.6, 0.8])
    incq = np.searchsorted(cuts, te["INCOME"].to_numpy(float), side="right") + 1
    rows = []
    for dim, lab in (("age", agecl.astype(str).to_numpy()), ("income_quintile", incq.astype(str))):
        for g in sorted(set(lab)):
            m = lab == g
            ok = y[m].min() != y[m].max()
            rows.append({"dimension": dim, "group": g, "families": int(m.sum()), "events": int(y[m].sum()),
                         "n_eff": kish_effective_n(w[m]), "prevalence": float(np.average(y[m], weights=w[m])),
                         "mean_pred": float(np.average(p[m], weights=w[m])),
                         "roc_auc": ev.roc_auc(y[m], p[m], w[m]) if ok else np.nan,
                         "pr_auc": ev.pr_auc(y[m], p[m], w[m]) if ok else np.nan,
                         "citl": ev.calibration_intercept_slope(y[m], p[m], w[m])["citl_intercept"] if ok else np.nan})
    return pd.DataFrame(rows)


def remove_bnpl(te: pd.DataFrame) -> pd.DataFrame:
    """2025 source variables with BNPL taken out of Fed debt/payment totals (Fed definitions otherwise)."""
    d = te.copy()
    bnpl = d["BNPL"].to_numpy(float)
    bnplpay = 0.33 * bnpl  # Fed macro: BNPLPAY = 0.33*BNPL (monthly)
    d["DEBT"] = d["DEBT"] - bnpl
    d["NETWORTH"] = d["NETWORTH"] + bnpl
    d["HDEBT"] = (d["DEBT"] > 0).astype(float)
    tpay_new = d["TPAY"] - bnplpay
    with np.errstate(divide="ignore", invalid="ignore"):
        d["PIRTOTAL"] = np.where(d["TPAY"] > 0, d["PIRTOTAL"] * tpay_new / d["TPAY"], d["PIRTOTAL"])
        d["PIRREV"] = np.where(d["REVPAY"] > 0, d["PIRREV"] * (d["REVPAY"] - bnplpay) / d["REVPAY"], d["PIRREV"])
    d["TPAY"] = tpay_new
    d["DEBT2INC"] = np.where((d["DEBT"] > 0) & (d["INCOME"] > 0), d["DEBT"] / d["INCOME"].where(d["INCOME"] > 0), 0.0)
    d["LEVRATIO"] = np.where((d["DEBT"] > 0) & (d["ASSET"] > 0), d["DEBT"] / d["ASSET"].where(d["ASSET"] > 0),
                             np.where((d["DEBT"] > 0) & (d["ASSET"] == 0), 1.0, 0.0))
    return d


# --------------------------------------------------------------------------- driver
def run_all(cfg, n_repeats: int = 5) -> None:
    meta = json.loads((ROOT / "results/run_metadata.json").read_text())
    tables = ROOT / "results/tables"
    expl = ROOT / "results/explain"
    expl.mkdir(parents=True, exist_ok=True)
    df, X = load(cfg)
    saved = pd.read_parquet(ROOT / "results/predictions/test_2025.parquet")
    groups = {f"[block] {b}": F.BLOCKS[b] for b in F.BLOCKS}
    imps = range(1, int(cfg["data"]["n_implicates"]) + 1)
    perm_rows, shap_rows, assoc_rows, pdp_rows, sub_rows, bnpl_rows, repro = [], [], [], [], [], [], []
    shap_store = {}
    rng = np.random.default_rng(int(cfg["seed"]))

    for outcome in F.OUTCOMES:
        for population in ("A_all", "B_indebted"):
            for model in ("logit", "xgb"):
                key = (outcome, population, "M4", model)
                hp = _hp(meta, key)
                for m in imps:
                    f, te, cols = fit_final(df, X, cfg, *key, hp, m)
                    Xte = X.loc[te.index, cols].to_numpy()
                    y = te[outcome].to_numpy()
                    w = te["WGT"].to_numpy()
                    p = f.predict(Xte)
                    s = saved[(saved.outcome == outcome) & (saved.population == population) & (saved.block == "M4")
                              & (saved.model == model) & (saved.implicate == m)]
                    repro.append(float(np.max(np.abs(np.sort(s["p"].to_numpy()) - np.sort(p)))))
                    tag = dict(outcome=outcome, population=population, model=model, implicate=m)

                    pi = permutation_importance(f, Xte, y, w, cols, groups, n_repeats, rng)
                    perm_rows.append(pi.assign(**tag))

                    sub_rows.append(subgroups(te, p, y, w).assign(**tag))

                    te2 = te.assign(_y=y)
                    for fname, raw in PDP_VARS.items():
                        assoc_rows.append(binned_association(te2, Xte, p, raw, w).assign(variable=raw, **tag))
                        if population == "A_all":
                            xs = X.loc[te.index, fname].to_numpy()
                            grid = _weighted_quantiles(xs, w, np.linspace(0.02, 0.98, 25))
                            grid = np.unique(grid)
                            pd_vals = partial_dependence(f, Xte, cols, fname, w, grid)
                            pdp_rows.append(pd.DataFrame({"feature": fname, "variable": raw, "grid_transformed": grid,
                                                          "mean_pred": pd_vals, **tag}))

                    # BNPL sensitivity
                    te_nb = remove_bnpl(te)
                    X_nb = F.build_features(te_nb)[cols].to_numpy()
                    p_nb = f.predict(X_nb)
                    for variant, pp in (("as_built", p), ("bnpl_removed", p_nb)):
                        met = ev.all_metrics(y, pp, w)
                        bnpl_rows.append({**tag, "variant": variant, **met})

                    if model == "xgb":
                        import xgboost as xgb
                        contrib = f.obj.get_booster().predict(xgb.DMatrix(Xte, feature_names=None), pred_contribs=True)
                        mabs = np.average(np.abs(contrib[:, :-1]), axis=0, weights=w)
                        shap_rows.append(pd.DataFrame({"feature": cols, "mean_abs_shap": mabs, **tag}))
                        if m == 1:
                            shap_store[f"{outcome}|{population}"] = {"contrib": contrib[:, :-1], "X": Xte, "w": w,
                                                                    "features": cols}
                print(f"[explain] {'/'.join(key)} done", flush=True)

    print(f"[explain] max |p_refit - p_saved| = {max(repro):.2e}", flush=True)
    perm = pd.concat(perm_rows)
    perm.to_csv(expl / "permutation_by_implicate.csv", index=False)
    shap_df = pd.concat(shap_rows)
    shap_df.to_csv(expl / "shap_by_implicate.csv", index=False)
    pd.concat(assoc_rows).to_csv(expl / "binned_association_by_implicate.csv", index=False)
    pd.concat(pdp_rows).to_csv(expl / "partial_dependence_by_implicate.csv", index=False)
    np.savez_compressed(expl / "shap_implicate1.npz", **{
        f"{k}|{part}": (np.array(v[part]) if part != "features" else np.array(v[part], dtype=object))
        for k, v in shap_store.items() for part in ("contrib", "X", "w", "features")})

    # ---- summary tables (mean, min, max over implicates)
    def agg(d, by, vals):
        g = d.groupby(by)[vals].agg(["mean", "min", "max"])
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        return g.reset_index()

    sub = agg(pd.concat(sub_rows), ["outcome", "population", "model", "dimension", "group"],
              ["families", "events", "n_eff", "prevalence", "mean_pred", "roc_auc", "pr_auc", "citl"])
    sub["unstable_small_cell"] = sub["events_mean"] < 50
    sub.to_csv(tables / "subgroup_performance.csv", index=False)
    agg(pd.DataFrame(bnpl_rows), ["outcome", "population", "model", "variant"],
        ["roc_auc", "pr_auc", "brier", "log_loss", "citl", "cal_slope"]).to_csv(tables / "sensitivity_bnpl.csv", index=False)

    # ---- feature_importance.csv: one long table across methods
    fi = []
    p_feat = agg(perm[perm.kind == "feature"], ["outcome", "population", "model", "feature"], ["auc_drop", "logloss_increase"])
    for metric in ("auc_drop", "logloss_increase"):
        fi.append(p_feat[["outcome", "population", "model", "feature", f"{metric}_mean", f"{metric}_min", f"{metric}_max"]]
                  .set_axis(["outcome", "population", "model", "feature", "value_mean", "value_min", "value_max"], axis=1)
                  .assign(method=f"permutation_{metric}_2025"))
    s = agg(shap_df, ["outcome", "population", "model", "feature"], ["mean_abs_shap"])
    fi.append(s.set_axis(["outcome", "population", "model", "feature", "value_mean", "value_min", "value_max"], axis=1)
              .assign(method="mean_abs_treeshap_logodds_2025"))
    coefs = pd.read_csv(tables / "logit_coefficients.csv")
    c = coefs[coefs.block == "M4"]
    fi.append(c[["outcome", "population", "model", "feature", "coef_standardized_mean", "coef_standardized_min",
                 "coef_standardized_max"]]
              .set_axis(["outcome", "population", "model", "feature", "value_mean", "value_min", "value_max"], axis=1)
              .assign(method="logit_standardized_coefficient"))
    fi = pd.concat(fi, ignore_index=True)
    fi = fi[fi.population.isin(["A_all", "B_indebted"])]
    fi.to_csv(tables / "feature_importance.csv", index=False)
    agg(perm[perm.kind == "group"], ["outcome", "population", "model", "feature"], ["auc_drop", "logloss_increase"]) \
        .rename(columns={"feature": "block"}).to_csv(tables / "block_permutation_importance.csv", index=False)


if __name__ == "__main__":
    run_all(load_config(ROOT / "config.yaml"))
