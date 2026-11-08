"""Uncertainty, interpretation and figures for the 2007-2009 panel models.

* Confidence intervals: SCF panel replicate weights (999) + implicates, Fed codebook
  formula, applied to the out-of-fold predictions held fixed (``variance.py``).
* Permutation importance: the cross-validation models are refit (same folds,
  hyperparameters and seeds) and features are permuted on each held-out fold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import features as F
from . import models as M
from . import panel_model as PM
from . import variance as V

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results/panel/tables"
FIG = ROOT / "results/panel/figures"
RW = ROOT / "data/external/panel2009/replicate_weights/p09p_rw1.dta"


def confidence_intervals(jobs: int = 6) -> None:
    from joblib import Parallel, delayed

    pr = pd.read_parquet(PM.OUT_PRED / "oof_predictions.parquet").rename(columns={"fset": "block"})
    pr = pr.sort_values(["outcome", "population", "block", "model", "implicate", "family_id"]).reset_index(drop=True)
    R1 = V.load_replicate_file(RW)
    cells = sorted({tuple(r) for r in pr[["outcome", "population", "block", "model"]].drop_duplicates().to_numpy()
                    if r[3] != "constant"})
    diffs = []
    for o in F.OUTCOMES:
        for p in ("A_all", "B_indebted07", "C_onset"):
            for m in ("logit", "xgb"):
                diffs += [((o, p, "P_M4", m), (o, p, "P_M1", m)), ((o, p, "P_M4", m), (o, p, "P_M3", m)),
                          ((o, p, "R_M4", m), (o, p, "P_M4", m))]
                if p != "C_onset":
                    diffs += [((o, p, "P_M4LAG", m), (o, p, "P_M4", m)), ((o, p, "P_M4LAG", m), (o, p, "P_LAG", "logit"))]
            diffs.append(((o, p, "P_M4", "xgb"), (o, p, "P_M4", "logit")))

    def job(k, other):
        lab = dict(zip(["outcome", "population", "fset", "model"], k))
        if other is not None:
            lab["minus"] = f"{other[2]} {other[3]}"
        return [{**lab, **r} for r in V.cell_metric_se(pr, R1, k, other)]

    out = Parallel(n_jobs=jobs)(delayed(job)(k, None) for k in cells)
    pd.DataFrame([r for rs in out for r in rs]).to_csv(TAB / "cv_metrics_ci.csv", index=False)
    out = Parallel(n_jobs=jobs)(delayed(job)(k, o) for k, o in diffs)
    d = pd.DataFrame([r for rs in out for r in rs])
    d["excludes_zero"] = (d.ci_lo > 0) | (d.ci_hi < 0)
    d.to_csv(TAB / "cv_differences_ci.csv", index=False)


def permutation_importance(n_repeats: int = 3) -> None:
    df = pd.read_parquet(PM.PROC)
    meta = json.loads((ROOT / "results/panel/run_metadata.json").read_text())
    outer = PM.assign_folds(df["family_id"].to_numpy(), PM.N_OUTER, PM.SEED)
    rng = np.random.default_rng(PM.SEED)
    groups = {f"[block] {b}": [f"f07_{c}" for c in F.BLOCKS[b]] for b in F.BLOCKS}
    groups["[block] LAG"] = list(PM.LAG_FEATURES)
    rows = []
    for oc in F.OUTCOMES:
        for pop in ("A_all", "B_indebted07"):
            for model in ("logit", "xgb"):
                key = (oc, pop, "P_M4LAG", model)
                chosen = meta["chosen_hyperparameters"]["/".join(key)]
                cols = PM.feature_columns("P_M4LAG")
                sub = df[PM.population_mask(df, pop)]
                fold = sub["family_id"].map(outer).to_numpy()
                for k in range(PM.N_OUTER):
                    hp = {kk: v for kk, v in chosen[k].items() if kk != "fold"}
                    for m in range(1, 6):
                        a = sub[(fold != k) & (sub["implicate"] == m).to_numpy()]
                        b = sub[(fold == k) & (sub["implicate"] == m).to_numpy()]
                        f = M.FITTERS[model](a[cols].to_numpy(), a[oc].to_numpy(), a["WGT"].to_numpy() / a["WGT"].mean(),
                                             **hp, seed=PM.SEED + m, n_jobs=2)
                        X, y, w = b[cols].to_numpy(), b[oc].to_numpy(), b["WGT"].to_numpy()
                        base = ev.roc_auc(y, f.predict(X), w)
                        items = [(c, [i]) for i, c in enumerate(cols)] + [(g, [cols.index(c) for c in cs]) for g, cs in groups.items()]
                        for name, idx in items:
                            drops = []
                            for _ in range(n_repeats):
                                Xp = X.copy()
                                Xp[:, idx] = X[rng.permutation(len(X))][:, idx]
                                drops.append(base - ev.roc_auc(y, f.predict(Xp), w))
                            rows.append({"outcome": oc, "population": pop, "model": model, "fold": k, "implicate": m,
                                         "feature": name, "auc_drop": float(np.mean(drops))})
                print(f"[panel-importance] {'/'.join(key)} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(TAB / "permutation_by_fold_implicate.csv", index=False)
    s = d.groupby(["outcome", "population", "model", "feature"])["auc_drop"].agg(["mean", "min", "max"]).reset_index()
    s.to_csv(TAB / "permutation_importance.csv", index=False)


def figures() -> None:
    from . import figures as FG
    import matplotlib.pyplot as plt

    FG._style()
    FIG.mkdir(parents=True, exist_ok=True)
    ci = pd.read_csv(TAB / "cv_metrics_ci.csv")
    ci = ci[ci.metric == "roc_auc"]
    sets = [("P_M1", "2007 demographics"), ("P_M4", "2007 balance sheet"), ("P_LAG", "2007 delinquency only"),
            ("P_M4LAG", "2007 balance sheet\n+ 2007 delinquency"), ("R_M4", "2009 balance sheet\n(same-year benchmark)")]
    pops = [("A_all", "All families", FG.BLUE), ("B_indebted07", "Indebted in 2007", FG.ORANGE),
            ("C_onset", "Not late in 2007 (onset)", FG.AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8), sharey=True)
    for ax, oc in zip(axes, F.OUTCOMES):
        for j, (pop, plab, col) in enumerate(pops):
            for i, (fs, _) in enumerate(sets):
                mdl = "logit"
                r = ci[(ci.outcome == oc) & (ci.population == pop) & (ci.fset == fs) & (ci.model == mdl)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = len(sets) - 1 - i + (1 - j) * 0.22
                ax.hlines(y, r.ci_lo, r.ci_hi, color=col, lw=1.5)
                ax.scatter([r.estimate], [y], color=col, s=28, zorder=3, label=plab if i == 1 else None)
        ax.set_yticks(range(len(sets)), [s[1] for s in sets][::-1], fontsize=8)
        ax.set_xlabel("ROC AUC (cross-validated, 95% CI)")
        ax.set_title("Any late payment in 2009" if oc == "LATE" else "60+ days late in 2009")
        ax.axvline(0.5, color=FG.MUTED, lw=0.8, ls=":")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0.55, 0.92)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.55, -0.07))
    fig.text(0.0, -0.12, "SCF 2007-2009 panel, logistic regression; family-grouped 5-fold cross-validation with nested tuning; "
             "CIs from panel replicate weights + implicates.", fontsize=7, color=FG.MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "figP1_prospective_auc.png", bbox_inches="tight")
    fig.savefig(FIG / "figP1_prospective_auc.pdf", bbox_inches="tight")
    plt.close(fig)

    pi = pd.read_csv(TAB / "permutation_importance.csv")
    blocks = [("[block] M1", "M1 Demographics"), ("[block] M2", "M2 Income"), ("[block] M3", "M3 Wealth/liquidity"),
              ("[block] M4", "M4 Debt/housing"), ("[block] LAG", "2007 delinquency")]
    colors = list(FG.BLOCK_COLORS.values()) + [FG.ORANGE]
    rows = [(pop, mdl) for pop in ("A_all", "B_indebted07") for mdl in ("logit", "xgb")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, oc in zip(axes, F.OUTCOMES):
        y0 = np.arange(len(rows))[::-1]
        h = 0.15
        for k, (bk, bl) in enumerate(blocks):
            vals = [pi[(pi.outcome == oc) & (pi.population == p) & (pi.model == m) & (pi.feature == bk)]["mean"].iloc[0]
                    for p, m in rows]
            ax.barh(y0 + (2 - k) * h, vals, height=h - 0.02, color=colors[k], label=bl)
        ax.set_yticks(y0, [f"{'All families' if p == 'A_all' else 'Indebted 2007'}\n{FG.MODEL_LABEL[m]}" for p, m in rows],
                      fontsize=8)
        ax.set_xlabel("AUC drop when block is permuted")
        ax.set_title("Any late payment in 2009" if oc == "LATE" else "60+ days late in 2009")
        ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.55, -0.08))
    fig.text(0.0, -0.14, "Prospective model (2007 balance sheet + 2007 delinquency); held-out folds; mean over 5 folds x 5 "
             "implicates x 3 repeats.", fontsize=7, color=FG.MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "figP2_prospective_block_importance.png", bbox_inches="tight")
    fig.savefig(FIG / "figP2_prospective_block_importance.pdf", bbox_inches="tight")
    plt.close(fig)


def run_all(jobs: int = 6) -> None:
    confidence_intervals(jobs)
    print("[panel-analysis] CIs done", flush=True)
    permutation_importance()
    figures()
    print("[panel-analysis] done", flush=True)


if __name__ == "__main__":
    run_all()
