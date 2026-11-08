"""Publication figures (PNG 300 dpi + PDF) from saved results; nothing is re-estimated.

Colors follow the reference data-viz palette: categorical slots 1-3 (blue, orange,
aqua; documented to pass all-pairs CVD checks) for identities, a single-hue blue
ramp (steps 250-700) for the ordered feature blocks M1-M4, and neutral inks for text.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results/figures"
TAB = ROOT / "results/tables"
EXPL = ROOT / "results/explain"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLOCK_COLORS = {"M1": "#86b6ef", "M2": "#3987e5", "M3": "#1c5cab", "M4": "#0d366b"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
POP_LABEL = {"A_all": "All families", "B_indebted": "Indebted families", "B2_payments_due": "Payments due"}
MODEL_LABEL = {"logit": "Logit (L2)", "xgb": "XGBoost", "constant": "Constant"}
BLOCK_LABEL = {"M1": "M1 Demographics", "M2": "M2 + Income", "M3": "M3 + Wealth/liquidity", "M4": "M4 + Debt/housing"}
FEATURE_LABEL = {
    "asinh_liq": "Liquid assets", "levratio": "Debt / assets", "unsecured_debt_share": "Unsecured debt share",
    "hdebt": "Has debt", "asinh_debt": "Total debt", "asinh_networth": "Net worth", "liq_to_debt": "Liquid assets / debt",
    "liq_to_income": "Liquid assets / income", "asinh_install": "Installment debt", "asinh_houses": "Home value",
    "asinh_homeeq": "Home equity", "pircons": "Consumer payments / income", "pirtotal": "Payments / income",
    "pirrev": "Revolving payments / income", "debt2inc": "Debt / income", "age": "Age", "age_sq": "Age (squared)",
    "kids": "Number of children", "asinh_fin": "Financial assets", "asinh_asset": "Total assets",
    "asinh_veh_inst": "Vehicle loans", "asinh_edn_inst": "Education loans", "asinh_ccbal": "Credit card balance",
    "asinh_mrthel": "Mortgage + HELOC", "asinh_tpay": "Monthly payments", "income_unusually_low": "Income unusually low",
    "income_unusually_high": "Income unusually high", "asinh_income": "Income", "asinh_wageinc": "Wage income",
    "asinh_norminc": "Normal income", "negative_networth": "Negative net worth", "mortgage_debt_share": "Mortgage share of debt",
    "education_debt_share": "Education share of debt", "homeowner": "Homeowner", "married": "Married",
    "asinh_retqliq": "Retirement accounts", "no_liquid_assets": "No liquid assets", "income_zero": "Zero income",
    "income_negative": "Negative income", "edcl_college": "College degree", "edcl_somecollege": "Some college",
    "edcl_highschool": "High school", "fam_single_kids": "Single with children", "fam_couple_kids": "Couple with children",
    "fam_single_55plus": "Single, 55+", "fam_single_under55": "Single, under 55",
}


def _style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "lines.linewidth": 2.0, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _preds():
    return pd.read_parquet(ROOT / "results/predictions/test_2025.parquet")


# --------------------------------------------------------------------------- 1 prevalence
def fig_prevalence():
    d = pd.read_csv(TAB / "outcome_prevalence.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, pop in zip(axes, ["A_all", "B_indebted"]):
        for oc, col in (("LATE", BLUE), ("LATE60", ORANGE)):
            g = d[(d.population == pop) & (d.outcome == oc)].sort_values("year")
            y = 100 * g.weighted_rate.to_numpy()
            yrs = g.year.to_numpy()
            ax.plot(yrs[:-1], y[:-1], color=col, marker="o", ms=5)
            ax.plot(yrs[-2:], y[-2:], color=col, ls=(0, (2, 2)), lw=1.5)
            ax.plot(yrs[-1:], y[-1:], color=col, marker="o", ms=6, mfc="white", mew=2)
            ax.vlines(yrs, 100 * g.rate_min_imp, 100 * g.rate_max_imp, color=col, lw=1)
            ax.annotate("Any late payment" if oc == "LATE" else "60+ days late", (yrs[0], y[0]),
                        xytext=(0, 7), textcoords="offset points", color=INK2, fontsize=8)
        ax.set_title(POP_LABEL[pop])
        ax.set_xticks([2010, 2013, 2016, 2019, 2025])
        ax.axvspan(2020.5, 2023.5, color=GRID, alpha=0.35, lw=0)
        ax.text(2022, 1.5, "2022\nnot used", ha="center", fontsize=7, color=MUTED)
    axes[0].set_ylabel("Weighted share of families (%)")
    axes[0].set_ylim(0, 30)
    fig.text(0.0, -0.04, "Open marker: 2025 test wave. Vertical ticks: range across the 5 implicates. "
             "Past-12-month delinquency; SCF full public data.", fontsize=7, color=MUTED)
    _save(fig, "fig1_prevalence_by_year")


# --------------------------------------------------------------------------- 2 ROC / PR
def _mean_curve(groups, kind):
    grid = np.linspace(0, 1, 201)
    ys = []
    for _, g in groups:
        if kind == "roc":
            fpr, tpr, _ = roc_curve(g.y, g.p, sample_weight=g.wgt)
            ys.append(np.interp(grid, fpr, tpr))
        else:
            prec, rec, _ = precision_recall_curve(g.y, g.p, sample_weight=g.wgt)
            ys.append(np.interp(grid, rec[::-1], prec[::-1]))
    return grid, np.mean(ys, axis=0)


def fig_roc_pr(pr=None):
    pr = _preds() if pr is None else pr
    perf = pd.read_csv(TAB / "model_performance.csv")
    perf = perf[perf.role == "locked_test"]
    specs = [("M4", "xgb", BLUE), ("M4", "logit", ORANGE), ("M1", "logit", AQUA)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    for r, oc in enumerate(["LATE", "LATE60"]):
        for c, kind in enumerate(["roc", "pr"]):
            ax = axes[r, c]
            for blk, mdl, col in specs:
                g = pr[(pr.outcome == oc) & (pr.population == "A_all") & (pr.block == blk) & (pr.model == mdl)]
                x, y = _mean_curve(g.groupby("implicate"), kind)
                row = perf[(perf.outcome == oc) & (perf.population == "A_all") & (perf.block == blk) & (perf.model == mdl)].iloc[0]
                val = row.roc_auc_mean if kind == "roc" else row.pr_auc_mean
                ax.plot(x, y, color=col, label=f"{blk} {MODEL_LABEL[mdl]} ({'AUC' if kind == 'roc' else 'AP'} {val:.2f})")
            if kind == "roc":
                ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
                ax.set_xlabel("False positive rate")
                ax.set_ylabel("True positive rate")
            else:
                prev = np.average(g.y, weights=g.wgt)
                ax.axhline(prev, color=MUTED, lw=1, ls=":")
                ax.text(0.01, prev + 0.015, f"prevalence {prev:.2f}", ha="left", fontsize=7, color=MUTED)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_ylim(0, 1)
            ax.set_title(f"{'Any late payment' if oc == 'LATE' else '60+ days late'}: {'ROC' if kind == 'roc' else 'Precision-recall'}")
            ax.legend(loc="lower right" if kind == "roc" else "upper right", fontsize=7.5)
    fig.text(0.0, -0.02, "2025 locked test, all families; survey-weighted; curves averaged over the 5 implicates.",
             fontsize=7, color=MUTED)
    fig.tight_layout()
    _save(fig, "fig2_roc_pr_2025")


# --------------------------------------------------------------------------- 3 calibration + histograms
def _cal_bins(g, n=10):
    order = np.argsort(g.p.to_numpy(), kind="stable")
    w = g.wgt.to_numpy()[order]
    cum = np.cumsum(w) / w.sum()
    b = np.minimum((cum * n - 1e-12).astype(int), n - 1)
    p = g.p.to_numpy()[order]
    y = g.y.to_numpy()[order]
    return pd.DataFrame({"b": b, "p": p * w, "y": y * w, "w": w}).groupby("b").sum().pipe(
        lambda t: pd.DataFrame({"pred": t.p / t.w, "obs": t.y / t.w}))


def fig_calibration(pr=None):
    pr = _preds() if pr is None else pr
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    for r, oc in enumerate(["LATE", "LATE60"]):
        ax = axes[r, 0]
        top = 0
        for mdl, col in (("xgb", BLUE), ("logit", ORANGE)):
            g = pr[(pr.outcome == oc) & (pr.population == "A_all") & (pr.block == "M4") & (pr.model == mdl)]
            tabs = [_cal_bins(gi) for _, gi in g.groupby("implicate")]
            t = sum(tabs) / len(tabs)
            ax.plot(100 * t.pred, 100 * t.obs, color=col, marker="o", ms=5, label=f"M4 {MODEL_LABEL[mdl]}")
            top = max(top, 100 * t.obs.max(), 100 * t.pred.max())
        ax.plot([0, top * 1.05], [0, top * 1.05], color=MUTED, lw=1, ls=":")
        ax.set_xlabel("Mean predicted (%)")
        ax.set_ylabel("Observed in 2025 (%)")
        ax.set_title(f"{'Any late payment' if oc == 'LATE' else '60+ days late'}: calibration")
        ax.legend(loc="upper left", fontsize=7.5)
        ax = axes[r, 1]
        g = pr[(pr.outcome == oc) & (pr.population == "A_all") & (pr.block == "M4") & (pr.model == "xgb")]
        bins = np.linspace(0, min(1, g.p.quantile(0.995) * 1.1), 40)
        for yv, col, lab in ((0, AQUA, "Not delinquent"), (1, ORANGE, "Delinquent")):
            gg = g[g.y == yv]
            ax.hist(gg.p, bins=bins, weights=gg.wgt / gg.wgt.sum(), histtype="step", lw=2, color=col, label=lab)
        ax.set_xlabel("Predicted probability (M4 XGBoost)")
        ax.set_ylabel("Share of weighted families")
        ax.set_title("Predicted-risk distribution")
        ax.legend(fontsize=7.5)
    fig.text(0.0, -0.02, "2025 locked test, all families. Calibration bins: weighted deciles of predicted risk, "
             "averaged over implicates. Points above the diagonal = under-prediction.", fontsize=7, color=MUTED)
    fig.tight_layout()
    _save(fig, "fig3_calibration_2025")


# --------------------------------------------------------------------------- 4 incremental value
def fig_incremental():
    perf = pd.read_csv(TAB / "model_performance.csv")
    t = perf[(perf.role == "locked_test")]
    blocks = ["M1", "M2", "M3", "M4"]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.4), sharex=True)
    for c, oc in enumerate(["LATE", "LATE60"]):
        for pop, col in (("A_all", BLUE), ("B_indebted", ORANGE)):
            const = t[(t.outcome == oc) & (t.population == pop) & (t.model == "constant") & (t.block == "M1")].iloc[0]
            for mdl, ls in (("xgb", "-"), ("logit", "--")):
                g = t[(t.outcome == oc) & (t.population == pop) & (t.model == mdl)].set_index("block").loc[blocks]
                x = np.arange(4)
                vals = [g.roc_auc_mean, g.pr_auc_mean / g.prevalence_mean, 1 - g.brier_mean / const.brier_mean]
                los = [g.roc_auc_min, g.pr_auc_min / g.prevalence_mean, 1 - g.brier_max / const.brier_mean]
                his = [g.roc_auc_max, g.pr_auc_max / g.prevalence_mean, 1 - g.brier_min / const.brier_mean]
                for r in range(3):
                    ax = axes[r, c]
                    ax.plot(x, vals[r], color=col, ls=ls, marker="o", ms=5,
                            label=f"{POP_LABEL[pop]}, {MODEL_LABEL[mdl]}")
                    ax.vlines(x, los[r], his[r], color=col, lw=1)
        axes[0, c].set_title("Any late payment" if oc == "LATE" else "60+ days late")
    axes[0, 0].set_ylabel("ROC AUC")
    axes[1, 0].set_ylabel("PR-AUC / prevalence")
    axes[2, 0].set_ylabel("Brier skill vs constant")
    for ax in axes[2]:
        ax.set_xticks(range(4), ["M1\nDemog.", "M2\n+Income", "M3\n+Wealth", "M4\n+Debt"])
    axes[0, 1].legend(fontsize=7, loc="lower right")
    fig.text(0.0, -0.02, "2025 locked test. Each block adds features to the previous one. Solid: XGBoost; dashed: logit. "
             "Ticks: range across implicates.", fontsize=7, color=MUTED)
    fig.tight_layout()
    _save(fig, "fig4_incremental_blocks_2025")


# --------------------------------------------------------------------------- 5 importance
def fig_importance():
    b = pd.read_csv(TAB / "block_permutation_importance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    blocks = ["M1", "M2", "M3", "M4"]
    rows = [(pop, mdl) for pop in ("A_all", "B_indebted") for mdl in ("logit", "xgb")]
    for ax, oc in zip(axes, ["LATE", "LATE60"]):
        y0 = np.arange(len(rows))[::-1]
        h = 0.18
        for k, blk in enumerate(blocks):
            vals = [b[(b.outcome == oc) & (b.population == p) & (b.model == m) & (b.block == f"[block] {blk}")]
                    .auc_drop_mean.iloc[0] for p, m in rows]
            ax.barh(y0 + (1.5 - k) * h, vals, height=h - 0.02, color=BLOCK_COLORS[blk], label=BLOCK_LABEL[blk])
        ax.set_yticks(y0, [f"{POP_LABEL[p]}\n{MODEL_LABEL[m]}" for p, m in rows])
        ax.set_xlabel("AUC drop when block is permuted")
        ax.set_title("Any late payment" if oc == "LATE" else "60+ days late")
        ax.grid(axis="y", visible=False)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.text(0.0, -0.04, "2025 locked test, M4 models; each block's own features permuted jointly; mean of 5 implicates x 5 repeats.",
             fontsize=7, color=MUTED)
    fig.tight_layout()
    _save(fig, "fig5a_block_importance_2025")

    fi = pd.read_csv(TAB / "feature_importance.csv")
    fi = fi[(fi.method == "permutation_auc_drop_2025") & (fi.model == "xgb")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
    for r, oc in enumerate(["LATE", "LATE60"]):
        for c, pop in enumerate(["A_all", "B_indebted"]):
            ax = axes[r, c]
            g = fi[(fi.outcome == oc) & (fi.population == pop)].nlargest(12, "value_mean")[::-1]
            y = np.arange(len(g))
            ax.barh(y, g.value_mean, height=0.6, color=BLUE)
            ax.hlines(y, g.value_min, g.value_max, color=INK2, lw=1)
            ax.set_yticks(y, [FEATURE_LABEL.get(f, f) for f in g.feature], fontsize=7.5)
            ax.set_title(f"{'Any late' if oc == 'LATE' else '60+ days late'} - {POP_LABEL[pop]}", fontsize=9)
            ax.set_xlabel("AUC drop (permutation)")
            ax.grid(axis="y", visible=False)
    fig.text(0.0, -0.02, "M4 XGBoost, 2025 locked test; single-feature permutation, which understates correlated features "
             "(see block importance). Lines: range across implicates.", fontsize=7, color=MUTED)
    fig.tight_layout()
    _save(fig, "fig5b_feature_importance_2025")

    z = np.load(EXPL / "shap_implicate1.npz", allow_pickle=True)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#3987e5", "#0d366b"])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), gridspec_kw={"wspace": 0.95})
    rng = np.random.default_rng(0)
    for ax, oc in zip(axes, ["LATE", "LATE60"]):
        k = f"{oc}|A_all"
        contrib, X, w, feats = z[f"{k}|contrib"], z[f"{k}|X"], z[f"{k}|w"], list(z[f"{k}|features"])
        imp = np.average(np.abs(contrib), axis=0, weights=w)
        top = np.argsort(imp)[::-1][:10][::-1]
        for row, j in enumerate(top):
            x = contrib[:, j]
            xv = X[:, j]
            pct = pd.Series(xv).rank(pct=True).to_numpy()
            sub = rng.choice(len(x), size=min(1500, len(x)), replace=False)
            jit = rng.normal(0, 0.11, len(sub))
            ax.scatter(x[sub], row + jit, c=pct[sub], cmap=ramp, s=5, lw=0, alpha=0.8, rasterized=True)
        ax.set_yticks(range(len(top)), [FEATURE_LABEL.get(feats[j], feats[j]) for j in top], fontsize=7.5)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("TreeSHAP contribution (log-odds)")
        ax.set_title("Any late payment" if oc == "LATE" else "60+ days late")
        ax.grid(axis="y", visible=False)
    sm = plt.cm.ScalarMappable(cmap=ramp)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Feature value (percentile)", color=INK2)
    fig.text(0.0, -0.02, "M4 XGBoost, all families, 2025, implicate 1 (1,500 sampled families per row). "
             "Associations within the model, not causal effects.", fontsize=7, color=MUTED)
    _save(fig, "fig5c_shap_summary_2025")


# --------------------------------------------------------------------------- 6 subgroups
def fig_subgroups():
    s = pd.read_csv(TAB / "subgroup_performance.csv")
    s = s[(s.population == "A_all") & (s.model == "xgb")]
    order = {"age": ["<35", "35-44", "45-54", "55-64", "65-74", "75+"], "income_quintile": ["1", "2", "3", "4", "5"]}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey="row")
    for c, dim in enumerate(["age", "income_quintile"]):
        for oc, col, off in (("LATE", BLUE, -0.12), ("LATE60", ORANGE, 0.12)):
            g = s[(s.dimension == dim) & (s.outcome == oc)].set_index("group").loc[order[dim]]
            x = np.arange(len(g)) + off
            st = g.unstable_small_cell.to_numpy()
            lab = "Any late payment" if oc == "LATE" else "60+ days late"
            ax = axes[0, c]
            ax.vlines(x, g.roc_auc_min, g.roc_auc_max, color=col, lw=1)
            ax.scatter(x[~st], g.roc_auc_mean[~st], color=col, s=30, zorder=3, label=lab)
            ax.scatter(x[st], g.roc_auc_mean[st], facecolor="white", edgecolor=col, lw=1.5, s=30, zorder=3)
            ax = axes[1, c]
            ax.scatter(x, 100 * g.prevalence_mean, color=col, s=30, zorder=3, label=f"{lab}: observed")
            ax.scatter(x, 100 * g.mean_pred_mean, color=col, marker="_", s=120, lw=2, zorder=3, label=f"{lab}: predicted")
            ax.vlines(x, 100 * g.mean_pred_mean, 100 * g.prevalence_mean, color=col, lw=1, alpha=0.6)
        for r in range(2):
            axes[r, c].set_xticks(range(len(order[dim])), order[dim] if dim == "age" else
                                  ["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"])
        axes[1, c].set_xlabel("Age of reference person" if dim == "age" else "Income quintile (2025, weighted)")
    axes[0, 0].set_ylabel("ROC AUC")
    axes[1, 0].set_ylabel("Delinquency rate (%)")
    axes[0, 0].set_title("Discrimination by subgroup")
    axes[1, 0].set_title("Observed vs predicted rate")
    axes[0, 1].legend(fontsize=7, loc="lower right")
    h, l = axes[1, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.075))
    fig.text(0.0, -0.11, "M4 XGBoost, all families, 2025 locked test. Hollow markers: fewer than 50 delinquent families per "
             "implicate (unstable). Lines: range across implicates.", fontsize=7, color=MUTED)
    fig.tight_layout()
    _save(fig, "fig6_subgroups_2025")


# --------------------------------------------------------------------------- 7 response
def fig_response():
    a = pd.read_csv(EXPL / "binned_association_by_implicate.csv")
    a = a[(a.population == "A_all") & (a.model == "xgb")]
    pdp = pd.read_csv(EXPL / "partial_dependence_by_implicate.csv")
    pdp = pdp[pdp.model == "xgb"]
    vars_ = [("AGE", "age", "Age", False), ("LIQ", "asinh_liq", "Liquid assets ($)", True),
             ("DEBT2INC", "debt2inc", "Debt / income", False), ("PIRTOTAL", "pirtotal", "Payments / income", False),
             ("NETWORTH", "asinh_networth", "Net worth ($)", True), ("CCBAL", "asinh_ccbal", "Credit card balance ($)", True)]
    for oc in ("LATE", "LATE60"):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharey=True)
        for ax, (raw, feat, lab, dollars) in zip(axes.flat, vars_):
            g = a[(a.outcome == oc) & (a.variable == raw)].groupby("bin")[["x_median", "obs_rate", "pred_rate"]].mean()
            q = pdp[(pdp.outcome == oc) & (pdp.feature == feat)].copy()
            q["k"] = q.groupby("implicate").cumcount()
            q = q.groupby("k")[["grid_transformed", "mean_pred"]].mean()
            xg = q.grid_transformed.to_numpy() if feat == "age" else np.sinh(q.grid_transformed.to_numpy())
            ax.plot(xg, 100 * q.mean_pred.to_numpy(), color=AQUA, lw=1.8, label="Partial dependence (model, synthetic)")
            ax.plot(g.x_median, 100 * g.obs_rate, color=ORANGE, marker="o", ms=4, lw=1.2, label="Observed (decile bins)")
            ax.plot(g.x_median, 100 * g.pred_rate, color=BLUE, marker="s", ms=4, lw=1.2, label="Predicted (decile bins)")
            if dollars:
                ax.set_xscale("symlog", linthresh=1000, linscale=0.5)
                ticks = [-1e5, 0, 1e4, 1e6] if raw == "NETWORTH" else [0, 1e3, 1e4, 1e5, 1e6]
                ax.set_xticks(ticks, [_dollar(t) for t in ticks], fontsize=7)
                ax.minorticks_off()
            elif raw in ("DEBT2INC", "PIRTOTAL"):
                ax.set_xscale("symlog", linthresh=0.1, linscale=0.5)
                ticks = [0, 0.1, 0.3, 1, 3] if raw == "DEBT2INC" else [0, 0.1, 0.3, 1]
                ax.set_xticks(ticks, [f"{t:g}" for t in ticks], fontsize=7)
                ax.minorticks_off()
            ax.set_title(lab, fontsize=9)
        axes[0, 0].set_ylabel("Delinquency (%)")
        axes[1, 0].set_ylabel("Delinquency (%)")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(("Any late payment" if oc == "LATE" else "60+ days late") + " - 2025, all families, M4 XGBoost",
                     x=0.01, ha="left", fontsize=10, fontweight="bold")
        fig.text(0.0, -0.1, "Observed/predicted: conditional associations by weighted deciles (x = bin median). Partial "
                 "dependence: one input set for everyone, other inputs unchanged (correlated inputs not adjusted).",
                 fontsize=7, color=MUTED)
        fig.tight_layout()
        _save(fig, f"fig7_response_{oc.lower()}_2025")


def _dollar(t):
    a = abs(t)
    lab = "0" if a == 0 else (f"{a / 1e6:g}M" if a >= 1e6 else f"{a / 1e3:g}k")
    return ("-" if t < 0 else "") + ("$" if a else "") + lab


def make_all():
    _style()
    pr = _preds()
    fig_prevalence()
    fig_roc_pr(pr)
    fig_calibration(pr)
    fig_incremental()
    fig_importance()
    fig_subgroups()
    fig_response()


if __name__ == "__main__":
    make_all()
