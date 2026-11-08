"""Deployable two-years-ahead delinquency risk model (SCF 2007-2009 panel).

* Inputs: the seven questions of the cross-sectional risk score (selected on 2010-2019 data,
  so the panel evaluation is free of selection bias) plus past delinquency ("late on a loan
  payment in the past 12 months").
* Training data: 2007 answers -> delinquency reported in the 2009 re-interview; panel dollars
  converted from 2009 to 2025 dollars so the model accepts today's amounts.
* Evaluation: family-grouped 5-fold cross-validation (identical folds for every model and
  implicate), compared with all candidate inputs and with the 46-feature research model.
* Calibration: the intercept is set so that the average predicted two-year risk across 2025
  households (scored with their 2025 answers) equals the 2025 delinquency rate, i.e. the level is
  anchored to current conditions. Missing answers take 2025 survey-weighted feature means.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import deploy as D
from . import evaluate as ev
from . import features as F
from . import models as M
from . import panel_model as PM
from . import variance as V

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/predictive_score_v1.json"
TABLES = ROOT / "results/deploy"
CPI_2009_TO_2025 = 4789 / 3171  # Fed macro CPI-U-RS bases (September 2025 / September 2009)
SEED = PM.SEED
PRED_INPUTS = ["age", "liquid_assets", "total_debt", "home_value", "income_unusually_low", "education_debt",
               "education", "recently_late"]
EXTRA_DESC = {"recently_late": ("LATE", "Late or missed a loan payment in the past 12 months", "yes/no")}


def panel_inputs() -> pd.DataFrame:
    """2007 answers (as deployment inputs, 2025 dollars) with 2009 outcomes, one row per family-implicate."""
    b07, _ = PM._prepare_year(2007)
    b09, _ = PM._prepare_year(2009)
    assert (b07["Y1"].to_numpy() == b09["Y1"].to_numpy()).all()
    x = D.scf_to_inputs(b07)
    for k, (_, _, unit) in D.INPUTS.items():
        if unit.startswith("dollars"):
            x[k] = x[k] * CPI_2009_TO_2025
    x["recently_late"] = b07["LATE"].astype(float)
    meta = pd.DataFrame({"family_id": b09["family_id"], "implicate": b09["implicate"], "Y1": b09["Y1"],
                         "WGT": b09["WGT"], "LATE": b09["LATE"], "LATE60": b09["LATE60"]})
    return pd.concat([meta, x], axis=1)


def cross_section_2025() -> pd.DataFrame:
    df = F.load_processed(ROOT / "data/processed", [2025])
    df, _ = F.make_labels(df)
    x = D.scf_to_inputs(df)
    x["recently_late"] = df["LATE"].astype(float)
    return pd.concat([df[["family_id", "implicate", "WGT", "LATE", "LATE60"]], x], axis=1)


def cv_predictions(d: pd.DataFrame, inputs: list, outcome: str, C: float, folds: dict) -> pd.DataFrame:
    fold = d["family_id"].map(folds).to_numpy()
    out = []
    for k in range(PM.N_OUTER):
        for m in range(1, 6):
            tr = d[(fold != k) & (d.implicate == m).to_numpy()]
            te = d[(fold == k) & (d.implicate == m).to_numpy()]
            f = M.fit_logit(D.build_features(tr, inputs, homeowner=False).to_numpy(), tr[outcome].to_numpy(),
                            tr.WGT.to_numpy() / tr.WGT.mean(), C=C)
            out.append(pd.DataFrame({"implicate": m, "family_id": te.family_id.to_numpy(), "y": te[outcome].to_numpy(),
                                     "p": f.predict(D.build_features(te, inputs, homeowner=False).to_numpy()), "wgt": te.WGT.to_numpy()}))
    return pd.concat(out).sort_values(["implicate", "family_id"]).reset_index(drop=True)


def _metrics(pr):
    rows = [ev.all_metrics(g.y.to_numpy(), g.p.to_numpy(), g.wgt.to_numpy()) for _, g in pr.groupby("implicate")]
    return pd.DataFrame(rows).mean().to_dict()


def run() -> dict:
    TABLES.mkdir(parents=True, exist_ok=True)
    d = panel_inputs()
    folds = PM.assign_folds(d["family_id"].to_numpy(), PM.N_OUTER, SEED)
    R1 = V.load_replicate_file(ROOT / "data/external/panel2009/replicate_weights/p09p_rw1.dta")

    # choose C by CV log loss for the deployed input set (cheap; reported, not tuned on any test set)
    c_ll = {C: _metrics(cv_predictions(d, PRED_INPUTS, "LATE", C, folds))["log_loss"] for C in (0.01, 0.1, 1.0)}
    C = min(c_ll, key=c_ll.get)

    rows, preds = [], {}
    sets = {"deployed_8_inputs": PRED_INPUTS, "deployed_without_past_delinquency": PRED_INPUTS[:-1],
            "all_18_inputs": list(D.INPUTS) + ["recently_late"]}
    for oc in F.OUTCOMES:
        for name, inp in sets.items():
            pr = cv_predictions(d, inp, oc, C, folds)
            preds[(oc, name)] = pr
            ci = {r["metric"]: r for r in V.cell_metric_se(pr.assign(outcome=oc, population="A", block=name, model="logit"),
                                                           R1, (oc, "A", name, "logit"))}
            rows.append({"outcome": oc, "inputs": name, "n_inputs": len(inp), **_metrics(pr),
                         "roc_auc_ci_lo": ci["roc_auc"]["ci_lo"], "roc_auc_ci_hi": ci["roc_auc"]["ci_hi"]})
    res = pd.DataFrame(rows)
    ref = pd.read_csv(ROOT / "results/panel/tables/cv_performance.csv")
    ref = ref[(ref.population == "A_all") & (ref.fset == "P_M4LAG") & (ref.model == "logit")]
    res = pd.concat([res, pd.DataFrame({"outcome": ref.outcome, "inputs": "research_model_46_features_plus_lag",
                                        "n_inputs": 48, "roc_auc": ref.roc_auc_mean, "pr_auc": ref.pr_auc_mean})])
    res.to_csv(TABLES / "predictive_cv_performance.csv", index=False)

    # final fit on the full panel, pooled over implicates
    cs = cross_section_2025()
    f25 = D.build_features(cs, PRED_INPUTS, homeowner=False)
    w25 = cs.WGT.to_numpy()
    means = {c: float(np.average(f25[c], weights=w25)) for c in f25.columns}
    model = {"homeowner_indicator": False, "name": "SCF two-years-ahead household delinquency risk", "version": "1.0", "created": str(date.today()),
             "horizon": "a 12-month period about two years after the answers are given",
             "inputs": {k: dict(zip(("scf_variable", "description", "unit"), (D.INPUTS | EXTRA_DESC)[k]))
                        for k in PRED_INPUTS},
             "feature_means_2025": means, "outcomes": {},
             "training": {"data": "SCF 2007-2009 panel (3,857 families, 5 implicates pooled): 2007 answers -> 2009 "
                                  "delinquency; dollars converted to 2025 dollars",
                          "model": f"L2 logistic regression, C={C}",
                          "inputs": "seven inputs selected on 2010-2019 cross-sections + past delinquency",
                          "calibration": "intercept set so mean predicted risk over 2025 households equals the 2025 "
                                         "delinquency rate"},
             "cv_performance": res.to_dict(orient="records")}
    for oc, label in (("LATE", "any_late_payment_2y"), ("LATE60", "late_60_days_2y")):
        coefs, ints, names = [], [], None
        for m in range(1, 6):
            g = d[d.implicate == m]
            X = D.build_features(g, PRED_INPUTS, homeowner=False)
            names = list(X.columns)
            f = M.fit_logit(X.to_numpy(), g[oc].to_numpy(), g.WGT.to_numpy() / g.WGT.mean(), C=C)
            mu, sd, clf = f.obj
            b = clf.coef_[0] / sd
            coefs.append(b)
            ints.append(float(clf.intercept_[0] - np.sum(b * mu)))
        coef, b0 = np.mean(coefs, axis=0), float(np.mean(ints))
        X25 = f25[names].to_numpy()
        target = float(np.average(cs[oc], weights=w25))
        lo, hi = -5.0, 5.0
        for _ in range(60):
            mid = (lo + hi) / 2
            pm = np.average(1 / (1 + np.exp(-(X25 @ coef + b0 + mid))), weights=w25)
            lo, hi = (mid, hi) if pm < target else (lo, mid)
        shift = (lo + hi) / 2
        p25 = 1 / (1 + np.exp(-(X25 @ coef + b0 + shift)))
        order = np.argsort(p25)
        cw = np.cumsum(w25[order]) / w25.sum()
        model["outcomes"][label] = {
            "scf_outcome": oc, "features": names, "coefficients": dict(zip(names, coef.tolist())),
            "intercept": b0 + shift, "intercept_panel": b0, "recalibration_shift": shift, "rate_2025": target,
            "rate_panel_2009": float(np.average(d[oc], weights=d.WGT)),
            "percentiles_2025": np.interp(np.linspace(0, 1, 101), cw, p25[order]).tolist()}
    MODEL_PATH.write_text(json.dumps(model, indent=2))
    print(res.round(3).to_string(index=False))
    print(f"[predict-deploy] C={C}; model written to {MODEL_PATH.relative_to(ROOT)}")
    return model


if __name__ == "__main__":
    run()
