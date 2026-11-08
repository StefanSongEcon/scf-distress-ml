"""A parsimonious, deployable household delinquency risk score.

Pipeline
--------
1. **Inputs a household can report** (``INPUTS``), each mapped to an SCF variable.
   Features are built from these inputs by ``build_features`` — the *same* function
   is used for training (SCF values) and for scoring new households, so training and
   deployment cannot drift apart.
2. **Forward selection** of inputs on the 2010-2019 validation folds only
   (train 2010-13 -> validate 2016; train 2010-16 -> validate 2019), adding the input
   that most reduces validation log loss, averaged over folds and implicates.
3. **Evaluation** of the selected model trained on 2010-2019 on the 2025 wave
   (2025 played no role in selection).
4. **Deployment fit** on all five waves (2010-2025), coefficients averaged over the five
   implicates, intercept recalibrated so that predicted delinquency matches the 2025
   weighted rate. Missing inputs are imputed with 2025 survey-weighted feature means.
5. Export to ``models/distress_score_v1.json``; scoring lives in ``score.py``.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import features as F
from . import models as M
from .weights import normalize_training_weights

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/distress_score_v1.json"
TABLES = ROOT / "results/deploy"
WAVES = [2010, 2013, 2016, 2019, 2025]
FOLDS = [((2010, 2013), 2016), ((2010, 2013, 2016), 2019)]
C_GRID = (0.01, 0.1, 1.0)

# name: (SCF source, description, unit)
INPUTS = {
    "age": ("AGE", "Age of household head", "years"),
    "education": ("EDCL", "Education: 1 no HS diploma, 2 HS diploma/GED, 3 some college, 4 college degree", "category"),
    "married": ("MARRIED", "Married or living with partner", "yes/no"),
    "children": ("KIDS", "Number of children in household", "count"),
    "income": ("INCOME", "Annual household income (before taxes)", "dollars"),
    "income_unusually_low": ("X7650", "Income this year unusually low compared with normal", "yes/no"),
    "liquid_assets": ("LIQ", "Checking, savings, money-market and prepaid balances", "dollars"),
    "retirement_assets": ("RETQLIQ", "Retirement accounts (401k, IRA, ...)", "dollars"),
    "financial_assets": ("FIN", "All financial assets (incl. liquid, retirement, stocks, funds)", "dollars"),
    "total_assets": ("ASSET", "All assets (financial + home, vehicles, business, other)", "dollars"),
    "total_debt": ("DEBT", "All debt (mortgages, cards, loans, other)", "dollars"),
    "credit_card_balance": ("CCBAL", "Credit card balance after last payment", "dollars"),
    "monthly_debt_payments": ("TPAY", "Total required monthly debt payments", "dollars/month"),
    "home_value": ("HOUSES", "Value of primary residence (0 if renting)", "dollars"),
    "mortgage_debt": ("MRTHEL", "Mortgage and home-equity debt on primary residence", "dollars"),
    "education_debt": ("EDN_INST", "Student loan balance", "dollars"),
    "vehicle_debt": ("VEH_INST", "Vehicle loan balance", "dollars"),
}
CORE = ("age",)  # always included (needed for life-cycle terms)


def scf_to_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Map SCF summary variables to the deployment inputs."""
    out = pd.DataFrame(index=df.index)
    for name, (src, _, _) in INPUTS.items():
        out[name] = df[src].astype(float)
    out["married"] = (df["MARRIED"] == 1).astype(float)
    out["income_unusually_low"] = (df["X7650"] == 2).astype(float)
    return out


def _k(x):
    """Dollars -> asinh(dollars / 1000): smooth through zero, log-like for large values, monotone."""
    return np.arcsinh(x.astype(float) / 1000.0)


def build_features(x: pd.DataFrame, inputs: tuple | list, homeowner: bool = True) -> pd.DataFrame:
    """Features computable from the given inputs (columns of x). Same code for training and scoring.

    Deployment design: each input enters through one smooth monotone transform (plus a has-debt
    indicator, because households without loans cannot be late), with no offsetting indicators or
    overlapping ratios, so the score moves in one direction as any single input increases.
    """
    have = set(inputs)
    f = pd.DataFrame(index=x.index)
    if "age" in have:
        f["age"] = x["age"].astype(float)
        f["age_sq"] = (x["age"].astype(float) - 50.0) ** 2 / 100.0
    if "education" in have:
        for k, nm in ((2, "edu_highschool"), (3, "edu_somecollege"), (4, "edu_college")):
            f[nm] = (x["education"] == k).astype(float)
    if "married" in have:
        f["married"] = x["married"].astype(float)
    if "children" in have:
        f["children"] = x["children"].astype(float).clip(0, 6)
    if "income" in have:
        f["income_k"] = _k(x["income"])
    if "income_unusually_low" in have:
        f["income_unusually_low"] = x["income_unusually_low"].astype(float)
    if "recently_late" in have:  # predictive model only (past delinquency is a legitimate predictor of future)
        f["recently_late"] = x["recently_late"].astype(float)
    for name, feat in (("liquid_assets", "liquid_k"), ("retirement_assets", "retirement_k"),
                       ("financial_assets", "financial_k"), ("total_assets", "assets_k"),
                       ("credit_card_balance", "credit_card_k"), ("monthly_debt_payments", "payments_k"),
                       ("home_value", "home_value_k"), ("mortgage_debt", "mortgage_k"),
                       ("education_debt", "student_loans_k"), ("vehicle_debt", "vehicle_loans_k")):
        if name in have:
            f[feat] = _k(x[name])
    if "total_debt" in have:
        f["has_debt"] = (x["total_debt"] > 0).astype(float)
        f["debt_k"] = _k(x["total_debt"])
    if "home_value" in have and homeowner:
        f["homeowner"] = (x["home_value"] > 0).astype(float)
    return f.astype(float)


# --------------------------------------------------------------------------- data
def load() -> pd.DataFrame:
    df = F.load_processed(ROOT / "data/processed", WAVES)
    df, _ = F.make_labels(df)
    x = scf_to_inputs(df)
    return pd.concat([df[["year", "family_id", "implicate", "WGT", "LATE", "LATE60"]], x], axis=1)


def _fit_eval(d: pd.DataFrame, inputs, outcome, train_years, eval_year, C, implicates=range(1, 6)):
    lls, aucs = [], []
    for m in implicates:
        g = d[d.implicate == m]
        tr, te = g[g.year.isin(train_years)], g[g.year == eval_year]
        Xtr, Xte = build_features(tr, inputs), build_features(te, inputs)
        w = normalize_training_weights(pd.Series(tr.WGT.to_numpy()), pd.Series(tr.year.to_numpy()), "equal_wave").to_numpy()
        f = M.fit_logit(Xtr.to_numpy(), tr[outcome].to_numpy(), w, C=C)
        p = f.predict(Xte.to_numpy())
        lls.append(ev.log_loss(te[outcome].to_numpy(), p, te.WGT.to_numpy()))
        aucs.append(ev.roc_auc(te[outcome].to_numpy(), p, te.WGT.to_numpy()))
    return float(np.mean(lls)), float(np.mean(aucs))


def validation_score(d, inputs, outcome="LATE", C=0.1):
    res = [_fit_eval(d, inputs, outcome, tr, ev_, C) for tr, ev_ in FOLDS]
    return float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res]))


def forward_selection(d: pd.DataFrame, outcome="LATE", jobs=6) -> pd.DataFrame:
    from joblib import Parallel, delayed

    chosen = list(CORE)
    path = [{"step": 0, "added": "+".join(chosen), "inputs": list(chosen), **dict(zip(("val_logloss", "val_auc"),
                                                                                     validation_score(d, chosen, outcome)))}]
    remaining = [k for k in INPUTS if k not in chosen]
    while remaining:
        scores = Parallel(n_jobs=jobs)(delayed(validation_score)(d, chosen + [c], outcome) for c in remaining)
        best = int(np.argmin([s[0] for s in scores]))
        chosen.append(remaining.pop(best))
        path.append({"step": len(path), "added": chosen[-1], "inputs": list(chosen),
                     "val_logloss": scores[best][0], "val_auc": scores[best][1]})
        print(f"[deploy] step {len(path) - 1}: +{chosen[-1]:24s} val AUC {scores[best][1]:.4f}  logloss {scores[best][0]:.4f}",
              flush=True)
    return pd.DataFrame(path)


def choose_parsimonious(path: pd.DataFrame, tol_auc: float = 0.005) -> list:
    """Smallest input set whose validation AUC is within `tol_auc` of the best set on the path."""
    best = path.val_auc.max()
    ok = path[path.val_auc >= best - tol_auc]
    return list(ok.iloc[0]["inputs"])


# --------------------------------------------------------------------------- fit + export
def _pooled_fit(d, inputs, outcome, years, C):
    """Fit per implicate; return coefficients on the raw feature scale averaged over implicates."""
    coefs, ints, names = [], [], None
    for m in range(1, 6):
        g = d[(d.implicate == m) & d.year.isin(years)]
        X = build_features(g, inputs)
        names = list(X.columns)
        w = normalize_training_weights(pd.Series(g.WGT.to_numpy()), pd.Series(g.year.to_numpy()), "equal_wave").to_numpy()
        f = M.fit_logit(X.to_numpy(), g[outcome].to_numpy(), w, C=C)
        mu, sd, clf = f.obj
        b = clf.coef_[0] / sd
        coefs.append(b)
        ints.append(float(clf.intercept_[0] - np.sum(b * mu)))
    return names, np.mean(coefs, axis=0), float(np.mean(ints))


def _predict(X: np.ndarray, coef, intercept):
    return 1 / (1 + np.exp(-(X @ coef + intercept)))


def run(jobs: int = 6) -> dict:
    t0 = time.time()
    TABLES.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    d = load()

    # 1. selection on validation folds (2025 untouched)
    path = forward_selection(d, "LATE", jobs)
    path.assign(inputs=path.inputs.map(lambda v: ", ".join(v))).to_csv(TABLES / "forward_selection_path.csv", index=False)
    chosen = choose_parsimonious(path)
    # tune C on validation for the chosen set
    c_scores = {C: validation_score(d, chosen, "LATE", C)[0] for C in C_GRID}
    C = min(c_scores, key=c_scores.get)
    print(f"[deploy] chosen {len(chosen)} inputs: {chosen}; C={C}", flush=True)

    # 2. evaluation on 2025: models trained on 2010-2019 (per implicate, like the paper)
    rows = []
    for oc in F.OUTCOMES:
        for label, inp in (("parsimonious", chosen), ("all_inputs", list(INPUTS))):
            for m in range(1, 6):
                g = d[d.implicate == m]
                tr, te = g[g.year.isin(WAVES[:-1])], g[g.year == 2025]
                w = normalize_training_weights(pd.Series(tr.WGT.to_numpy()), pd.Series(tr.year.to_numpy()), "equal_wave").to_numpy()
                f = M.fit_logit(build_features(tr, inp).to_numpy(), tr[oc].to_numpy(), w, C=C)
                p = f.predict(build_features(te, inp).to_numpy())
                rows.append({"outcome": oc, "model": label, "n_inputs": len(inp), "implicate": m,
                             **ev.all_metrics(te[oc].to_numpy(), p, te.WGT.to_numpy())})
    test = pd.DataFrame(rows)
    summ = test.groupby(["outcome", "model", "n_inputs"])[["roc_auc", "pr_auc", "brier", "recall_top10", "citl"]].agg(["mean", "min", "max"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(TABLES / "test_2025_parsimonious_vs_all.csv", index=False)

    # 3. deployment fit: all waves, pooled over implicates, recalibrated to 2025
    d25 = d[d.year == 2025]
    feats25 = build_features(d25, chosen)
    w25 = d25.WGT.to_numpy()
    means = {c: float(np.average(feats25[c], weights=w25)) for c in feats25.columns}
    model = {"name": "SCF household delinquency risk score", "version": "1.0", "created": str(date.today()),
             "inputs": {k: {"scf_variable": INPUTS[k][0], "description": INPUTS[k][1], "unit": INPUTS[k][2]}
                        for k in chosen},
             "feature_means_2025": means, "outcomes": {}, "training": {
                 "data": "SCF full public data 2010, 2013, 2016, 2019, 2025 (all families); 5 implicates pooled",
                 "weights": "survey weights, each wave equal total weight", "model": f"L2 logistic regression, C={C}",
                 "selection": "forward selection of inputs on 2010-2019 validation folds (log loss); smallest set within "
                              "0.005 validation AUC of the best", "dollars": "2025 dollars"},
             "evaluation_2025": summ.to_dict(orient="records")}
    for oc, label in (("LATE", "any_late_payment_12m"), ("LATE60", "late_60_days_12m")):
        names, coef, b0 = _pooled_fit(d, chosen, oc, WAVES, C)
        # recalibrate the intercept to the 2025 weighted rate (pooled over implicates)
        X25 = feats25[names].to_numpy()
        target = float(np.average(d25[oc], weights=w25))
        lo, hi = -5.0, 5.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if np.average(_predict(X25, coef, b0 + mid), weights=w25) < target:
                lo = mid
            else:
                hi = mid
        shift = (lo + hi) / 2
        p25 = _predict(X25, coef, b0 + shift)
        qs = np.linspace(0, 1, 101)
        order = np.argsort(p25)
        cw = np.cumsum(w25[order]) / w25.sum()
        quant = np.interp(qs, cw, p25[order]).tolist()
        model["outcomes"][label] = {
            "scf_outcome": oc, "features": names, "coefficients": dict(zip(names, coef.tolist())),
            "intercept": b0 + shift, "intercept_before_recalibration": b0, "recalibration_shift": shift,
            "rate_2025": target, "percentiles_2025": quant}
    MODEL_PATH.write_text(json.dumps(model, indent=2))
    print(f"[deploy] model written to {MODEL_PATH.relative_to(ROOT)} in {time.time() - t0:.0f}s", flush=True)
    return model


if __name__ == "__main__":
    run()
