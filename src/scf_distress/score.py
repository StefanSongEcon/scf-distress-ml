"""Score households with the deployed SCF delinquency risk model.

Python
------
>>> from scf_distress.score import score
>>> score({"age": 34, "liquid_assets": 800, "total_debt": 42000, "education_debt": 25000})

Command line
------------
    python -m scf_distress.score --age 34 --liquid-assets 800 --total-debt 42000 --education-debt 25000
    python -m scf_distress.score --json household.json          # one household (object) or many (list)
    python -m scf_distress.score --csv households.csv --out scores.csv

Any input may be omitted; omitted inputs are replaced by the 2025 survey-weighted sample mean of the
model features they feed (i.e. an "average household" on that dimension), and are listed in the output.

What the probability means: the share of U.S. households with this profile that reported a late or
missed loan payment (or one 60+ days late) in the previous 12 months, at 2025 conditions (SCF).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .deploy import INPUTS, MODEL_PATH, build_features

DOLLAR_INPUTS = {k for k, v in INPUTS.items() if v[2].startswith("dollars")}
EDUCATION_LABELS = {"no_hs": 1, "less_than_high_school": 1, "high_school": 2, "hs": 2, "ged": 2,
                    "some_college": 3, "college": 4, "bachelor": 4, "graduate": 4}
BANDS = ((50, "Low"), (80, "Moderate"), (95, "High"), (101, "Very high"))
OUTCOME_LABELS = {"any_late_payment_12m": "Any late or missed loan payment (past 12 months)",
                  "late_60_days_12m": "Payment 60+ days late (past 12 months)",
                  "any_late_payment_2y": "Late or missed loan payment, about two years ahead",
                  "late_60_days_2y": "Payment 60+ days late, about two years ahead"}


@lru_cache(maxsize=4)
def load_model(path: str | None = None) -> dict:
    return json.loads(Path(path or MODEL_PATH).read_text())


def _as_bool(v) -> float:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "y", "yes", "true", "t"):
            return 1.0
        if s in ("0", "n", "no", "false", "f"):
            return 0.0
        raise ValueError(f"expected yes/no, got {v!r}")
    return 1.0 if bool(v) else 0.0


def clean_inputs(raw: dict, model: dict) -> dict:
    """Validate and normalize one household's inputs; drop empty values (treated as missing)."""
    allowed = set(model["inputs"])
    valid = set(INPUTS) | allowed
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"unknown inputs: {sorted(unknown)}; valid: {sorted(valid)}")
    out = {}
    for k, v in raw.items():
        if v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip() == ""):
            continue
        if k not in allowed:
            continue  # accepted but not used by this (parsimonious) model
        if k == "education":
            v = EDUCATION_LABELS.get(str(v).strip().lower(), v)
            v = int(float(v))
            if v not in (1, 2, 3, 4):
                raise ValueError("education must be 1-4 or one of " + ", ".join(sorted(EDUCATION_LABELS)))
        elif k in ("income_unusually_low", "married", "recently_late"):
            v = _as_bool(v)
        else:
            v = float(v)
            if k in DOLLAR_INPUTS and v < 0:
                raise ValueError(f"{k} cannot be negative")
            if k == "age" and not 18 <= v <= 100:
                raise ValueError("age must be between 18 and 100")
        out[k] = v
    if "education_debt" in out and "total_debt" in out and out["education_debt"] > out["total_debt"]:
        raise ValueError("education_debt is part of total_debt and cannot exceed it")
    return out


def _feature_matrix(rows: list[dict], model: dict, outcome: str) -> tuple[np.ndarray, list[list[str]]]:
    """Features for each household; features fed only by missing inputs take the 2025 mean."""
    names = model["outcomes"][outcome]["features"]
    means = model["feature_means_2025"]
    X = np.empty((len(rows), len(names)))
    imputed = []
    for i, r in enumerate(rows):
        present = [k for k in model["inputs"] if k in r]
        f = build_features(pd.DataFrame([r]), present) if present else pd.DataFrame(index=[0])
        X[i] = [float(f[c].iloc[0]) if c in f.columns else means[c] for c in names]
        imputed.append([k for k in model["inputs"] if k not in r])
    return X, imputed


def _percentile(p: float, table: list) -> float:
    q = np.asarray(table)
    return float(np.interp(p, q, np.linspace(0, 100, len(q))))


def _band(pct: float) -> str:
    return next(lbl for cut, lbl in BANDS if pct < cut)


@lru_cache(maxsize=4)
def _feature_owner(inputs: tuple) -> dict:
    """Which input each feature comes from."""
    owner = {}
    probe = pd.DataFrame([{k: 1.0 for k in inputs}])
    for k in inputs:
        for c in build_features(probe, [k]).columns:
            owner[c] = k
    return owner


def score(household: dict, model_path: str | None = None, top_drivers: int = 3) -> dict:
    """Score one household. Returns probabilities, 2025 percentiles, risk bands and main drivers."""
    model = load_model(model_path)
    r = clean_inputs(household, model)
    out = {"inputs_used": r, "imputed_inputs": [], "scores": {}, "drivers": []}
    for oc, spec in model["outcomes"].items():
        X, imputed = _feature_matrix([r], model, oc)
        coef = np.array([spec["coefficients"][c] for c in spec["features"]])
        p = float(1 / (1 + np.exp(-(X[0] @ coef + spec["intercept"]))))
        pct = _percentile(p, spec["percentiles_2025"])
        out["scores"][oc] = {"label": OUTCOME_LABELS.get(oc, oc), "probability": round(p, 4),
                             "percentile_2025": round(pct, 1), "risk_band": _band(pct),
                             "us_average_2025": round(spec["rate_2025"], 4)}
        out["imputed_inputs"] = imputed[0]
        if oc in ("any_late_payment_12m", "any_late_payment_2y"):
            owner = _feature_owner(tuple(model["inputs"]))
            means = model["feature_means_2025"]
            contrib = {}
            for c, x in zip(spec["features"], X[0]):
                contrib[owner[c]] = contrib.get(owner[c], 0.0) + spec["coefficients"][c] * (x - means[c])
            ranked = sorted(((k, v) for k, v in contrib.items() if k in r), key=lambda kv: -abs(kv[1]))
            out["drivers"] = [{"input": k, "direction": "raises risk" if v > 0 else "lowers risk",
                               "log_odds_vs_average": round(v, 3)} for k, v in ranked[:top_drivers]]
    return out


def score_frame(df: pd.DataFrame, model_path: str | None = None) -> pd.DataFrame:
    """Score many households (one per row); returns probabilities, percentiles and bands."""
    model = load_model(model_path)
    rows = [clean_inputs({k: v for k, v in row.items() if k in INPUTS}, model) for row in df.to_dict(orient="records")]
    res = pd.DataFrame(index=df.index)
    for oc, spec in model["outcomes"].items():
        X, imputed = _feature_matrix(rows, model, oc)
        coef = np.array([spec["coefficients"][c] for c in spec["features"]])
        p = 1 / (1 + np.exp(-(X @ coef + spec["intercept"])))
        pct = np.array([_percentile(v, spec["percentiles_2025"]) for v in p])
        res[f"p_{oc}"] = p.round(4)
        res[f"pct_{oc}"] = pct.round(1)
        res[f"band_{oc}"] = [_band(v) for v in pct]
    res["n_imputed_inputs"] = [len(i) for i in imputed]
    return res


def _format(result: dict) -> str:
    lines = []
    for oc, s in result["scores"].items():
        lines.append(f"{s['label']}:")
        lines.append(f"  probability {s['probability']:.1%}  (U.S. average {s['us_average_2025']:.1%})")
        lines.append(f"  percentile  {s['percentile_2025']:.0f}  ->  {s['risk_band']} risk")
    if result["drivers"]:
        lines.append("Main drivers vs. an average household:")
        for d in result["drivers"]:
            lines.append(f"  {d['input']:22s} {d['direction']} ({d['log_odds_vs_average']:+.2f} log-odds)")
    if result["imputed_inputs"]:
        lines.append("Not provided (set to the 2025 average): " + ", ".join(result["imputed_inputs"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Household delinquency risk score (SCF model).")
    for k, (_, desc, unit) in INPUTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, help=f"{desc} [{unit}]")
    ap.add_argument("--recently-late", dest="recently_late",
                    help="Late or missed a loan payment in the past 12 months [yes/no] (predictive model)")
    ap.add_argument("--json", help="JSON file with one household (object) or many (list); '-' for stdin")
    ap.add_argument("--csv", help="CSV file, one household per row, columns named like the inputs")
    ap.add_argument("--out", help="write batch scores to this CSV")
    ap.add_argument("--format", choices=("text", "json"), default="text")
    ap.add_argument("--model", help="path to a model JSON (default: models/distress_score_v1.json)")
    a = ap.parse_args(argv)
    if a.csv:
        res = score_frame(pd.read_csv(a.csv), a.model)
        res.to_csv(a.out, index=False) if a.out else print(res.to_string())
        return
    if a.json:
        data = json.load(sys.stdin if a.json == "-" else open(a.json))
        if isinstance(data, list):
            res = score_frame(pd.DataFrame(data), a.model)
            res.to_csv(a.out, index=False) if a.out else print(res.to_string())
            return
        household = data
    else:
        household = {k: getattr(a, k) for k in list(INPUTS) + ["recently_late"] if getattr(a, k) is not None}
    result = score(household, a.model)
    print(json.dumps(result, indent=2) if a.format == "json" else _format(result))


if __name__ == "__main__":
    main()
