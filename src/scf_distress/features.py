"""Labels, estimation samples and whitelisted predictors.

Every predictor is listed explicitly in ``BLOCKS``; nothing else reaches a model.
All dollar amounts are 2025 dollars from the ported Fed macro. Transforms are
fixed functions (no fitted parameters), so they cannot leak test information:

* dollar amounts -> ``asinh`` (sign-preserving, so negative net worth and home
  equity keep their sign; never set to zero);
* ratios -> ``asinh`` of the ratio, with denominators floored at $1,000 where the
  ratio is constructed here; the Fed's ``PIRTOTAL``/``PIRCONS``/``PIRREV`` keep the
  Fed's own income floor; ``DEBT2INC`` follows the Fed (0 without debt, 10 with debt
  but zero income) and ``income_zero`` flags zero income;
* categorical codes -> one-hot indicators (reference level dropped).

Note: the Fed sets ``INCOME = max(0, X5729)``; the public data record negative
income only as codes (-1, -9), so the sign is kept as the flag ``income_negative``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTCOMES = ("LATE", "LATE60")

# Raw/Fed columns needed to build features, labels and weights.
SOURCE_COLS = [
    "year", "family_id", "implicate", "WGT", "X3004", "X3005", "X5729", "X7650", "LATE", "LATE60",
    "AGE", "EDCL", "MARRIED", "KIDS", "FAMSTRUCT",
    "INCOME", "WAGEINC", "NORMINC",
    "ASSET", "FIN", "LIQ", "RETQLIQ",
    "NETWORTH", "HDEBT", "DEBT", "CCBAL", "MRTHEL", "INSTALL", "EDN_INST", "VEH_INST", "OTH_INST",
    "OTHLOC", "TPAY", "PIRTOTAL", "PIRCONS", "PIRREV", "DEBT2INC", "LEVRATIO",
    "HOUSECL", "HOUSES", "HOMEEQ",
]

_EDCL = {2: "edcl_highschool", 3: "edcl_somecollege", 4: "edcl_college"}       # ref 1 = no HS diploma
_FAMSTRUCT = {1: "fam_single_kids", 2: "fam_single_55plus", 3: "fam_single_under55",
              4: "fam_couple_kids"}                                             # ref 5 = couple, no kids

BLOCKS: dict[str, list[str]] = {
    "M1": ["age", "age_sq", "married", "kids", *_EDCL.values(), *_FAMSTRUCT.values()],
    "M2": ["asinh_income", "asinh_wageinc", "asinh_norminc", "income_zero", "income_negative",
           "income_unusually_high", "income_unusually_low"],
    # NETWORTH moved to M4: with ASSET in M3 it would reveal total debt (NETWORTH = ASSET - DEBT).
    "M3": ["asinh_asset", "asinh_fin", "asinh_liq", "asinh_retqliq", "no_liquid_assets",
           "liq_to_income"],
    "M4": ["asinh_networth", "negative_networth", "hdebt", "asinh_debt", "asinh_ccbal", "asinh_mrthel",
           "asinh_install", "asinh_edn_inst", "asinh_veh_inst", "asinh_tpay",
           "pirtotal", "pircons", "pirrev", "debt2inc", "levratio",
           "liq_to_debt", "unsecured_debt_share", "mortgage_debt_share", "education_debt_share",
           "homeowner", "asinh_houses", "asinh_homeeq"],
}

# Never allowed as predictors (outcome sources, IDs, weights, split identifiers, distress events).
BANNED = {
    "LATE", "LATE60", "X3004", "X3005", "Y1", "YY1", "family_id", "implicate", "year", "WGT", "X42001",
    "BNKRUPLAST5", "FORECLLAST5", "HPAYDAY", "TURNDOWN", "TURNFEAR", "FEARDENIAL", "EMERGBORR", "EMERGSAV",
    "EMERGPSTP", "EMERGCUT", "EMERGWORK",
}


def block_features(block: str) -> list[str]:
    """Cumulative feature list for a nested block (M2 = M1 + income, ...)."""
    order = list(BLOCKS)
    out: list[str] = []
    for b in order[: order.index(block) + 1]:
        out += BLOCKS[b]
    return out


def load_processed(processed_dir: str | Path, years) -> pd.DataFrame:
    frames = [pd.read_parquet(Path(processed_dir) / f"bullit_{y}.parquet", columns=SOURCE_COLS) for y in years]
    return pd.concat(frames, ignore_index=True)


def make_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """LATE = (X3004==5), LATE60 = (X3005==1) as in the Fed macro, with LATE60 forced <= LATE.

    The only violation in 2010-2025 is one 2016 family with X3005=1 but X3004=0
    (no payments due), which is set to LATE60=0.
    """
    out = df.copy()
    assert ((out["LATE"] == (out["X3004"] == 5)) & (out["LATE60"] == (out["X3005"] == 1))).all()
    viol = (out["LATE60"] == 1) & (out["LATE"] == 0)
    info = {"late60_without_late_records": int(viol.sum()),
            "late60_without_late_families": out.loc[viol, ["year", "family_id"]].drop_duplicates().values.tolist()}
    out.loc[viol, "LATE60"] = 0
    out["LATE"] = out["LATE"].astype(int)
    out["LATE60"] = out["LATE60"].astype(int)
    return out, info


def population_mask(df: pd.DataFrame, population: str) -> pd.Series:
    if population == "A_all":
        return pd.Series(True, index=df.index)
    if population == "B_indebted":
        return df["HDEBT"] == 1
    if population == "B2_payments_due":
        return df["X3004"] != 0
    raise ValueError(population)


def _ratio(num, den, floor=1000.0):
    return np.arcsinh(num / np.maximum(den, floor))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return all whitelisted features (columns of every block) for the rows of df."""
    f = pd.DataFrame(index=df.index)
    f["age"] = df["AGE"]
    f["age_sq"] = (df["AGE"] - 50.0) ** 2 / 100.0
    f["married"] = (df["MARRIED"] == 1).astype(float)
    f["kids"] = df["KIDS"]
    for code, name in _EDCL.items():
        f[name] = (df["EDCL"] == code).astype(float)
    for code, name in _FAMSTRUCT.items():
        f[name] = (df["FAMSTRUCT"] == code).astype(float)

    f["asinh_income"] = np.arcsinh(df["INCOME"])
    f["asinh_wageinc"] = np.arcsinh(df["WAGEINC"])
    f["asinh_norminc"] = np.arcsinh(df["NORMINC"])
    f["income_zero"] = (df["INCOME"] <= 0).astype(float)
    f["income_negative"] = (df["X5729"] < 0).astype(float)
    f["income_unusually_high"] = (df["X7650"] == 1).astype(float)  # X7650: 1 high, 2 low, 3 normal
    f["income_unusually_low"] = (df["X7650"] == 2).astype(float)

    for c in ("ASSET", "FIN", "LIQ", "RETQLIQ"):
        f[f"asinh_{c.lower()}"] = np.arcsinh(df[c])
    f["no_liquid_assets"] = (df["LIQ"] <= 0).astype(float)
    f["liq_to_income"] = _ratio(df["LIQ"], df["INCOME"])

    f["asinh_networth"] = np.arcsinh(df["NETWORTH"])
    f["negative_networth"] = (df["NETWORTH"] < 0).astype(float)
    f["hdebt"] = df["HDEBT"].astype(float)
    for c in ("DEBT", "CCBAL", "MRTHEL", "INSTALL", "EDN_INST", "VEH_INST", "TPAY", "HOUSES", "HOMEEQ"):
        f[f"asinh_{c.lower()}"] = np.arcsinh(df[c])
    for c in ("PIRTOTAL", "PIRCONS", "PIRREV", "DEBT2INC", "LEVRATIO"):
        f[c.lower()] = np.arcsinh(df[c])
    f["liq_to_debt"] = _ratio(df["LIQ"], df["DEBT"])
    debt = df["DEBT"].where(df["DEBT"] > 0)
    unsecured = df["CCBAL"] + df["OTHLOC"] + df["EDN_INST"] + df["OTH_INST"]
    f["unsecured_debt_share"] = (unsecured / debt).fillna(0.0).clip(0, 1)
    f["mortgage_debt_share"] = (df["MRTHEL"] / debt).fillna(0.0).clip(0, 1)
    f["education_debt_share"] = (df["EDN_INST"] / debt).fillna(0.0).clip(0, 1)
    f["homeowner"] = (df["HOUSECL"] == 1).astype(float)

    expected = block_features("M4")
    assert list(f.columns) == expected or set(f.columns) == set(expected), set(f.columns) ^ set(expected)
    if f[expected].isna().any().any():
        raise ValueError(f"NaN in features: {f.columns[f.isna().any()].tolist()}")
    return f[expected]
