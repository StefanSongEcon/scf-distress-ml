"""Build Fed summary variables from the full public SCF data, and check them.

``build_wave`` runs the ported Fed macro (see ``sasport``) on one wave and
returns one row per family-implicate with the macro's variables (upper case),
plus ``year``, ``family_id`` and ``implicate``. Dollar values are in the base
year given by ``cpibase`` (CPI-U-RS x 100; 4789 = the macro's 2025 value).

``compare_to_extract`` checks the result against the Fed summary extract,
variable by variable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import sasport, schema

CPIBASE_2025 = 4789

# Percentile/category variables the macro computes after the main DATA step
# (not needed for modelling, so not ported).
POST_STEP_VARS = {"NWCAT", "INCCAT", "ASSETCAT", "NINCCAT", "NINC2CAT", "NWPCTLECAT",
                  "INCPCTLECAT", "NINCPCTLECAT", "INCQRTCAT", "NINCQRTCAT"}


def build_wave(raw_dir: str | Path, year: int, cpibase: int = CPIBASE_2025,
               extra_raw: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (frame, info). `extra_raw` keeps additional raw X-variables (lower case)."""
    program = sasport.compile_year(year, cpibase)
    names = {n.lower() for n in sasport.referenced_names(program[1])}
    path = schema.wave_file(raw_dir, year)
    available = set(schema.wave_columns(path))
    extra = [c.lower() for c in (extra_raw or [])]
    need = sorted((names & available) | set(extra) | {"y1", "yy1"})
    raw = schema.read_wave(raw_dir, year, need)
    raw_nans = raw.columns[raw.isna().any()].tolist()
    ids = raw[["year", "family_id", "implicate"]]
    res = sasport.run_bullit(raw.drop(columns=["year", "family_id", "implicate"]), year, cpibase, program)
    if len(res.data) != len(raw):
        raise ValueError(f"{year}: macro subsetting IF dropped {len(raw) - len(res.data)} records")
    out = pd.concat([ids.reset_index(drop=True), res.data], axis=1)
    info = {"year": year, "n_statements": res.n_statements, "n_raw_columns": len(need),
            "uninitialized": sorted(res.uninitialized), "raw_columns_with_nan": raw_nans}
    return out, info


def read_extract(extract_dir: str | Path, year: int) -> pd.DataFrame:
    """Read a Fed summary extract CSV. SAS special missings exported as letters
    (e.g. 'B' for .B) become NaN."""
    df = pd.read_csv(Path(extract_dir) / str(year) / f"SCFP{year}.csv", low_memory=False)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].astype(str).str.strip()
            df[c] = pd.to_numeric(s.where(~s.str.fullmatch(r"[A-Z_]?|nan")), errors="raise")
    return df


def compare_to_extract(built: pd.DataFrame, extract: pd.DataFrame, rtol: float = 1e-6,
                       atol: float = 1e-4) -> pd.DataFrame:
    """Per-variable agreement between the Python build and the Fed extract (matched on Y1)."""
    m = built.merge(extract, on="Y1", how="outer", suffixes=("_py", "_fed"), indicator=True)
    unmatched = int((m["_merge"] != "both").sum())
    m = m[m["_merge"] == "both"]
    rows = []
    for c in extract.columns:
        if c in ("Y1", "YY1"):
            continue
        if c in POST_STEP_VARS:
            rows.append({"variable": c, "status": "not_ported", "n": len(m)})
            continue
        if f"{c}_py" not in m:
            rows.append({"variable": c, "status": "missing_in_build", "n": len(m)})
            continue
        x = m[f"{c}_py"].to_numpy(float)
        z = m[f"{c}_fed"].to_numpy(float)
        ok = np.isclose(x, z, rtol=rtol, atol=atol) | (np.isnan(x) & np.isnan(z))
        bad = ~ok
        rows.append({"variable": c, "status": "match" if ok.all() else "mismatch", "n": len(m),
                     "n_mismatch": int(bad.sum()), "families_mismatch": int(m.loc[bad, "YY1_fed"].nunique()),
                     "max_abs_diff": float(np.nanmax(np.abs(x[bad] - z[bad]))) if bad.any() else 0.0,
                     "unmatched_records": unmatched})
    return pd.DataFrame(rows)
