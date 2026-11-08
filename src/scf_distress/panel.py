"""SCF 2007-2009 panel: build 2007 and 2009 Fed summary variables for the same families.

The public panel file ``p09pi6.dta`` holds the 2007 interview as ``X`` variables and
the 2009 re-interview as ``P`` variables (same question numbers). The Fed's panel
macro (``docs/panel/fedstables.macro.txt``) points each variable reference
(``&Vnnnn``) at the ``X`` or ``P`` version depending on the year; it is run here
with the same SAS interpreter as the cross-sections and checked against the Fed
panel extract (``rscfp2009panel``), whose variables carry ``07``/``09`` suffixes.

Both years use the 2009 panel weight (``P42001``); dollar values are in 2009 dollars
by default (``cpibase=3171``, the macro's September-2009 CPI-U-RS value).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import sasport

ROOT = Path(__file__).resolve().parents[2]
PANEL_FILE = ROOT / "data/raw/p09pi6.dta"
PANEL_MACRO = ROOT / "docs/panel/fedstables.macro.txt"
PANEL_EXTRACT = ROOT / "data/external/panel2009/extract/rscfp2009panel.dta"
CPIBASE_2009 = 3171
# computed after the main DATA step (percentile categories; COMBO-step ratios)
POST_STEP = {"NWCAT", "INCCAT", "ASSETCAT", "HIPIR", "LEVERAGE"}


def build_panel_year(year: int, cpibase: int = CPIBASE_2009, panel_file: Path = PANEL_FILE,
                     extra_raw: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    program = sasport.compile_year(year, cpibase, macro_path=PANEL_MACRO)
    names = sasport.referenced_names(program[1])
    with pd.io.stata.StataReader(panel_file) as r:
        actual = list(r.variable_labels())
    lookup = {c.upper(): c for c in actual}
    need = sorted((names & set(lookup)) | {"Y1", "YY1"} | {c.upper() for c in (extra_raw or [])})
    raw = pd.read_stata(panel_file, columns=[lookup[c] for c in need], convert_categoricals=False)
    raw.columns = [c.upper() for c in raw.columns]
    yy1 = raw["YY1"].astype(np.int64)
    imp = raw["Y1"].astype(np.int64) - 10 * yy1
    if not imp.between(1, 5).all():
        raise ValueError("implicate outside 1..5")
    res = sasport.run_bullit(raw, year, cpibase, program)
    if len(res.data) != len(raw):
        raise ValueError(f"{year}: subsetting IF dropped {len(raw) - len(res.data)} records")
    out = pd.concat([pd.DataFrame({"year": year, "family_id": yy1.to_numpy(), "implicate": imp.to_numpy()}),
                     res.data.reset_index(drop=True)], axis=1)
    info = {"year": year, "n_statements": res.n_statements, "uninitialized": sorted(res.uninitialized),
            "raw_columns": len(need)}
    return out, info


def read_panel_extract(path: Path = PANEL_EXTRACT) -> pd.DataFrame:
    d = pd.read_stata(path, convert_categoricals=False)
    d.columns = [c.upper() for c in d.columns]
    return d


def compare_panel(built: pd.DataFrame, extract: pd.DataFrame, year: int, rtol=1e-6, atol=1e-4) -> pd.DataFrame:
    """Compare built variables V with extract column V07 / V09 (matched on Y1)."""
    suf = str(year)[2:]
    m = built.merge(extract, on="Y1", how="inner", suffixes=("", "_EXT"))
    rows = []
    for c in extract.columns:
        if not c.endswith(suf) or c in ("Y1", "YY1"):
            continue
        base = c[: -len(suf)]
        if base not in built.columns:  # extract drops underscores (VEHINST = VEH_INST)
            alt = {k.replace("_", ""): k for k in built.columns}.get(base)
            base = alt or base
        if base in POST_STEP:
            rows.append({"year": year, "variable": base, "extract_col": c, "status": "not_ported"})
            continue
        if base not in built.columns:
            rows.append({"year": year, "variable": base, "extract_col": c, "status": "missing_in_build"})
            continue
        x = m[base].to_numpy(float)
        z = m[c].to_numpy(float)
        ok = np.isclose(x, z, rtol=rtol, atol=atol) | (np.isnan(x) & np.isnan(z))
        rows.append({"year": year, "variable": base, "extract_col": c, "status": "match" if ok.all() else "mismatch",
                     "n": len(m), "n_mismatch": int((~ok).sum()),
                     "max_abs_diff": float(np.nanmax(np.abs(x[~ok] - z[~ok]))) if (~ok).any() else 0.0})
    return pd.DataFrame(rows)
