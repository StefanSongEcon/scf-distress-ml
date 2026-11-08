"""Locate and read full public SCF Stata files; identify families and implicates.

File convention: ``data/raw/p{yy}i6.dta`` (Fed naming), or the same file inside
``data/raw/<year>/``. Column names are lowercased on read (the 2010 file uses
upper case, later files lower case).

IDs in the public full data:
* ``yy1`` — public case (family) ID
* ``y1``  — implicate case ID, ``y1 = 10 * yy1 + implicate`` with implicate in 1..5
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = ["yy1", "y1"]


def wave_file(raw_dir: str | Path, year: int) -> Path:
    raw_dir = Path(raw_dir)
    name = f"p{year % 100:02d}i6.dta"
    for cand in (raw_dir / name, raw_dir / str(year) / name):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No full public SCF file for {year}: expected {raw_dir / name}")


def wave_columns(path: str | Path) -> list[str]:
    with pd.io.stata.StataReader(path) as r:
        return [c.lower() for c in r.variable_labels()]


def read_wave(raw_dir: str | Path, year: int, columns: list[str] | None = None) -> pd.DataFrame:
    """Read selected columns (case-insensitive) of one wave and attach IDs.

    Returned frame has lowercase X-variables plus ``year``, ``family_id``
    (= yy1) and ``implicate`` (= y1 - 10*yy1).
    """
    path = wave_file(raw_dir, year)
    with pd.io.stata.StataReader(path) as r:
        actual = list(r.variable_labels())
    lookup = {c.lower(): c for c in actual}
    want = list(dict.fromkeys(ID_COLS + [c.lower() for c in (columns or [])]))
    missing = [c for c in want if c not in lookup]
    if missing:
        raise KeyError(f"{year}: columns not in {path.name}: {missing}")
    df = pd.read_stata(path, columns=[lookup[c] for c in want], convert_categoricals=False)
    df.columns = [c.lower() for c in df.columns]
    # Cast before arithmetic: yy1 is stored as int16 in some files, so 10*yy1 overflows.
    yy1 = df["yy1"].astype(np.int64)
    y1 = df["y1"].astype(np.int64)
    df.insert(0, "year", year)
    df.insert(1, "family_id", yy1)
    df.insert(2, "implicate", y1 - 10 * yy1)
    if not df["implicate"].between(1, 5).all():
        raise ValueError(f"{year}: implicate derived from y1 - 10*yy1 outside 1..5")
    return df
