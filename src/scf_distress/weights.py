"""Survey-weight and multiple-imputation handling.

Conventions
-----------
* ``WGT = X42001 / 5`` (Fed macro). Within one implicate, weights sum to the
  population divided by 5; do not divide by 5 again. Uniform rescaling within an
  implicate does not change normalized metrics.
* A family-year is one household; its five implicates are not independent
  observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def check_implicates(df: pd.DataFrame, family_col: str = "family_id", implicate_col: str = "implicate",
                     wave_col: str = "year", n_implicates: int = 5) -> pd.DataFrame:
    """Return families whose implicate set is not exactly {1..n}. Empty frame means all good."""
    expected = set(range(1, n_implicates + 1))
    g = df.groupby([wave_col, family_col])[implicate_col].agg(lambda s: tuple(sorted(s)))
    bad = g[g.map(lambda t: set(t) != expected or len(t) != n_implicates)]
    return bad.rename("implicates").reset_index()


def normalize_training_weights(w: pd.Series, wave: pd.Series, scheme: str = "equal_wave") -> pd.Series:
    """Rescale survey weights for model fitting.

    ``equal_wave``: each wave gets the same total weight (so waves with larger
    populations do not dominate), then everything is scaled to mean 1.
    ``raw``: population weights as given, scaled to mean 1 (targets the pooled
    household population across waves).
    """
    w = pd.Series(w, dtype=float)
    if (w < 0).any() or w.isna().any():
        raise ValueError("Weights must be non-missing and non-negative")
    if scheme == "equal_wave":
        out = w / w.groupby(wave).transform("sum")
    elif scheme == "raw":
        out = w.copy()
    else:
        raise ValueError(f"Unknown weight scheme {scheme!r}")
    return out * (len(out) / out.sum())


def kish_effective_n(w) -> float:
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0


def rubin_combine(estimates, within_variances) -> dict:
    """Combine per-implicate estimates with Rubin's rules.

    Returns the pooled estimate, within variance W, between variance B, and
    total variance T = W + (1 + 1/m) B.
    """
    q = np.asarray(estimates, dtype=float)
    u = np.asarray(within_variances, dtype=float)
    m = len(q)
    if m < 2:
        raise ValueError("Need at least two implicates")
    qbar = q.mean()
    w_var = u.mean()
    b_var = q.var(ddof=1)
    return {"estimate": qbar, "within": w_var, "between": b_var,
            "total": w_var + (1 + 1 / m) * b_var, "m": m}
