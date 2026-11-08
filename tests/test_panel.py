from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scf_distress import features as F
from scf_distress import panel_model as PM

ROOT = Path(__file__).resolve().parents[1]


def test_prospective_sets_use_only_2007_information():
    for fs in ("P_M1", "P_M2", "P_M3", "P_M4", "P_LAG", "P_M4LAG"):
        cols = PM.feature_columns(fs)
        assert all(c.startswith("f07_") or c in PM.LAG_FEATURES for c in cols), fs
        assert not any(c.startswith("f09_") for c in cols)
    assert all(c.startswith("f09_") for c in PM.feature_columns("R_M4"))


def test_feature_sets_nested_and_whitelisted():
    assert PM.feature_columns("P_M4LAG") == PM.feature_columns("P_M4") + PM.LAG_FEATURES
    banned = {b.lower() for b in F.BANNED}
    for fs in ("P_M4LAG", "R_M4"):
        assert not {c.split("_", 1)[1] for c in PM.feature_columns(fs) if "_" in c} & banned


def test_folds_group_families_and_are_shared():
    fam = np.repeat(np.arange(100), 5)  # 5 implicates per family
    f = PM.assign_folds(fam, 5, 1)
    folds = pd.Series(fam).map(f)
    assert (pd.DataFrame({"fam": fam, "fold": folds}).groupby("fam").fold.nunique() == 1).all()
    assert PM.assign_folds(fam, 5, 1) == f  # deterministic, so identical across cells
    assert sorted(pd.Series(f).value_counts().tolist()) == [20] * 5


def test_fed_ratio_rules():
    d = pd.DataFrame({"DEBT": [0, 100, 100, 100], "ASSET": [10, 0, 50, 50], "INCOME": [5, 0, 0, 200]})
    out = PM._fed_ratios(d.astype(float))
    assert out["LEVRATIO"].tolist() == [0, 1, 2, 2]
    assert out["DEBT2INC"].tolist() == [0, 10, 10, 0.5]


@pytest.mark.skipif(not (ROOT / "data/raw/p09pi6.dta").exists(), reason="panel data not available")
def test_panel_frame_aligned_and_labels_consistent():
    df, info = PM.build_panel_frame()
    assert info["families"] * 5 == info["records"]
    assert (df.groupby("family_id").implicate.nunique() == 5).all()
    assert (df["LATE60"] <= df["LATE"]).all() and (df["late6007"] <= df["late07"]).all()
    assert (df.loc[df["X3004_09"] == 0, "LATE"] == 0).all()
    assert np.isfinite(df.filter(regex="^f0[79]_").to_numpy()).all()
