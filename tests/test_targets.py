from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scf_distress import features as F

ROOT = Path(__file__).resolve().parents[1]
WAVES = [2010, 2013, 2016, 2019, 2025]


def _synthetic():
    return pd.DataFrame({"X3004": [0, 1, 5, 5, 0], "X3005": [0, 0, 1, 5, 1],
                         "LATE": [0, 0, 1, 1, 0], "LATE60": [0, 0, 1, 0, 1],
                         "year": 2016, "family_id": [1, 2, 3, 4, 5], "HDEBT": [0, 1, 1, 1, 0]})


def test_make_labels_forces_late60_le_late():
    out, info = F.make_labels(_synthetic())
    assert (out["LATE60"] <= out["LATE"]).all()
    assert out["LATE60"].tolist() == [0, 0, 1, 0, 0]
    assert info["late60_without_late_records"] == 1


def test_population_masks():
    df = _synthetic()
    assert F.population_mask(df, "A_all").all()
    assert F.population_mask(df, "B_indebted").tolist() == [False, True, True, True, False]
    assert F.population_mask(df, "B2_payments_due").tolist() == [False, True, True, True, False]


def test_blocks_are_nested_and_whitelisted():
    prev = []
    for b in F.BLOCKS:
        cur = F.block_features(b)
        assert cur[: len(prev)] == prev and len(cur) > len(prev)
        assert len(set(cur)) == len(cur)
        prev = cur
    banned = {x.lower() for x in F.BANNED}
    assert not set(F.block_features("M4")) & banned
    assert "asinh_networth" not in F.block_features("M3")  # NETWORTH = ASSET - DEBT would leak debt into M3


@pytest.fixture(scope="module")
def processed():
    files = [ROOT / "data/processed" / f"bullit_{y}.parquet" for y in WAVES]
    if not all(f.exists() for f in files):
        pytest.skip("processed SCF waves not available")
    df = F.load_processed(ROOT / "data/processed", WAVES)
    return F.make_labels(df)


def test_real_labels_binary_and_consistent(processed):
    df, info = processed
    for oc in F.OUTCOMES:
        assert set(df[oc].unique()) <= {0, 1}
    assert (df["LATE60"] <= df["LATE"]).all()
    assert info["late60_without_late_families"] == [[2016, 3866]]
    # LATE is defined only for families with payments due; otherwise 0 by construction
    assert (df.loc[df["X3004"] == 0, "LATE"] == 0).all()


def test_real_features_complete(processed):
    df, _ = processed
    X = F.build_features(df)
    assert list(X.columns) == F.block_features("M4")
    assert np.isfinite(X.to_numpy()).all()
