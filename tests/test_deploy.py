import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scf_distress import deploy as D
from scf_distress import score as S

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not D.MODEL_PATH.exists(), reason="deployment model not built")


def p_any(h):
    return S.score(h)["scores"]["any_late_payment_12m"]["probability"]


def test_model_file_consistent():
    m = S.load_model()
    assert set(m["inputs"]) <= set(D.INPUTS)
    for spec in m["outcomes"].values():
        assert set(spec["features"]) <= set(m["feature_means_2025"])
        q = np.array(spec["percentiles_2025"])
        assert len(q) == 101 and np.all(np.diff(q) >= 0)


def test_monotone_in_key_inputs():
    base = {"age": 40, "liquid_assets": 2000, "total_debt": 30000, "home_value": 0, "education_debt": 5000}
    liq = [p_any({**base, "liquid_assets": v}) for v in (0, 100, 1000, 10000, 100000)]
    assert all(a >= b for a, b in zip(liq, liq[1:])), liq
    debt = [p_any({**base, "total_debt": v}) for v in (5000, 20000, 80000, 300000)]
    assert all(a <= b for a, b in zip(debt, debt[1:])), debt
    home = [p_any({**base, "home_value": v}) for v in (0, 100000, 400000)]
    assert all(a >= b for a, b in zip(home, home[1:])), home
    stud = [p_any({**base, "education_debt": v}) for v in (0, 5000, 25000)]
    assert all(a <= b for a, b in zip(stud, stud[1:])), stud
    assert p_any({**base, "total_debt": 0, "education_debt": 0}) < p_any(base)


def test_missing_inputs_use_feature_means():
    m = S.load_model()
    spec = m["outcomes"]["any_late_payment_12m"]
    z = spec["intercept"] + sum(spec["coefficients"][c] * m["feature_means_2025"][c] for c in spec["features"])
    expected = 1 / (1 + math.exp(-z))
    out = S.score({})
    assert out["scores"]["any_late_payment_12m"]["probability"] == pytest.approx(expected, abs=1e-4)
    assert set(out["imputed_inputs"]) == set(m["inputs"])


def test_partial_input_only_changes_its_features():
    m = S.load_model()
    full = S.score({"liquid_assets": 500})
    assert "liquid_assets" not in full["imputed_inputs"]
    assert len(full["imputed_inputs"]) == len(m["inputs"]) - 1


def test_input_validation():
    with pytest.raises(ValueError):
        S.score({"liquid_assets": -5})
    with pytest.raises(ValueError):
        S.score({"total_debt": 1000, "education_debt": 5000})
    with pytest.raises(ValueError):
        S.score({"education": 7})
    with pytest.raises(ValueError):
        S.score({"favorite_color": "blue"})
    assert S.score({"education": "college", "income_unusually_low": "no"})["inputs_used"]["education"] == 4


def test_batch_matches_single():
    hh = [{"age": 30, "liquid_assets": 300, "total_debt": 20000},
          {"age": 60, "liquid_assets": 50000, "total_debt": 0, "home_value": 300000}]
    batch = S.score_frame(pd.DataFrame(hh))
    for i, h in enumerate(hh):
        assert batch["p_any_late_payment_12m"].iloc[i] == pytest.approx(p_any(h), abs=1e-4)


def test_training_and_scoring_share_feature_code():
    # SCF records mapped to inputs and scored through score_frame reproduce the model's direct prediction
    df = D.load()
    g = df[(df.year == 2025) & (df.implicate == 1)].head(50)
    m = S.load_model()
    spec = m["outcomes"]["any_late_payment_12m"]
    X = D.build_features(g, list(m["inputs"]))[spec["features"]].to_numpy()
    coef = np.array([spec["coefficients"][c] for c in spec["features"]])
    direct = 1 / (1 + np.exp(-(X @ coef + spec["intercept"])))
    via_api = S.score_frame(g[list(m["inputs"])])["p_any_late_payment_12m"].to_numpy()
    assert np.allclose(direct, via_api, atol=1e-4)
