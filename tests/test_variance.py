import numpy as np
import pandas as pd
import pytest

from scf_distress import variance as V


def test_combine_matches_fed_formula():
    ti = [1.0, 2.0, 3.0, 4.0, 5.0]            # imputation variance (ddof=1) = 2.5
    tr = np.array([0.0, 2.0] * 500)[:999]       # sampling variance computed with /998
    out = V.combine(ti, tr)
    v_samp = np.sum((tr - tr.mean()) ** 2) / 998
    assert out["estimate"] == pytest.approx(3.0)
    assert out["se"] == pytest.approx(np.sqrt(1.2 * 2.5 + v_samp))
    assert out["ci_hi"] - out["estimate"] == pytest.approx(V.Z95 * out["se"])


def test_combine_drops_undefined_replicates():
    out = V.combine([1, 1, 1, 1, 1], [np.nan, 2.0, 4.0])
    assert out["n_replicates_used"] == 2
    assert out["se_sampling"] == pytest.approx(np.sqrt(2.0))


def test_load_replicates_treats_missing_as_not_selected(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "N_REP", 3)
    d = pd.DataFrame({"Y1": [11, 21], "YY1": [1, 2],
                      "WT1B1": [10.0, np.nan], "WT1B2": [5.0, 7.0], "WT1B3": [np.nan, 2.0],
                      "MM1": [2.0, np.nan], "MM2": [1.0, 3.0], "MM3": [np.nan, 1.0]})
    (tmp_path / "2019").mkdir()
    d.to_stata(tmp_path / "2019" / "p19_rw1.dta", write_index=False)
    R = V.load_replicates(tmp_path, 2019)
    assert R.loc[1].tolist() == [20.0, 5.0, 0.0]
    assert R.loc[2].tolist() == [0.0, 21.0, 2.0]


def test_replicate_weighted_mean_has_no_sampling_variance_when_replicates_equal():
    y = np.array([0, 1, 1, 0], float)
    R = np.tile(np.array([[1.0], [2.0], [3.0], [4.0]]), (1, 999))
    tr = (R * y[:, None]).sum(0) / R.sum(0)
    assert V.combine([0.5] * 5, tr)["se"] == pytest.approx(0.0)
