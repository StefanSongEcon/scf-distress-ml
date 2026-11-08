import numpy as np
import pandas as pd
import pytest

from scf_distress.weights import check_implicates, kish_effective_n, normalize_training_weights, rubin_combine


def test_equal_wave_gives_equal_wave_totals_and_mean_one():
    wave = pd.Series([2010] * 3 + [2019] * 5)
    w = pd.Series([1.0, 2.0, 3.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    out = normalize_training_weights(w, wave, "equal_wave")
    totals = out.groupby(wave).sum()
    assert totals.iloc[0] == pytest.approx(totals.iloc[1])
    assert out.mean() == pytest.approx(1.0)
    # relative weights within a wave are preserved
    assert out.iloc[2] / out.iloc[0] == pytest.approx(3.0)


def test_raw_scheme_preserves_ratios():
    wave = pd.Series([2010, 2010, 2019])
    w = pd.Series([1.0, 3.0, 6.0])
    out = normalize_training_weights(w, wave, "raw")
    assert out.mean() == pytest.approx(1.0)
    assert out.iloc[2] / out.iloc[0] == pytest.approx(6.0)


def test_negative_weights_rejected():
    with pytest.raises(ValueError):
        normalize_training_weights(pd.Series([1.0, -1.0]), pd.Series([1, 1]))


def test_check_implicates_flags_bad_families():
    rows = [(2019, 1, i) for i in range(1, 6)] + [(2019, 2, i) for i in (1, 2, 3, 4)] + \
           [(2019, 3, i) for i in (1, 2, 3, 4, 4)]
    df = pd.DataFrame(rows, columns=["year", "family_id", "implicate"])
    bad = check_implicates(df)
    assert sorted(bad["family_id"]) == [2, 3]


def test_kish():
    assert kish_effective_n(np.ones(10)) == pytest.approx(10)
    assert kish_effective_n([1, 0, 0]) == pytest.approx(1)


def test_rubin():
    r = rubin_combine([1.0, 2.0, 3.0, 4.0, 5.0], [0.1] * 5)
    assert r["estimate"] == pytest.approx(3.0)
    assert r["total"] == pytest.approx(0.1 + 1.2 * 2.5)
