import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from scf_distress import evaluate as ev

rng = np.random.default_rng(0)
N = 2000
P = rng.uniform(0.01, 0.99, N)
Y = (rng.uniform(size=N) < P).astype(int)
W = rng.integers(1, 4, N).astype(float)


def test_uniform_weights_match_unweighted():
    assert ev.roc_auc(Y, P, np.ones(N)) == pytest.approx(roc_auc_score(Y, P))
    assert ev.brier(Y, P, np.full(N, 7.0)) == pytest.approx(np.mean((P - Y) ** 2))


def test_integer_weights_equal_row_replication():
    yr, pr = np.repeat(Y, W.astype(int)), np.repeat(P, W.astype(int))
    for f in (ev.roc_auc, ev.pr_auc, ev.brier, ev.log_loss):
        assert f(Y, P, W) == pytest.approx(f(yr, pr), rel=1e-9)
    assert ev.weighted_prevalence(Y, W) == pytest.approx(yr.mean())


def test_metrics_invariant_to_weight_scale():
    m1, m2 = ev.all_metrics(Y, P, W), ev.all_metrics(Y, P, W * 1234.5)
    for k in m1:
        assert m1[k] == pytest.approx(m2[k], rel=1e-6)


def test_top_share_recall_exact_example():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(1, 0.1, 10)
    out = ev.top_share_recall(y, p, None, share=0.2)
    assert out["recall"] == pytest.approx(0.5)  # top 2 contain one of two positives
    assert out["lift"] == pytest.approx(0.5 / 0.2)


def test_top_share_uses_weight_not_rows():
    y = np.array([1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    w = np.array([1.0, 1.0, 1.0, 7.0])  # top 10% of weight = the first family only
    assert ev.top_share_recall(y, p, w, share=0.1)["recall"] == pytest.approx(1.0)


def test_perfect_calibration_slope_near_one():
    rng2 = np.random.default_rng(1)
    p = rng2.uniform(0.02, 0.6, 200_000)
    y = (rng2.uniform(size=p.size) < p).astype(int)
    cal = ev.calibration_intercept_slope(y, p)
    assert cal["slope"] == pytest.approx(1.0, abs=0.03)
    assert cal["citl_intercept"] == pytest.approx(0.0, abs=0.03)


def test_constant_predictions_have_citl_but_no_slope():
    p = np.full(N, 0.3)
    cal = ev.calibration_intercept_slope(Y, p, W)
    assert np.isnan(cal["slope"])
    prev = ev.weighted_prevalence(Y, W)
    expected = np.log(prev / (1 - prev)) - np.log(0.3 / 0.7)
    assert cal["citl_intercept"] == pytest.approx(expected, abs=1e-6)
    assert ev.roc_auc(Y, p, W) == pytest.approx(0.5)


def test_calibration_table_weight_shares():
    t = ev.calibration_table(Y, P, W, n_bins=10)
    assert t["weight_share"].sum() == pytest.approx(1.0)
    assert len(t) == 10


def test_threshold_for_share_then_threshold_metrics():
    thr = ev.threshold_for_share(P, W, 0.1)
    flagged = ev.threshold_metrics(Y, P, thr, W)["flagged_share"]
    assert 0.09 < flagged < 0.12


def test_rejects_nonbinary_labels():
    with pytest.raises(ValueError):
        ev.brier(np.array([0, 2]), np.array([0.1, 0.2]))
