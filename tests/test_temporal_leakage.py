from pathlib import Path

import pandas as pd
import pytest

from scf_distress.config import load_config
from scf_distress.splits import LeakageError, Split, assert_disjoint, partition, splits_from_config

CFG = load_config(Path(__file__).resolve().parents[1] / "config.yaml")


def test_project_config_splits():
    sp = splits_from_config(CFG)
    test = [s for s in sp if s.role == "test"]
    assert len(test) == 1 and test[0].evaluate == 2025
    for s in sp:
        assert 2025 not in s.train
        assert 2022 not in s.train and s.evaluate != 2022
    for s in sp:
        if s.role == "tuning":
            assert s.evaluate != 2025


def test_future_training_wave_rejected():
    with pytest.raises(LeakageError):
        Split("bad", (2010, 2019), 2016, "tuning")


def test_test_wave_in_tuning_rejected():
    cfg = {"data": CFG["data"], "splits": {"tuning_folds": [{"train": [2010, 2025], "validate": 2016}],
                                           "final_train": [2010], "test": 2025}}
    with pytest.raises(LeakageError):
        splits_from_config(cfg)


def test_partition_keeps_implicates_together():
    df = pd.DataFrame({"year": [2016] * 5 + [2019] * 5, "family_id": [1] * 5 + [1] * 5,
                       "implicate": list(range(1, 6)) * 2})
    tr, ev = partition(df, Split("t", (2016,), 2019, "tuning"))
    assert len(tr) == 5 and len(ev) == 5


def test_assert_disjoint_detects_overlap():
    a = pd.DataFrame({"year": [2019], "family_id": [7]})
    with pytest.raises(LeakageError):
        assert_disjoint(a, a)
