"""Chronological splits by survey wave, with leakage checks.

All splitting is by survey wave. Within a wave, the five implicates of a family
always travel together, so a family can never be on both sides of a split.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class LeakageError(AssertionError):
    """Raised when a split would let test information reach training."""


@dataclass(frozen=True)
class Split:
    name: str
    train: tuple[int, ...]
    evaluate: int
    role: str  # "tuning" or "test"

    def __post_init__(self):
        if self.evaluate in self.train:
            raise LeakageError(f"{self.name}: evaluation wave {self.evaluate} is also a training wave")
        if max(self.train) >= self.evaluate:
            raise LeakageError(f"{self.name}: training wave {max(self.train)} is not earlier than {self.evaluate}")


def splits_from_config(cfg: dict) -> list[Split]:
    """Build tuning folds and the final locked test split from the config."""
    s = cfg["splits"]
    test_wave = int(s["test"])
    out = []
    for fold in s["tuning_folds"]:
        train = tuple(int(y) for y in fold["train"])
        val = int(fold["validate"])
        if test_wave in train or val == test_wave:
            raise LeakageError(f"Test wave {test_wave} used in tuning fold {train} -> {val}")
        out.append(Split(f"tune_{val}", train, val, "tuning"))
    final = tuple(int(y) for y in s["final_train"])
    out.append(Split(f"test_{test_wave}", final, test_wave, "test"))
    excluded = set(int(y) for y in cfg["data"]["waves"])
    for sp in out:
        unknown = set(sp.train) | {sp.evaluate}
        if not unknown <= excluded:
            raise ValueError(f"{sp.name} uses waves not listed in data.waves: {sorted(unknown - excluded)}")
    return out


def partition(df: pd.DataFrame, split: Split, wave_col: str = "year") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, evaluate) rows for a split, after checking for overlap."""
    train = df[df[wave_col].isin(split.train)]
    ev = df[df[wave_col] == split.evaluate]
    assert_disjoint(train, ev)
    return train, ev


def assert_disjoint(train: pd.DataFrame, ev: pd.DataFrame, key_cols=("year", "family_id")) -> None:
    """Fail if any (year, family) key appears in both partitions."""
    k_train = set(map(tuple, train[list(key_cols)].drop_duplicates().to_numpy()))
    k_ev = set(map(tuple, ev[list(key_cols)].drop_duplicates().to_numpy()))
    overlap = k_train & k_ev
    if overlap:
        raise LeakageError(f"{len(overlap)} family keys in both partitions, e.g. {sorted(overlap)[:3]}")
