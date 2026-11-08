import numpy as np
import pandas as pd
import pytest

from scf_distress import schema


def _write_fake_wave(tmp_path, yy1_dtype):
    yy1 = np.repeat(np.arange(6000, 6003), 5).astype(yy1_dtype)  # 10*yy1 > int16 max
    imp = np.tile(np.arange(1, 6), 3)
    df = pd.DataFrame({"YY1": yy1, "Y1": (yy1.astype(np.int64) * 10 + imp).astype(np.int32),
                       "X3004": np.tile([0, 1, 5, 5, 1], 3).astype(np.int8)})
    df.to_stata(tmp_path / "p19i6.dta", write_index=False)


@pytest.mark.parametrize("dtype", [np.int16, np.int32])
def test_ids_parse_without_overflow(tmp_path, dtype):
    _write_fake_wave(tmp_path, dtype)
    d = schema.read_wave(tmp_path, 2019, ["x3004"])
    assert list(d.columns[:3]) == ["year", "family_id", "implicate"]
    assert d["implicate"].tolist() == list(range(1, 6)) * 3
    assert d.groupby("family_id")["implicate"].nunique().eq(5).all()


def test_missing_column_raises(tmp_path):
    _write_fake_wave(tmp_path, np.int32)
    with pytest.raises(KeyError):
        schema.read_wave(tmp_path, 2019, ["x99999"])


def test_wave_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        schema.wave_file(tmp_path, 2025)
