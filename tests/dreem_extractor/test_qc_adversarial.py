from __future__ import annotations

import h5py
import numpy as np
import pytest

from dreem_extractor.config import load_config
from dreem_extractor.pipeline import extract_record
from extractor_hardened.errors import ExtractionError


def test_qc_fails_on_required_ecg_nan_signal(tmp_path) -> None:
    path = tmp_path / "recording.h5"
    with h5py.File(path, "w") as h5file:
        h5file.create_dataset("hypnogram", data=np.array([0, 1, 2], dtype=np.int8))
        signals = h5file.create_group("signals")
        emg = signals.create_group("emg")
        ds = emg.create_dataset("ECG", data=np.array([np.nan] * 900, dtype=np.float32))
        ds.attrs["sampling_rate"] = 10.0
    with pytest.raises(ExtractionError) as exc:
        extract_record(path, load_config())
    assert exc.value.code == "E_QC_FAIL"
