from __future__ import annotations

import hashlib

import h5py
import numpy as np

from dreem_extractor.config import load_config
from dreem_extractor.pipeline import extract_record
from dreem_extractor.serialize.writers import write_record_outputs
from extractor_hardened.validator import validate_artifact_dir


def test_artifact_golden_checksum_stable(tmp_path) -> None:
    record_path = tmp_path / "recording.h5"
    fs = 50.0
    n_epochs = 3
    samples = int(fs * 30 * n_epochs)
    t = np.arange(samples, dtype=np.float32) / fs
    ecg = 0.05 * np.sin(2 * np.pi * 1.0 * t)
    ecg[np.arange(0, samples, int(fs))] += 3.0

    with h5py.File(record_path, "w") as h5file:
        h5file.create_dataset("hypnogram", data=np.array([0, 1, 2], dtype=np.int8))
        signals = h5file.create_group("signals")
        emg = signals.create_group("emg")
        ds = emg.create_dataset("ECG", data=ecg)
        ds.attrs["sampling_rate"] = fs

    result = extract_record(record_path, load_config())
    out_a = write_record_outputs(result, tmp_path / "out_a")
    out_b = write_record_outputs(result, tmp_path / "out_b")
    validate_artifact_dir(out_a["record_dir"])
    validate_artifact_dir(out_b["record_dir"])

    digest_a = _artifact_digest(out_a["record_dir"])
    digest_b = _artifact_digest(out_b["record_dir"])
    assert digest_a == digest_b


def _artifact_digest(record_dir) -> str:
    digest = hashlib.sha256()
    for name in ["features.npy", "labels.npy", "valid_mask.npy", "manifest.json"]:
        payload = (record_dir / name).read_bytes()
        digest.update(payload)
    return digest.hexdigest()
