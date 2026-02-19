from __future__ import annotations

from app.reproducibility.hashing import hash_artifact_dir


def test_hash_artifact_dir_changes_on_edit(tmp_path) -> None:
    (tmp_path / "weights.npy").write_bytes(b"abc")
    (tmp_path / "bias.npy").write_bytes(b"def")
    first = hash_artifact_dir(tmp_path)
    (tmp_path / "bias.npy").write_bytes(b"defg")
    second = hash_artifact_dir(tmp_path)
    assert first != second
