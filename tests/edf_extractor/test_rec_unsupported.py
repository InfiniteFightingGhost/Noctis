from __future__ import annotations

import pytest

from edf_extractor.config import load_config
from edf_extractor.pipeline import extract_record
from extractor_hardened.errors import ExtractionError


def test_rec_is_explicitly_unsupported(tmp_path) -> None:
    rec_path = tmp_path / "sample.rec"
    rec_path.write_bytes(b"dummy")
    hyp_path = tmp_path / "sample_1.txt"
    hyp_path.write_text("0\n", encoding="utf-8")
    with pytest.raises(ExtractionError) as exc:
        extract_record(rec_path, hyp_path, load_config())
    assert exc.value.code == "E_UNSUPPORTED_REC"
