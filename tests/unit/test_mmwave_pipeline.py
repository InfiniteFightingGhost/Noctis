from __future__ import annotations

from app.training.mmwave import FOUR_CLASS_LABELS, remap_stage_label


def test_label_remapping_matches_unified_4_class_contract() -> None:
    assert FOUR_CLASS_LABELS == ["W", "Light", "Deep", "REM"]
    assert remap_stage_label("W") == "W"
    assert remap_stage_label("N1") == "Light"
    assert remap_stage_label("N2") == "Light"
    assert remap_stage_label("N3") == "Deep"
    assert remap_stage_label("REM") == "REM"
