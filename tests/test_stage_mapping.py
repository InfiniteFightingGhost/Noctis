import numpy as np

from extractor.hypnogram import map_hypnogram


def test_stage_mapping_numeric_pass_through():
    stages, known = map_hypnogram(np.array([0, 1, 2, 3, 4]))
    assert stages.tolist() == [0, 1, 2, 3, 4]
    assert known.tolist() == [True, True, True, True, True]


def test_stage_mapping_numeric_shift():
    stages, known = map_hypnogram(np.array([1, 2, 3, 4, 5]))
    assert stages.tolist() == [0, 1, 2, 3, 4]
    assert all(known)


def test_stage_mapping_unknowns():
    stages, known = map_hypnogram(np.array([0, -1, 9, 2]))
    assert stages.tolist() == [0, -1, -1, 2]
    assert known.tolist() == [True, False, False, True]


def test_stage_mapping_strings():
    stages, known = map_hypnogram(np.array(["W", "N1", "N2", "N3", "REM"]))
    assert stages.tolist() == [0, 1, 2, 3, 4]
    assert all(known)
