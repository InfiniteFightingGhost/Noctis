FEATURE_SPEC_VERSION = "v1"
EXTRACTOR_VERSION = "0.1.0"
EPOCH_SEC = 30

STAGE_UNKNOWN = -1
STAGE_WAKE = 0
STAGE_N1 = 1
STAGE_N2 = 2
STAGE_N3 = 3
STAGE_REM = 4

FEATURE_ORDER = [
    "hr_mean",
    "hr_std",
    "dhr",
    "rr_mean",
    "rr_std",
    "drr",
    "minor_move_pct",
    "large_move_pct",
    "turnovers_delta",
    "in_bed_pct",
]

DISORDER_BY_PREFIX = {
    "n": "control",
    "ins": "insomnia",
    "brux": "bruxism",
    "plm": "plm",
    "rbd": "rbd",
    "sdb": "sdb",
    "narco": "narcolepsy",
    "nfle": "other",
}
