FEATURE_SPEC_VERSION = "v1"
EXTRACTOR_VERSION = "0.1.0"
EPOCH_SEC = 30

STAGE_UNKNOWN = -1
STAGE_WAKE = 0
STAGE_N1 = 1
STAGE_N2 = 2
STAGE_N3 = 3
STAGE_REM = 4

UINT8_UNKNOWN = 255
INT8_UNKNOWN = -1

FEATURE_ORDER = [
    "in_bed_pct",
    "hr_mean",
    "hr_std",
    "dhr",
    "rr_mean",
    "rr_std",
    "drr",
    "large_move_pct",
    "minor_move_pct",
    "turnovers_delta",
    "apnea_delta",
    "flags",
    "vib_move_pct",
    "vib_resp_q",
    "agree_flags",
]


class FlagBits:
    EPOCH_VALID = 0
    STAGE_SCORED = 1
    ECG_PRESENT = 2
    HR_VALID = 3
    RESP_PRESENT = 4
    RR_VALID = 5
    RR_FROM_EDR = 6
    UNSUPPORTED_FIELDS = 7


class AgreeBits:
    HR_RANGE_OK = 0
    RR_RANGE_OK = 1
    HR_RR_PLAUSIBLE = 2


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
