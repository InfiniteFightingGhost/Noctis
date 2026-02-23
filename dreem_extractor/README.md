# Dreem Extractor

Production-grade data extraction for Dreem Open Dataset (DOD-H / DOD-O) HDF5 files.

## Install
```bash
pip install -r requirements.txt
```

## CLI
```bash
dreem_extractor extract --input /path/to/records --output /tmp/dreem-out --config dreem_extractor/config/defaults.yaml
```

Outputs per record:
- `<record_id>.npz` with `hypnogram`, `features`, `valid_mask`, optional `timestamps`
- `<record_id>.json` metadata
- `<record_id>.qc.json` QC summary
- `manifest.jsonl` (dataset index, optional)

## Feature Order
```
in_bed_pct, hr_mean, hr_std, dhr, rr_mean, rr_std, drr, large_move_pct,
minor_move_pct, turnovers_delta, apnea_delta, flags, vib_move_pct, vib_resp_q, agree_flags
```

## Flags
- bit0: EPOCH_VALID
- bit1: STAGE_SCORED
- bit2: ECG_PRESENT
- bit3: HR_VALID
- bit4: RESP_PRESENT
- bit5: RR_VALID
- bit6: RR_FROM_EDR
- bit7: UNSUPPORTED_FIELDS

## Agree Flags
- bit0: HR_RANGE_OK
- bit1: RR_RANGE_OK
- bit2: HR_RR_PLAUSIBLE

## Tests
```bash
pytest tests/dreem_extractor
```

## Lint + Types
```bash
ruff check dreem_extractor tests/dreem_extractor
mypy dreem_extractor
```
