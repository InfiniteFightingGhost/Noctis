# Dreem Extractor SPEC

## Assumptions
- Input: Dreem Open Dataset HDF5 (.h5)
- Hypnogram at `/hypnogram` with codes: -1, 0..4
- Epoch length: 30s
- ECG may exist at `/signals/emg/ECG`

## Feature Schema (Fixed Order)
1. in_bed_pct (uint8, sentinel 255)
2. hr_mean (uint8, sentinel 255)
3. hr_std (uint8, sentinel 255)
4. dhr (int8, sentinel -1)
5. rr_mean (uint8, sentinel 255)
6. rr_std (uint8, sentinel 255)
7. drr (int8, sentinel -1)
8. large_move_pct (uint8, sentinel 255)
9. minor_move_pct (uint8, sentinel 255)
10. turnovers_delta (uint8, sentinel 255)
11. apnea_delta (uint8, sentinel 255)
12. flags (uint8 bitfield)
13. vib_move_pct (uint8, sentinel 255)
14. vib_resp_q (uint8, sentinel 255)
15. agree_flags (uint8 bitfield, default 0)

## HR Definition
`hr_std` is the standard deviation of instantaneous HR values within the epoch.

## RR Definition
Tier A: respiration channel if available. Tier B: ECG-derived respiration (EDR) fallback.

## Sentinel Policy
- If missing channel or invalid computation: use sentinel and unset validity flags.
- Unsupported movement/apnea fields: sentinel + UNSUPPORTED_FIELDS flag set.

## Flags
- bit0: EPOCH_VALID
- bit1: STAGE_SCORED
- bit2: ECG_PRESENT
- bit3: HR_VALID
- bit4: RESP_PRESENT
- bit5: RR_VALID
- bit6: RR_FROM_EDR
- bit7: UNSUPPORTED_FIELDS

## Valid Mask
`valid_mask = (hypnogram != -1) AND EPOCH_VALID`
