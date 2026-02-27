# Extraction Contract

This repository enforces a deterministic extraction contract for `.h5` and `.edf` inputs.

## Artifact Layout (per record)

- `features.npy` (`float32`, shape: `[n_epochs, n_features]`)
- `labels.npy` (`int8`, values in `{-1,0,1,2,3,4}`)
- `valid_mask.npy` (`bool`, shape: `[n_epochs]`)
- `timestamps.npy` (`int64`, epoch start offset seconds)
- `manifest.json`
- `night_summary.json`

Extraction must fail fast if manifest construction fails. Partial artifact directories are contract violations.

## Error Taxonomy

- `E_ALIGN_MISMATCH`
- `E_QC_FAIL`
- `E_STAGING_MISSING`
- `E_STAGING_CONFLICT`
- `E_UNSUPPORTED_REC`
- `E_CHANNEL_AMBIGUOUS`
- `E_CONTRACT_VIOLATION`

## Alignment Modes

- `strict`: mismatch between expected and available samples fails extraction.
- `reconcile`: deterministic truncation/padding with invalid mask propagation and reason logging.

## Causality

All per-epoch ML features are causal: epoch `t` can only use signal prefix `<= t`.
Night summaries are isolated from per-epoch feature generation and are not fed back into features.
