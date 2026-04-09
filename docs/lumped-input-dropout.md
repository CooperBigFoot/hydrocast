# Lumped Input Dropout As Data Augmentation

This file is written for LLM agents and humans who need to configure training quickly without re-reading the code.

## Purpose

Use `data.input_dropout` to train lumped models that do not over-rely on specific dynamic inputs.

Supported paths:

- classic lumped inputs from `parquet` or `zarr`, using `data.dynamic_features`
- packed lumped inputs from `format: packed` with `dynamic_layout: lumped`, using `data.spatial_channels`

Not supported in this implementation:

- `format: spatial`
- packed gridded inputs
- any gridded dropout path

## Exact Semantics

Dropout is applied **per sample**, not per batch.

For each fetched training sample:

1. Build a deterministic RNG from `seed + sample_idx`
2. Sample activation using `apply_prob`
3. If activated, sample how many features to drop using `count_probs`
4. Sample that many distinct features from `features`
5. For each dropped feature:
   - set the feature to `0.0` for the sample's full input sequence
   - set the matching `<feature>_was_filled` channel to `1.0` for that same full input sequence
   - if the feature also appears in `future`, zero it there too

Important consequences:

- the same sample always gets the same dropout mask
- masks do **not** vary across epochs
- dropped features are zeroed across all timesteps of the sample input window
- this is not a per-timestep mask
- this is not one shared decision for the whole batch

## Config Shape

Place the config under `data.input_dropout`.

```yaml
data:
  input_dropout:
    enabled: true
    features: [precip, temp]
    apply_prob: 0.35
    count_probs: [0.60, 0.35]
    seed: 42
```

Meaning:

- `enabled`: master switch
- `features`: base feature names eligible for dropout
- `apply_prob`: probability that dropout logic runs for a sample
- `count_probs[k]`: probability weight for dropping exactly `k` features
- `seed`: deterministic seed base

Rules:

- `features` must contain only base names like `precip`, never `precip_was_filled`
- `count_probs` must have length `len(features)`
- valid drop counts are `0..N-1`, where `N = len(features)`
- dropping all configured features is not allowed
- `count_probs` is normalized internally
- `count_probs` may still allow `0` even when dropout activates

## Required Channel Contract

For every droppable feature `x`, the active input channels must also include `x_was_filled`.

Example:

- if `features: [precip, temp]`
- then active input channels must include:
  - `precip`
  - `precip_was_filled`
  - `temp`
  - `temp_was_filled`

If the matching `_was_filled` channels are missing, config validation fails fast.

## Classic Lumped Setup

Use this for `format: parquet` or `format: zarr`.

The active dynamic inputs come from `data.dynamic_features`.

Example:

```yaml
data:
  format: parquet
  dynamic_features:
    - precip
    - precip_was_filled
    - temp
    - temp_was_filled
    - streamflow
  target_feature: streamflow
  input_dropout:
    enabled: true
    features: [precip, temp]
    apply_prob: 0.35
    count_probs: [0.60, 0.35]
    seed: 42
```

Notes:

- explicitly requesting `*_was_filled` channels keeps them in the loaded feature tensor
- `target_was_filled` metadata behavior is unchanged

## Packed Lumped Setup

Use this for `format: packed` with `dynamic_layout: lumped`.

The active dynamic inputs come from `data.spatial_channels`.

Example:

```yaml
data:
  format: packed
  dynamic_layout: lumped
  spatial_path: /path/to/packed
  spatial_channels:
    - precip
    - precip_was_filled
    - temp
    - temp_was_filled
  target_feature: streamflow
  dynamic_features: [streamflow]
  input_dropout:
    enabled: true
    features: [precip, temp]
    apply_prob: 0.35
    count_probs: [0.60, 0.35]
    seed: 42
```

Packed-specific notes:

- packed dropout uses packed channel names from `packed.zarr`
- packed data must already contain the matching `_was_filled` channels in `dynamic_channels`
- this is expected to come from the SLOTH pack pipeline
- model input width for packed lumped runs is derived from `spatial_channels`

## Choosing `count_probs`

If you have `N` droppable features, `count_probs` must have length `N`.

Examples:

- `features: [precip, temp]`
  - valid counts are `0` or `1`
  - example: `count_probs: [0.70, 0.30]`

- `features: [precip, temp, pet]`
  - valid counts are `0`, `1`, or `2`
  - example: `count_probs: [0.45, 0.40, 0.15]`

- `features: [a, b, c, d, e]`
  - valid counts are `0`, `1`, `2`, `3`, or `4`
  - example: `count_probs: [0.35, 0.35, 0.18, 0.09, 0.03]`

If you want the augmentation to prefer dropping only one feature, put most mass on `count_probs[1]`.

## What An LLM Agent Should Do

If the user says: "set up training with dropout as DA"

1. Identify whether the run is classic lumped or packed lumped
2. Pick the droppable base features
3. Ensure the active input channels also include matching `*_was_filled` channels
4. Add `data.input_dropout`
5. Set `count_probs` to length `len(features)`
6. Do not configure this for spatial or gridded inputs
7. For packed runs, verify `spatial_channels` names match packed `dynamic_channels`

## Common Mistakes

- Putting `precip_was_filled` inside `input_dropout.features`
- Forgetting to include `precip_was_filled` in the active input channel list
- Using `count_probs` with the wrong length
- Assuming masks vary across epochs
- Assuming dropout is applied once per batch
- Trying to use this on gridded inputs
