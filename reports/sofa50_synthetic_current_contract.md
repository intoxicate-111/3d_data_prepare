# Sofa50 synthetic current-query data contract

## Scope

This dataset is generated only for Experiment B. It does not replace or modify the existing GT-query dataset used by frozen Experiment A.

## Source and split

- Source manifest: `~/sofa_mesh/sofa50_refinement/multiview_nested_14_28_56_cpu_v3/gt_query_views_14_manifest.json`
- Output: `~/sofa_mesh/sofa50_synthetic_current/`
- Object split: 40 train, 5 validation, 5 test.
- Variants per object: 5.
- All variants of one object retain the object's source split.
- Resulting counts: 200 train, 25 validation, 25 test.

## Current graph and target

Each current graph is a deterministic smooth normal perturbation of the source GT topology. Perturbation magnitude is expressed relative to the source local mean incident-edge length. Local damping is applied only to vertices incident to a face that would otherwise flip or become newly degenerate.

The exact same-topology source GT vertices define `P_proxy`. For current graph connectivity `L_current` and current edge scale `h_current`, the stored targets are:

```text
delta_target = L_current @ P_proxy
delta_target_hat = delta_target / (h_current^2 + 1e-12)
```

`initial_laplacian` is `L_current @ C`; it is not zeroed for Experiment B.

## Views and visibility

- Exactly 14 existing 960 RGB observations are reused by lazy relative path.
- Intrinsics and extrinsics are copied without reinterpretation.
- Back-face and occlusion visibility is recomputed on every synthetic current graph with the repository face-ID renderer.
- The local machine uses the CPU reference backend because EGL initialization is unavailable.

## Generation

```bash
cd ~/data_prepare
conda run --no-capture-output -n test \
  python scripts/prepare_sofa50_synthetic_current.py \
    --output-root ~/sofa_mesh/sofa50_synthetic_current \
    --visibility-backend cpu \
    --workers 8
```

The generator writes:

- `manifest.json`
- `generation_config.json`
- `oracle_validation.json`
- `prepared/<split>/<object>/variant_XX.pt`
- `renderer_visibility/<split>/<object>/variant_XX.npz`

Generation fails if the object split is not 40/5/5, an object does not have five variants, the `L_current @ P_proxy` target check fails, or the h² round-trip check fails.
