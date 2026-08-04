# Task Specification: Prepare a 50-Model Thingi10K Dataset for the Existing Laplacian Mesh Experiment

## 1. Objective

Prepare a reproducible subset of **50 Thingi10K meshes** and convert them into the exact data format required by the existing mesh-refinement / per-vertex Laplacian-prediction experiment.

The final output must support the next experimental stage without requiring manual cleaning or path editing. Do **not** train the model in this task. Only collect, select, preprocess, validate, and package the data, then run a small end-to-end smoke test using the existing experiment code.

## 2. Important constraints

1. Work on Linux without assuming `sudo` access.
2. Use the existing Conda or virtual environment when possible. Do not modify system Python.
3. Inspect the existing experiment repository before implementing anything:
   - identify its normalization convention;
   - identify the Laplacian definition and sparse-matrix format;
   - identify the expected mesh, image, camera, and target file formats;
   - identify the existing midpoint-subdivision, nearest-surface projection, mesh simplification, and target-generation utilities;
   - reuse existing utilities rather than creating incompatible duplicates.
4. Use deterministic random seeds. Default seed: `20260804`.
5. Do not silently discard failed models. Record every rejected or failed model and the reason.
6. Do not perform destructive geometry repair on the ground-truth mesh unless it is explicitly required by the existing pipeline. In particular, do not fill holes, remesh the complete surface, or heavily smooth the source geometry by default.
7. Preserve source provenance and per-model licence metadata.

## 3. Official data source

Use the official `thingi10k` Python package and the official dataset mirrors. Prefer the package's `npz` variant because it directly exposes vertex and triangle arrays and avoids repeatedly parsing STL files.

Suggested setup inside an existing user-controlled environment:

```bash
python --version
python -m pip install --upgrade pip
python -m pip install thingi10k numpy scipy pandas trimesh tqdm fast-simplification
```

Do not use `apt`, `sudo`, or system-wide installation.

Initialise the dataset in a project-local or user-writable cache directory:

```python
import thingi10k
thingi10k.init(
    variant="npz",
    cache_dir="<PROJECT_OR_DATA_ROOT>/.cache/thingi10k",
)
```

Before filtering, inspect the actual installed API and metadata fields:

```python
import thingi10k
help(thingi10k.dataset)
entry = next(iter(thingi10k.dataset()))
print(sorted(entry.keys()))
```

Do not assume an undocumented metadata key exists. Adapt the filtering code to the installed package version.

Known corrupt source IDs that must be excluded if encountered:

- `49911`
- `74463`
- `286163`
- `81313`
- `77942`

## 4. Repository inspection

Before collecting data, create a short repository inspection note at:

```text
reports/existing_pipeline_contract.md
```

It must document:

- repository root and active Git commit;
- entry point used by the previous experiment;
- expected directory structure;
- coordinate normalization convention;
- mesh units and orientation assumptions;
- coarse-mesh target resolution;
- number of midpoint-subdivision steps;
- graph Laplacian type, for example uniform, cotangent, or normalized;
- whether the target is `L @ P_target`, displacement, absolute position, or another quantity;
- required multi-view inputs and camera convention;
- train/validation/test loader expectations.

If the repository contains several incompatible conventions, use the convention employed by the most recent successful experiment and state the reason.

## 5. Model-selection protocol

### 5.1 Candidate validity checks

Build a candidate pool from Thingi10K. A candidate must satisfy all of the following after loading:

- vertices and faces are non-empty;
- all coordinates are finite;
- all faces are triangles;
- all face indices are valid;
- at least 1,000 vertices and 2,000 faces;
- no more than 200,000 faces before preprocessing, unless the existing machine can process larger meshes safely;
- no zero-area or repeated-index triangles after minimal cleanup;
- no known corrupt source ID;
- the mesh can be loaded and exported successfully;
- preferably one connected component;
- preferably no self-intersection and manifold geometry when metadata is available.

Minimal cleanup may include:

- removing unreferenced vertices;
- removing exact duplicate faces;
- removing faces with repeated vertex indices;
- removing numerically zero-area faces;
- merging only exact or near-exact duplicate vertices using a very small tolerance relative to the bounding-box diagonal;
- fixing face orientation only when this can be done without changing geometry.

Record every cleanup operation in metadata.

### 5.2 Sampling strategy

Select exactly 50 unique models from the valid candidate pool using seed `20260804`.

Use a stratified sample so that the subset is not dominated by one complexity range:

- 15 models: approximately 2,000 to 20,000 faces;
- 20 models: approximately 20,000 to 60,000 faces;
- 15 models: approximately 60,000 to 200,000 faces.

Additional diversity requirements:

- use at most one file from the same Thingiverse `thing` when that metadata is available;
- avoid obvious duplicate or near-duplicate geometry;
- avoid selecting many models with identical tags or categories;
- include both free-form / organic objects and structured / functional objects where metadata permits;
- target approximately 40 closed meshes and 10 open meshes, but only include open meshes that remain numerically usable by the existing pipeline;
- do not select a model only because it is visually attractive; geometric diversity is more important.

If a stratum does not contain enough valid candidates, take replacements from the nearest complexity stratum and record the deviation.

### 5.3 Dataset split

Create a deterministic split:

- 40 training models;
- 5 validation models;
- 5 test models.

Stratify the split by complexity and open/closed status as far as practical. Store the split in both JSON and CSV form.

## 6. Preprocessing pipeline

For each selected model, produce the following stages.

### 6.1 Source mesh

Preserve the selected source geometry without overwriting it.

Save:

```text
source_mesh.npz
source_mesh.obj
source_metadata.json
```

The NPZ file should contain at least:

- `vertices`: `float32`, shape `[V, 3]`;
- `faces`: integer, shape `[F, 3]`.

### 6.2 Clean ground-truth mesh

Apply only the minimal cleanup defined above. Save:

```text
gt_mesh.npz
gt_mesh.obj
```

Record pre-cleaning and post-cleaning vertex/face counts.

### 6.3 Coordinate normalization

Use the exact normalization convention of the existing experiment.

If the existing repository does not define one clearly, use this fallback:

1. centre the mesh at the axis-aligned bounding-box centre;
2. uniformly scale it so that the longest bounding-box side is `2.0`;
3. keep the transformation reversible.

Save the transformation in:

```text
normalization.json
```

Include:

- original bounding box;
- centre translation;
- uniform scale;
- normalized bounding box;
- inverse transform.

Do not independently normalize the coarse and ground-truth meshes. They must share the same transform.

### 6.4 Coarse mesh

Generate the coarse mesh using the existing experiment's method and target resolution.

Preferred behaviour:

- use the repository's current simplification or coarse-mesh generator;
- preserve the overall shape and topology as far as possible;
- avoid smoothing away all high-frequency surface detail;
- save the simplification parameters.

Fallback only when no existing target is defined:

- simplify to approximately 3,500 vertices;
- keep a minimum of 1,000 vertices for small meshes;
- use deterministic quadric decimation;
- reject and replace a model if simplification produces invalid geometry that cannot be minimally repaired.

Save:

```text
coarse_mesh.npz
coarse_mesh.obj
```

### 6.5 Expanded mesh

Run the existing midpoint-subdivision implementation on the coarse mesh using the same number of subdivision steps as the previous experiment.

Fallback: one midpoint-subdivision step.

Save:

```text
expanded_mesh.npz
expanded_mesh.obj
```

Also preserve any parent-edge, parent-face, or interpolation mapping generated by the subdivision code.

### 6.6 Graph-compatible surface targets

For every expanded-mesh vertex, compute the closest point on the normalized ground-truth surface.

Save at least:

- `target_positions`: closest surface points, shape `[V_exp, 3]`;
- `target_displacements`: `target_positions - expanded_vertices`;
- `closest_face_indices`;
- `closest_barycentric_coordinates`, when available;
- `surface_distance`;
- `target_normals`, preferably interpolated from the ground-truth mesh;
- validity mask for every target.

Use the same robust nearest-surface method as the previous experiment. Do not replace surface projection with nearest-vertex matching.

Save these arrays in:

```text
targets.npz
```

### 6.7 Laplacian targets

Construct the expanded-mesh Laplacian using the exact convention of the existing experiment.

Store the sparse matrix in the format already used by the project. If no format exists, use `scipy.sparse.save_npz`.

Compute and save the target quantity required by the existing experiment. For the current per-vertex Laplacian-prediction formulation, this will normally include:

```text
delta_target = L_exp @ target_positions
```

Also save, where applicable:

- `laplacian_target`;
- `laplacian_target_normalized`;
- expanded-mesh vertex normals;
- local edge-length statistics;
- per-vertex curvature or dihedral proxy already used by the project;
- positional anchor mask or anchor indices.

Do not invent a second Laplacian convention. The target generator and training code must use the same implementation.

### 6.8 Multi-view inputs

Inspect the existing experiment and generate exactly the input modalities it requires.

If the experiment expects multi-view observations, reuse the current renderer and camera sampler. At minimum save:

- RGB or neutral-grey images;
- depth maps;
- object masks;
- camera intrinsics;
- camera extrinsics;
- optional normal maps if used by the model.

Use the previous experiment's view count and resolution. Fallback only when no prior configuration exists:

- 40 views;
- 1024 x 1024 resolution;
- cameras distributed around the normalized object with several elevation levels;
- identical camera convention for all models;
- deterministic sampling.

Save rendered data under:

```text
views/
```

Do not install or build the complete DMesh++ stack only to render the models. Prefer the existing project renderer. If no suitable renderer exists, implement the lightest user-space solution compatible with the current environment and document it.

## 7. Required output structure

Use a structure equivalent to:

```text
data/thingi10k50/
├── manifest.csv
├── manifest.json
├── split.json
├── split.csv
├── config.yaml
├── selection_candidates.csv
├── rejected_models.csv
├── failed_models.csv
├── models/
│   ├── <file_id>/
│   │   ├── source_mesh.npz
│   │   ├── source_mesh.obj
│   │   ├── source_metadata.json
│   │   ├── gt_mesh.npz
│   │   ├── gt_mesh.obj
│   │   ├── normalization.json
│   │   ├── coarse_mesh.npz
│   │   ├── coarse_mesh.obj
│   │   ├── expanded_mesh.npz
│   │   ├── expanded_mesh.obj
│   │   ├── subdivision_mapping.npz
│   │   ├── laplacian.npz
│   │   ├── targets.npz
│   │   ├── metrics.json
│   │   ├── preview.png
│   │   └── views/
│   └── ...
└── reports/
    ├── existing_pipeline_contract.md
    ├── preparation_report.md
    ├── validation_summary.csv
    └── contact_sheet.png
```

Adapt names only when the existing data loader requires a different convention. If names are changed, document the mapping.

## 8. Manifest requirements

The manifest must contain at least:

- `file_id`;
- `thing_id`, if available;
- source author/designer, if available;
- licence;
- category and tags;
- source dataset variant;
- source relative path;
- split;
- open/closed status;
- component count;
- manifold/solid/self-intersection metadata, when available;
- original vertex and face counts;
- cleaned vertex and face counts;
- coarse vertex and face counts;
- expanded vertex and face counts;
- bounding-box dimensions before normalization;
- normalization scale;
- cleanup operations;
- simplification settings;
- subdivision steps;
- Laplacian type;
- number of views and image resolution;
- mean, median, 95th percentile, and maximum projection distance;
- validation status;
- failure reason, if applicable;
- random seed;
- preparation script Git commit or checksum.

## 9. Validation

Every model must pass automated validation before it is included in the final set.

### 9.1 Geometry validation

Check:

- arrays have correct dimensions and dtypes;
- no NaN or infinite values;
- valid face indices;
- no repeated indices within a triangle;
- positive triangle area above a numerical tolerance;
- normalized geometry falls within the expected coordinate range;
- coarse and expanded meshes use the same normalization as the ground truth;
- the selected mesh has the expected connected-component status;
- exported OBJ files can be loaded again.

### 9.2 Target validation

Check:

- one target exists for every expanded vertex;
- all target positions and distances are finite;
- closest-face and barycentric data are internally consistent;
- sparse Laplacian dimensions equal `[V_exp, V_exp]`;
- `laplacian_target` has shape `[V_exp, 3]`;
- no unexpected all-zero target region exists;
- projection-distance statistics are plausible relative to the normalized object size.

### 9.3 Visual validation

Generate:

- one preview image per model showing ground truth, coarse mesh, and expanded mesh;
- projected target points or displacement vectors for a subset of vertices;
- a contact sheet for all 50 models.

The visualisation must make obvious coordinate, orientation, scaling, or projection failures.

### 9.4 End-to-end smoke test

Run the existing reconstruction/optimization code on at least:

- one training model;
- one validation model;
- one test model.

Use a short run sufficient to verify data loading, tensor shapes, sparse-matrix operations, loss computation, and output writing.

Do not claim success based only on file generation. The existing data loader and one complete forward/optimization path must run without manual changes.

## 10. Scripts and reproducibility

Create reusable command-line scripts rather than a one-off notebook.

Recommended entry point:

```bash
python scripts/prepare_thingi10k50.py \
  --config configs/thingi10k50.yaml \
  --output-root data/thingi10k50 \
  --seed 20260804
```

The pipeline should be restartable and idempotent:

- completed valid models should not be recomputed unless `--force` is passed;
- partial or corrupt outputs should be detected and regenerated;
- downloads should use a persistent cache;
- log progress to both the terminal and a file;
- save the resolved configuration with the output.

Add a validation command:

```bash
python scripts/validate_thingi10k50.py \
  --data-root data/thingi10k50
```

Add a smoke-test command using the existing experiment entry point.

## 11. Completion criteria

The task is complete only when all of the following are true:

1. Exactly 50 models are present and assigned to 40/5/5 splits.
2. All 50 pass geometry and target validation.
3. All required files can be loaded by the existing experiment without path edits.
4. Three end-to-end smoke tests complete successfully.
5. `manifest.csv`, `split.json`, rejection logs, previews, and reports are present.
6. The final report states:
   - environment and package versions;
   - data source and cache location;
   - selection criteria and seed;
   - number of candidates inspected and rejected;
   - reasons for rejection;
   - summary statistics for mesh complexity and projection error;
   - any deviations from this specification;
   - exact commands needed to reproduce the dataset.

## 12. Final response expected from the agent

When finished, report only concrete results:

- output directory;
- number of valid models and split counts;
- face/vertex count ranges;
- mean and worst projection-distance statistics;
- smoke-test status;
- scripts/configs added or modified;
- any unresolved issue that may affect the next experiment.

Do not report the task as complete if fewer than 50 models have passed validation.
