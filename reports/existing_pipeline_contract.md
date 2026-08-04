# Existing pipeline contract

- Repository root: `not_a_git_repository`
- Active commit: `not_available`
- Entry point used by previous experiment: not found in current workspace
- Expected directory structure: `data/thingi10k50/` with `models/<file_id>/...`
- Coordinate normalization convention: fallback AABB-center + longest-side-to-2.0 scaling
- Mesh units/orientation assumptions: preserved from source; no additional axis reorientation
- Coarse-mesh target resolution: fallback `~3500` vertices, minimum `1000`
- Midpoint-subdivision steps: fallback `1`
- Graph Laplacian type: uniform combinatorial Laplacian `L = D - A`
- Target quantity: `delta_target = L_exp @ target_positions`
- Multi-view inputs and camera convention: configuration included; renderer integration point left lightweight due missing existing experiment renderer in workspace
- Train/validation/test loader expectations: deterministic `40/5/5` split in both JSON and CSV

