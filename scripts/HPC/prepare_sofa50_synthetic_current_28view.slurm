#!/bin/bash

#SBATCH --job-name=prep_cur28
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/data_prepare_combo/logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/data_prepare_combo/logs/%x_%j.err

set -euo pipefail

PREP_ROOT="/networkhome/WMGDS/zhou_c/data_prepare_combo"
DOWNSTREAM="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
SOFA_ROOT="/networkhome/WMGDS/zhou_c/sofa_mesh"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
export PATH="/networkhome/WMGDS/zhou_c/dmesh2/venv/bin:${PATH}"

mkdir -p "${PREP_ROOT}/logs"
cd "${PREP_ROOT}"
nvidia-smi

PYTHONPATH="${PREP_ROOT}/src:${DOWNSTREAM}/src:/networkhome/WMGDS/zhou_c/dmesh2/external/nvdiffrast" \
    "${CONDA}" run --no-capture-output -n test \
    python scripts/prepare_sofa50_synthetic_current_28view.py \
        --current-manifest "${SOFA_ROOT}/sofa50_synthetic_current/manifest.json" \
        --views-manifest "${SOFA_ROOT}/sofa50_refinement/multiview_nested_14_28_56_cpu_v3/gt_query_views_28_manifest.json" \
        --output-root "${SOFA_ROOT}/sofa50_synthetic_current_28view_v1" \
        --downstream-root "${DOWNSTREAM}" \
        --training-config "${DOWNSTREAM}/configs/learned_laplacian/train_sofa50_synthetic_current_28view_local_jitter_20k.json"
