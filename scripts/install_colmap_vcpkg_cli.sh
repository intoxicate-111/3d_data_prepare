#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="colmap_vcpkg_cli"
VCPKG_ROOT="$HOME/vcpkg_colmap_cli"
TRIPLET="x64-linux"

echo "==> 1. Locate Conda"
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not on PATH."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "==> 2. Create/update dedicated build environment: $ENV_NAME"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda install -n "$ENV_NAME" --override-channels -c conda-forge -y         python=3.12 cmake ninja git pkg-config         autoconf autoconf-archive automake libtool         bison flex m4 make patch perl
else
    conda create -n "$ENV_NAME" --override-channels -c conda-forge -y         python=3.12 cmake ninja git pkg-config         autoconf autoconf-archive automake libtool         bison flex m4 make patch perl
fi

conda activate "$ENV_NAME"

unset LD_LIBRARY_PATH || true
unset PKG_CONFIG_PATH || true
unset CMAKE_PREFIX_PATH || true
unset LIBRARY_PATH || true
unset CPATH || true
unset C_INCLUDE_PATH || true
unset CPLUS_INCLUDE_PATH || true

echo "==> 3. Check required host tools"
for tool in git cmake ninja gcc g++ autoconf automake autoreconf libtoolize bison flex make patch perl; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: required tool not found: $tool"
        exit 1
    fi
    printf "  %-12s %s\n" "$tool" "$(command -v "$tool")"
done

echo "==> 4. Prepare dedicated vcpkg checkout"
if [ -d "$VCPKG_ROOT/.git" ]; then
    git -C "$VCPKG_ROOT" pull --ff-only
else
    rm -rf "$VCPKG_ROOT"
    git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
fi

cd "$VCPKG_ROOT"

echo "==> 5. Bootstrap vcpkg"
./bootstrap-vcpkg.sh -disableMetrics

echo "==> 6. Install COLMAP CLI core only (no GUI)"
./vcpkg install "colmap[core]:${TRIPLET}"

echo "==> 7. Locate COLMAP executable"
COLMAP_BIN="$(find "$VCPKG_ROOT/installed/$TRIPLET" -type f -name colmap -perm -u+x 2>/dev/null | head -n 1 || true)"

if [ -z "$COLMAP_BIN" ]; then
    echo "ERROR: COLMAP executable could not be located."
    echo "Try: find \"$VCPKG_ROOT/installed/$TRIPLET\" -type f -name colmap"
    exit 1
fi

echo "COLMAP_BIN=$COLMAP_BIN"

echo "==> 8. Smoke test"
"$COLMAP_BIN" -h | head -n 20
"$COLMAP_BIN" feature_extractor -h >/dev/null
echo "feature_extractor_OK"
"$COLMAP_BIN" exhaustive_matcher -h >/dev/null
echo "exhaustive_matcher_OK"
"$COLMAP_BIN" point_triangulator -h >/dev/null
echo "point_triangulator_OK"

echo
echo "============================================================"
echo "COLMAP CLI installation succeeded."
echo "Use this exact executable:"
echo "  $COLMAP_BIN"
echo "Do not use the broken Conda colmap from base/test."
echo "============================================================"