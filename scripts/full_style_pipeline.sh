#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 STYLE_NAME SOURCE_MESH"
  echo "Example: $0 bunny data/style_sources/bunny.obj"
  exit 2
fi

STYLE="$1"
SOURCE_MESH="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"

"${ROOT}/scripts/build_cxx_generator.sh"
"${ROOT}/scripts/generate_style_dataset.sh" "${STYLE}" "${SOURCE_MESH}" 200 20 500 2 20
"${PYTHON_BIN}" "${ROOT}/scripts/make_style_pkls.py" "${STYLE}" --num-train 200 --num-valid 20 --num-test 20
"${PYTHON_BIN}" "${ROOT}/scripts/write_style_hyperparams.py" "${STYLE}" --epochs 700 --device cpu
"${ROOT}/scripts/train_style.sh" "${STYLE}"
