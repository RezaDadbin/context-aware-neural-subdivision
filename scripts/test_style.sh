#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 STYLE_NAME TEST_OBJ"
  echo "Example: $0 bunny data/test_objects/egg_chair.obj"
  exit 2
fi

STYLE="$1"
TEST_OBJ="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURAL="${ROOT}/external/neuralSubdiv"
TEST_ABS="$(cd "$(dirname "${TEST_OBJ}")" && pwd)/$(basename "${TEST_OBJ}")"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"

if [[ ! -f "${TEST_ABS}" ]]; then
  echo "Missing test OBJ: ${TEST_OBJ}"
  exit 3
fi

echo "============================================================"
echo "Testing style model with official test.py"
echo "Style:    ${STYLE}"
echo "Test OBJ: ${TEST_ABS}"
echo "Output:   ${NEURAL}/jobs/net_style_${STYLE}/"
echo "============================================================"

(
  cd "${NEURAL}"
  "${PYTHON_BIN}" -u test.py "./jobs/net_style_${STYLE}/" "${TEST_ABS}"
)
