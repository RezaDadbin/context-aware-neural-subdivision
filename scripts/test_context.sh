#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 JOB_NAME TEST_OBJ [test_context.py options]"
  exit 2
fi

NAME="$1"
TEST_OBJ="$2"
shift 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURAL="${ROOT}/external/neuralSubdiv"
TEST_ABS="$(cd "$(dirname "${TEST_OBJ}")" && pwd)/$(basename "${TEST_OBJ}")"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"

(
  cd "${NEURAL}"
  "${PYTHON_BIN}" -u test_context.py "./jobs/net_context_${NAME}/" "${TEST_ABS}" "$@"
)
