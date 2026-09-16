#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 JOB_NAME"
  exit 2
fi

NAME="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURAL="${ROOT}/external/neuralSubdiv"
JOB="./jobs/net_context_${NAME}/"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"

(
  cd "${NEURAL}"
  "${PYTHON_BIN}" -u train_context.py "${JOB}"
)
