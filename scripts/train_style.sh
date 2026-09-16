#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 STYLE_NAME"
  echo "Example: $0 bunny"
  exit 2
fi

STYLE="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURAL="${ROOT}/external/neuralSubdiv"
JOB="./jobs/net_style_${STYLE}/"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"

echo "============================================================"
echo "Training style model"
echo "Style: ${STYLE}"
echo "Job:   ${NEURAL}/${JOB}"
echo "Run this script again to resume from checkpoint_latest.pt."
echo "============================================================"

(
  cd "${NEURAL}"
  "${PYTHON_BIN}" -u train_resume.py "${JOB}"
)
