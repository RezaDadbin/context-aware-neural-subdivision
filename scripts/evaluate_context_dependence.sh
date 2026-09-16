#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURAL="${ROOT}/external/neuralSubdiv"
LOG_ROOT="${ROOT}/logs/context_dependence"
SURFACE_SAMPLES=50000
EXPECTED_MESHES=20
EXPECTED_RECORDS=180
SEED=0
DEVICE=cpu

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
if [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
fi
if [[ ! -s "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}"
  exit 3
fi

LABELS=("Bimba Head" "Fat Dragon" "Gear16")
JOB_DIRS=(
  "${NEURAL}/jobs/net_context_bimba_pilot_hybrid"
  "${NEURAL}/jobs/net_context_fat_dragon_final_hybrid"
  "${NEURAL}/jobs/net_context_gear16_final_hybrid"
)
TEST_PKLS=(
  "${NEURAL}/data_PKL/style_bimba_head_test.pkl"
  "${NEURAL}/data_PKL/style_fat_dragon_test.pkl"
  "${NEURAL}/data_PKL/style_gear16_test.pkl"
)

mkdir -p "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/all_context_modes_50k_${STAMP}.log"
START_SECONDS="${SECONDS}"
CURRENT_MODEL="none"
RESUME_COMMAND="cd \"${ROOT}\" && ./scripts/evaluate_context_dependence.sh"

exec > >(tee -a "${LOG_FILE}") 2>&1

run_awake() {
  if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -dimsu "$@"
  else
    "$@"
  fi
}

require_file() {
  [[ -s "$1" ]] || {
    echo "Missing required file: $1"
    exit 3
  }
}

evaluation_complete() {
  local output="$1"
  local checkpoint="$2"
  local pkl="$3"

  "${PYTHON_BIN}" - "${output}" "${checkpoint}" "${pkl}" \
    "${SURFACE_SAMPLES}" "${EXPECTED_MESHES}" "${EXPECTED_RECORDS}" <<'PY'
import json
import os
import sys

output, checkpoint, pkl, samples, meshes, records = sys.argv[1:]
if not os.path.isfile(output) or os.path.getsize(output) == 0:
    raise SystemExit(1)
if os.path.getmtime(output) < max(os.path.getmtime(checkpoint), os.path.getmtime(pkl)):
    raise SystemExit(1)
try:
    with open(output, "r") as handle:
        report = json.load(handle)
except (OSError, ValueError):
    raise SystemExit(1)

aggregate = report.get("aggregate", {})
required_modes = {
    "context_level_2", "disabled_level_2", "shuffled_level_2"
}
valid = (
    report.get("model") == "context" and
    os.path.realpath(report.get("checkpoint", "")) == os.path.realpath(checkpoint) and
    os.path.realpath(report.get("pkl", "")) == os.path.realpath(pkl) and
    int(report.get("surface_samples_per_mesh", -1)) == int(samples) and
    int(report.get("meshes_evaluated", -1)) == int(meshes) and
    len(report.get("records", [])) == int(records) and
    required_modes.issubset(aggregate)
)
raise SystemExit(0 if valid else 1)
PY
}

finish() {
  status=$?
  elapsed=$((SECONDS - START_SECONDS))
  printf '\n============================================================\n'
  if [[ "${status}" -eq 0 ]]; then
    echo "ALL CONTEXT-DEPENDENCE EVALUATIONS COMPLETED"
  else
    echo "CONTEXT-DEPENDENCE EVALUATION STOPPED (exit ${status})"
    echo "Interrupted/current model: ${CURRENT_MODEL}"
  fi
  printf 'Elapsed this session: %02dh:%02dm:%02ds\n' \
    $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
  echo "Log: ${LOG_FILE}"
  echo "Resume with: ${RESUME_COMMAND}"
  printf '============================================================\n'
}
trap finish EXIT

echo "============================================================"
echo "PHASE 1 CONTEXT-DEPENDENCE EVALUATION"
echo "Modes: active context, disabled context, shuffled context"
echo "Models: three best-validation Phase 1 checkpoints"
echo "Test meshes: ${EXPECTED_MESHES} per model and mode"
echo "Surface samples: ${SURFACE_SAMPLES} per mesh and level"
echo "Device: ${DEVICE}"
echo "Seed: ${SEED}"
echo "Completed valid reports are skipped when this command is rerun."
echo "Log: ${LOG_FILE}"
echo "============================================================"

for index in 0 1 2; do
  CURRENT_MODEL="${LABELS[$index]}"
  job="${JOB_DIRS[$index]}"
  pkl="${TEST_PKLS[$index]}"
  checkpoint="${job}/netparams.dat"
  output="${job}/evaluation_final_700_context_dependence_50k.json"

  require_file "${job}/hyperparameters.json"
  require_file "${checkpoint}"
  require_file "${pkl}"

  if evaluation_complete "${output}" "${checkpoint}" "${pkl}"; then
    echo "[$((index + 1))/3] SKIP complete: ${CURRENT_MODEL}"
    echo "Report: ${output}"
    continue
  fi

  echo ""
  echo "[$((index + 1))/3] START: ${CURRENT_MODEL}"
  echo "Checkpoint: ${checkpoint}"
  echo "Test data: ${pkl}"
  echo "Report: ${output}"
  model_started="${SECONDS}"

  (
    cd "${NEURAL}"
    run_awake "${PYTHON_BIN}" -u evaluate_model.py \
      "${job}" "${pkl}" --model context \
      --modes context disabled shuffled --device "${DEVICE}" \
      --surface-samples "${SURFACE_SAMPLES}" --seed "${SEED}" \
      --output "${output}"
  )

  if ! evaluation_complete "${output}" "${checkpoint}" "${pkl}"; then
    echo "Evaluation report failed validation: ${output}"
    exit 4
  fi

  model_elapsed=$((SECONDS - model_started))
  printf '[%d/3] COMPLETE: %s (%02dh:%02dm:%02ds)\n' \
    $((index + 1)) "${CURRENT_MODEL}" \
    $((model_elapsed / 3600)) $(((model_elapsed % 3600) / 60)) \
    $((model_elapsed % 60))
done

CURRENT_MODEL="none"
echo ""
echo "Context-dependence reports:"
for index in 0 1 2; do
  echo "- ${LABELS[$index]}"
  echo "  ${JOB_DIRS[$index]}/evaluation_final_700_context_dependence_50k.json"
done
