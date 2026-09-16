#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURAL="${ROOT}/external/neuralSubdiv"
LOG_ROOT="${ROOT}/logs/final_evaluation"
SURFACE_SAMPLES=50000
EXPECTED_MESHES=20
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

LABELS=(
  "Bimba Head - Phase 0"
  "Bimba Head - Phase 1 hybrid"
  "Fat Dragon - Phase 0"
  "Fat Dragon - Phase 1 hybrid"
  "Gear16 - Phase 0"
  "Gear16 - Phase 1 hybrid"
)
MODEL_TYPES=(
  "phase0"
  "context"
  "phase0"
  "context"
  "phase0"
  "context"
)
JOB_DIRS=(
  "${NEURAL}/jobs/net_style_bimba_pilot_phase0"
  "${NEURAL}/jobs/net_context_bimba_pilot_hybrid"
  "${NEURAL}/jobs/net_style_fat_dragon_final_phase0"
  "${NEURAL}/jobs/net_context_fat_dragon_final_hybrid"
  "${NEURAL}/jobs/net_style_gear16_final_phase0"
  "${NEURAL}/jobs/net_context_gear16_final_hybrid"
)
TEST_PKLS=(
  "${NEURAL}/data_PKL/style_bimba_head_test.pkl"
  "${NEURAL}/data_PKL/style_bimba_head_test.pkl"
  "${NEURAL}/data_PKL/style_fat_dragon_test.pkl"
  "${NEURAL}/data_PKL/style_fat_dragon_test.pkl"
  "${NEURAL}/data_PKL/style_gear16_test.pkl"
  "${NEURAL}/data_PKL/style_gear16_test.pkl"
)

mkdir -p "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/all_six_test_50k_${STAMP}.log"
START_SECONDS="${SECONDS}"
CURRENT_MODEL="none"
RESUME_COMMAND="cd \"${ROOT}\" && ./scripts/evaluate_all_six_final_models.sh"

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
  local model_type="$4"

  "${PYTHON_BIN}" - "${output}" "${checkpoint}" "${pkl}" \
    "${model_type}" "${SURFACE_SAMPLES}" "${EXPECTED_MESHES}" <<'PY'
import json
import os
import sys

output, checkpoint, pkl, model_type, samples, meshes = sys.argv[1:]
if not os.path.isfile(output) or os.path.getsize(output) == 0:
    raise SystemExit(1)
if os.path.getmtime(output) < max(os.path.getmtime(checkpoint), os.path.getmtime(pkl)):
    raise SystemExit(1)
try:
    with open(output, "r") as handle:
        report = json.load(handle)
except (OSError, ValueError):
    raise SystemExit(1)

expected_records = int(meshes) * 3
valid = (
    report.get("model") == model_type and
    os.path.realpath(report.get("checkpoint", "")) == os.path.realpath(checkpoint) and
    os.path.realpath(report.get("pkl", "")) == os.path.realpath(pkl) and
    int(report.get("surface_samples_per_mesh", -1)) == int(samples) and
    int(report.get("meshes_evaluated", -1)) == int(meshes) and
    len(report.get("records", [])) == expected_records
)
raise SystemExit(0 if valid else 1)
PY
}

finish() {
  status=$?
  elapsed=$((SECONDS - START_SECONDS))
  printf '\n============================================================\n'
  if [[ "${status}" -eq 0 ]]; then
    echo "ALL SIX TEST EVALUATIONS COMPLETED"
  else
    echo "EVALUATION SEQUENCE STOPPED (exit ${status})"
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
echo "FINAL SIX-MODEL TEST EVALUATION"
echo "Models: best-validation netparams.dat checkpoints"
echo "Test meshes: ${EXPECTED_MESHES} per model"
echo "Surface samples: ${SURFACE_SAMPLES} per mesh and level"
echo "Device: ${DEVICE}"
echo "Seed: ${SEED}"
echo "Completed valid reports are skipped when this command is rerun."
echo "Log: ${LOG_FILE}"
echo "============================================================"

for index in 0 1 2 3 4 5; do
  CURRENT_MODEL="${LABELS[$index]}"
  job="${JOB_DIRS[$index]}"
  pkl="${TEST_PKLS[$index]}"
  model_type="${MODEL_TYPES[$index]}"
  checkpoint="${job}/netparams.dat"
  output="${job}/evaluation_final_700_test_50k.json"

  require_file "${job}/hyperparameters.json"
  require_file "${checkpoint}"
  require_file "${pkl}"

  if evaluation_complete "${output}" "${checkpoint}" "${pkl}" "${model_type}"; then
    echo "[$((index + 1))/6] SKIP complete: ${CURRENT_MODEL}"
    echo "Report: ${output}"
    continue
  fi

  echo ""
  echo "[$((index + 1))/6] START: ${CURRENT_MODEL}"
  echo "Checkpoint: ${checkpoint}"
  echo "Test data: ${pkl}"
  echo "Report: ${output}"
  model_started="${SECONDS}"

  if [[ "${model_type}" == "context" ]]; then
    (
      cd "${NEURAL}"
      run_awake "${PYTHON_BIN}" -u evaluate_model.py \
        "${job}" "${pkl}" --model context --modes context \
        --device "${DEVICE}" --surface-samples "${SURFACE_SAMPLES}" \
        --seed "${SEED}" --output "${output}"
    )
  else
    (
      cd "${NEURAL}"
      run_awake "${PYTHON_BIN}" -u evaluate_model.py \
        "${job}" "${pkl}" --model phase0 \
        --device "${DEVICE}" --surface-samples "${SURFACE_SAMPLES}" \
        --seed "${SEED}" --output "${output}"
    )
  fi

  if ! evaluation_complete "${output}" "${checkpoint}" "${pkl}" "${model_type}"; then
    echo "Evaluation report failed validation: ${output}"
    exit 4
  fi

  model_elapsed=$((SECONDS - model_started))
  printf '[%d/6] COMPLETE: %s (%02dh:%02dm:%02ds)\n' \
    $((index + 1)) "${CURRENT_MODEL}" \
    $((model_elapsed / 3600)) $(((model_elapsed % 3600) / 60)) \
    $((model_elapsed % 60))
done

CURRENT_MODEL="none"
echo ""
echo "Evaluation reports:"
for index in 0 1 2 3 4 5; do
  echo "- ${LABELS[$index]}"
  echo "  ${JOB_DIRS[$index]}/evaluation_final_700_test_50k.json"
done
