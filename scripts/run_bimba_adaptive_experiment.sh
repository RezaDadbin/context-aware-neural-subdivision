#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURAL="${ROOT}/external/neuralSubdiv"
NAME="bimba_adaptive_hybrid"
JOB="${NEURAL}/jobs/net_context_${NAME}"
TRAIN_PKL="${NEURAL}/data_PKL/style_bimba_head_train.pkl"
VALID_PKL="${NEURAL}/data_PKL/style_bimba_head_valid.pkl"
TEST_PKL="${NEURAL}/data_PKL/style_bimba_head_test.pkl"
CORE_START="${NEURAL}/jobs/net_context_bimba_pilot_hybrid/phase0_initial_state.dat"
EPOCHS="${EPOCHS:-700}"
DEVICE="${DEVICE:-cpu}"
SURFACE_SAMPLES="${SURFACE_SAMPLES:-50000}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
LOG_ROOT="${ROOT}/logs/bimba_adaptive_context"

if [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
fi

require_file() {
  [[ -s "$1" ]] || {
    echo "Missing required file: $1"
    exit 3
  }
}

for path in "${PYTHON_BIN}" "${TRAIN_PKL}" "${VALID_PKL}" "${TEST_PKL}" \
            "${CORE_START}"; do
  require_file "${path}"
done

mkdir -p "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/bimba_adaptive_${STAMP}.log"
START_SECONDS="${SECONDS}"
CURRENT_STAGE="configuration"
RESUME_COMMAND="cd \"${ROOT}\" && ./scripts/run_bimba_adaptive_experiment.sh"
exec > >(tee -a "${LOG_FILE}") 2>&1

finish() {
  status=$?
  elapsed=$((SECONDS - START_SECONDS))
  printf '\n============================================================\n'
  if [[ "${status}" -eq 0 ]]; then
    echo "BIMBA ADAPTIVE EXPERIMENT COMPLETED"
  else
    echo "BIMBA ADAPTIVE EXPERIMENT STOPPED (exit ${status})"
    echo "Current stage: ${CURRENT_STAGE}"
  fi
  printf 'Elapsed this session: %02dh:%02dm:%02ds\n' \
    $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
  echo "Log: ${LOG_FILE}"
  echo "Resume with: ${RESUME_COMMAND}"
  printf '============================================================\n'
}
trap finish EXIT

run_awake() {
  if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -dimsu "$@"
  else
    "$@"
  fi
}

echo "============================================================"
echo "BIMBA ADAPTIVE 1--6 DEPTH CONTEXT EXPERIMENT"
echo "Epochs: ${EPOCHS}"
echo "Device: ${DEVICE}"
echo "Validation: every epoch; best checkpoint retained"
echo "Training resumes from the latest completed epoch."
echo "Log: ${LOG_FILE}"
echo "============================================================"

"${PYTHON_BIN}" "${ROOT}/scripts/write_context_hyperparams.py" \
  "${NAME}" --train-pkl "${TRAIN_PKL}" --valid-pkl "${VALID_PKL}" \
  --epochs "${EPOCHS}" --seed 0 --device "${DEVICE}" --num-subd 2 \
  --local-layers 6 --global-layers 2 --global-chunk-size 256 \
  --adaptive-local --selector-hidden-dim 16 \
  --core-checkpoint "${CORE_START}"

CURRENT_STAGE="training"
echo "[1/3] Train or resume Adaptive V1 Bimba Phase 1"
run_awake env PYTHON="${PYTHON_BIN}" \
  "${ROOT}/scripts/train_context.sh" "${NAME}"
require_file "${JOB}/netparams.dat"

CURRENT_STAGE="evaluation"
echo "[2/3] Evaluate learned and diagnostic selector modes"
(
  cd "${NEURAL}"
  run_awake "${PYTHON_BIN}" -u evaluate_model.py \
    "${JOB}" "${TEST_PKL}" --model context \
    --modes context disabled shuffled uniform \
            fixed_1 fixed_2 fixed_3 fixed_4 fixed_5 fixed_6 \
    --device "${DEVICE}" --surface-samples "${SURFACE_SAMPLES}" --seed 0 \
    --output "${JOB}/evaluation_adaptive_test_50k.json" \
    --selector-output "${JOB}/adaptive_selector_weights.npz"
)

CURRENT_STAGE="summary"
echo "[3/3] Compare against the saved fixed-six-depth Phase 1"
"${PYTHON_BIN}" "${ROOT}/scripts/summarize_bimba_adaptive.py"

echo "Best model: ${JOB}/netparams.dat"
echo "Evaluation: ${JOB}/evaluation_adaptive_test_50k.json"
echo "Selector weights: ${JOB}/adaptive_selector_weights.npz"
echo "Comparison: ${ROOT}/artifacts/bimba_adaptive_context/RESULTS.md"
