#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURAL="${ROOT}/external/neuralSubdiv"
LOG_ROOT="${ROOT}/logs"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cpu}"
TRAIN_PKL="${NEURAL}/data_PKL/style_bimba_head_train.pkl"
VALID_PKL="${NEURAL}/data_PKL/style_bimba_head_valid.pkl"
HYBRID_NAME="bimba_pilot_hybrid"
LOCAL_NAME="bimba_pilot_local"
PHASE0_NAME="bimba_pilot_phase0"
INITIAL_STATE="${NEURAL}/jobs/net_context_${HYBRID_NAME}/phase0_initial_state.dat"

mkdir -p "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/bimba_pilot_${STAMP}.log"
START_SECONDS="${SECONDS}"
exec > >(tee -a "${LOG_FILE}") 2>&1

finish() {
  status=$?
  elapsed=$((SECONDS - START_SECONDS))
  printf '\n============================================================\n'
  if [[ "${status}" -eq 0 ]]; then
    echo "BIMBA PILOT TRAINING COMPLETED"
  else
    echo "BIMBA PILOT STOPPED (exit ${status})"
    echo "Run the same command again to resume completed epochs."
  fi
  printf 'Elapsed: %02dh:%02dm:%02ds\n' \
    $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
  echo "Log: ${LOG_FILE}"
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

require_file() {
  [[ -s "$1" ]] || {
    echo "Missing required file: $1"
    exit 3
  }
}

require_file "${TRAIN_PKL}"
require_file "${VALID_PKL}"
require_file "${PYTHON_BIN}"

echo "============================================================"
echo "Bimba controlled pilot"
echo "Models: hybrid context, Phase 0, local-only context"
echo "Epochs: ${EPOCHS} each"
echo "Device: ${DEVICE}"
echo "Seed: ${SEED}"
echo "Log: ${LOG_FILE}"
echo "Run this command again after interruption to resume."
echo "============================================================"

"${PYTHON_BIN}" "${ROOT}/scripts/write_context_hyperparams.py" \
  "${HYBRID_NAME}" --train-pkl "${TRAIN_PKL}" --valid-pkl "${VALID_PKL}" \
  --epochs "${EPOCHS}" --seed "${SEED}" --device "${DEVICE}" \
  --local-layers 6 --global-layers 2

echo "[1/3] Training Phase 1 hybrid..."
run_awake env PYTHON="${PYTHON_BIN}" \
  "${ROOT}/scripts/train_context.sh" "${HYBRID_NAME}"
require_file "${INITIAL_STATE}"

"${PYTHON_BIN}" "${ROOT}/scripts/write_style_hyperparams.py" \
  "${PHASE0_NAME}" --train-pkl "${TRAIN_PKL}" --valid-pkl "${VALID_PKL}" \
  --epochs "${EPOCHS}" --seed "${SEED}" --device "${DEVICE}" \
  --initial-checkpoint "${INITIAL_STATE}"

echo "[2/3] Training matching Phase 0..."
run_awake env PYTHON="${PYTHON_BIN}" \
  "${ROOT}/scripts/train_style.sh" "${PHASE0_NAME}"

"${PYTHON_BIN}" "${ROOT}/scripts/write_context_hyperparams.py" \
  "${LOCAL_NAME}" --train-pkl "${TRAIN_PKL}" --valid-pkl "${VALID_PKL}" \
  --epochs "${EPOCHS}" --seed "${SEED}" --device "${DEVICE}" \
  --local-layers 6 --global-layers 0 --core-checkpoint "${INITIAL_STATE}"

echo "[3/3] Training Phase 1 local-only ablation..."
run_awake env PYTHON="${PYTHON_BIN}" \
  "${ROOT}/scripts/train_context.sh" "${LOCAL_NAME}"

echo "Pilot checkpoints:"
echo "- ${NEURAL}/jobs/net_context_${HYBRID_NAME}/netparams.dat"
echo "- ${NEURAL}/jobs/net_style_${PHASE0_NAME}/netparams.dat"
echo "- ${NEURAL}/jobs/net_context_${LOCAL_NAME}/netparams.dat"
