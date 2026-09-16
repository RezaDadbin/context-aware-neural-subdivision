#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {bimba_head|fat_dragon|gear16} {phase0|phase1}"
  exit 2
fi

STYLE="$1"
PHASE="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURAL="${ROOT}/external/neuralSubdiv"
LOG_ROOT="${ROOT}/logs/final_training"
EPOCHS=700
SEED=0
DEVICE=cpu

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
if [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
fi

case "${STYLE}" in
  bimba_head)
    DISPLAY_NAME="Bimba Head"
    CONTEXT_JOB="bimba_pilot_hybrid"
    PHASE0_JOB="bimba_pilot_phase0"
    ;;
  fat_dragon)
    DISPLAY_NAME="Fat Dragon"
    CONTEXT_JOB="fat_dragon_final_hybrid"
    PHASE0_JOB="fat_dragon_final_phase0"
    ;;
  gear16)
    DISPLAY_NAME="Gear16"
    CONTEXT_JOB="gear16_final_hybrid"
    PHASE0_JOB="gear16_final_phase0"
    ;;
  *)
    echo "Unknown object: ${STYLE}"
    exit 2
    ;;
esac

TRAIN_PKL="${NEURAL}/data_PKL/style_${STYLE}_train.pkl"
VALID_PKL="${NEURAL}/data_PKL/style_${STYLE}_valid.pkl"
INITIAL_STATE="${NEURAL}/jobs/net_context_${CONTEXT_JOB}/phase0_initial_state.dat"

case "${PHASE}" in
  phase1)
    MODEL_LABEL="Phase 1 hybrid context"
    JOB_NAME="${CONTEXT_JOB}"
    JOB_DIR="${NEURAL}/jobs/net_context_${JOB_NAME}"
    TRAIN_SCRIPT="${ROOT}/scripts/train_context.sh"
    ;;
  phase0)
    MODEL_LABEL="Phase 0 core"
    JOB_NAME="${PHASE0_JOB}"
    JOB_DIR="${NEURAL}/jobs/net_style_${JOB_NAME}"
    TRAIN_SCRIPT="${ROOT}/scripts/train_style.sh"
    ;;
  *)
    echo "Unknown phase: ${PHASE}"
    exit 2
    ;;
esac

CHECKPOINT="${JOB_DIR}/checkpoint_latest.pt"
BEST_MODEL="${JOB_DIR}/netparams.dat"
mkdir -p "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/${STYLE}_${PHASE}_${STAMP}.log"
START_SECONDS="${SECONDS}"
RESUME_COMMAND="cd \"${ROOT}\" && ./scripts/train_all_six_models_700.sh"

exec > >(tee -a "${LOG_FILE}") 2>&1

checkpoint_summary() {
  if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "Checkpoint: none; this model will start from epoch 0."
    return
  fi

  "${PYTHON_BIN}" - "${CHECKPOINT}" <<'PY'
import sys
import torch

path = sys.argv[1]
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")

next_epoch = int(checkpoint.get("next_epoch", 0))
best_loss = float(checkpoint.get("best_loss", float("nan")))
completed = bool(checkpoint.get("completed", False))
print("Checkpoint: epoch %d, best validation %.6e, completed=%s" %
      (next_epoch, best_loss, completed))
PY
}

finish() {
  status=$?
  elapsed=$((SECONDS - START_SECONDS))
  printf '\n============================================================\n'
  if [[ "${status}" -eq 0 ]]; then
    echo "TRAINING COMMAND COMPLETED"
  else
    echo "TRAINING STOPPED (exit ${status})"
  fi
  printf 'Elapsed this session: %02dh:%02dm:%02ds\n' \
    $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
  checkpoint_summary || true
  echo "Best model: ${BEST_MODEL}"
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

require_file() {
  [[ -s "$1" ]] || {
    echo "Missing required file: $1"
    exit 3
  }
}

require_file "${TRAIN_PKL}"
require_file "${VALID_PKL}"
require_file "${PYTHON_BIN}"

if [[ "${PHASE}" == "phase0" ]]; then
  if [[ ! -s "${INITIAL_STATE}" ]]; then
    echo "Missing matching Phase 0 initial state: ${INITIAL_STATE}"
    echo "Run the ${DISPLAY_NAME} Phase 1 script first; it creates this file."
    exit 3
  fi

  "${PYTHON_BIN}" "${ROOT}/scripts/write_style_hyperparams.py" \
    "${JOB_NAME}" --train-pkl "${TRAIN_PKL}" --valid-pkl "${VALID_PKL}" \
    --epochs "${EPOCHS}" --seed "${SEED}" --device "${DEVICE}" \
    --initial-checkpoint "${INITIAL_STATE}"
else
  "${PYTHON_BIN}" "${ROOT}/scripts/write_context_hyperparams.py" \
    "${JOB_NAME}" --train-pkl "${TRAIN_PKL}" --valid-pkl "${VALID_PKL}" \
    --epochs "${EPOCHS}" --seed "${SEED}" --device "${DEVICE}" \
    --local-layers 6 --global-layers 2
fi

echo "============================================================"
echo "FINAL MODEL TRAINING"
echo "Object: ${DISPLAY_NAME}"
echo "Model: ${MODEL_LABEL}"
echo "Epoch target: ${EPOCHS} total"
echo "Device: ${DEVICE}"
echo "Seed: ${SEED}"
echo "Training data: ${TRAIN_PKL}"
echo "Validation data: ${VALID_PKL}"
echo "Job: ${JOB_DIR}"
echo "Log: ${LOG_FILE}"
echo "A checkpoint is saved after every completed epoch."
echo "Interrupt with Ctrl-C and rerun the same script to resume."
echo "============================================================"
checkpoint_summary

run_awake env PYTHON="${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${JOB_NAME}"
require_file "${BEST_MODEL}"
