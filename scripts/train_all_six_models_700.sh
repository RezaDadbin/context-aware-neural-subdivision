#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURAL="${ROOT}/external/neuralSubdiv"
LOG_ROOT="${ROOT}/logs/final_training"
TARGET_EPOCHS=700

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
if [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
fi
if [[ ! -s "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}"
  exit 3
fi

LABELS=(
  "Bimba Head - Phase 1 hybrid"
  "Bimba Head - Phase 0"
  "Fat Dragon - Phase 1 hybrid"
  "Fat Dragon - Phase 0"
  "Gear16 - Phase 1 hybrid"
  "Gear16 - Phase 0"
)
STYLES=(
  "bimba_head"
  "bimba_head"
  "fat_dragon"
  "fat_dragon"
  "gear16"
  "gear16"
)
PHASES=(
  "phase1"
  "phase0"
  "phase1"
  "phase0"
  "phase1"
  "phase0"
)
CHECKPOINTS=(
  "${NEURAL}/jobs/net_context_bimba_pilot_hybrid/checkpoint_latest.pt"
  "${NEURAL}/jobs/net_style_bimba_pilot_phase0/checkpoint_latest.pt"
  "${NEURAL}/jobs/net_context_fat_dragon_final_hybrid/checkpoint_latest.pt"
  "${NEURAL}/jobs/net_style_fat_dragon_final_phase0/checkpoint_latest.pt"
  "${NEURAL}/jobs/net_context_gear16_final_hybrid/checkpoint_latest.pt"
  "${NEURAL}/jobs/net_style_gear16_final_phase0/checkpoint_latest.pt"
)

mkdir -p "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/all_six_${STAMP}.log"
START_SECONDS="${SECONDS}"
CURRENT_MODEL="none"
RESUME_COMMAND="cd \"${ROOT}\" && ./scripts/train_all_six_models_700.sh"

exec > >(tee -a "${LOG_FILE}") 2>&1

checkpoint_status() {
  local label="$1"
  local checkpoint="$2"
  "${PYTHON_BIN}" - "${label}" "${checkpoint}" "${TARGET_EPOCHS}" <<'PY'
import os
import sys
import torch

label, path, target_text = sys.argv[1:]
target = int(target_text)
if not os.path.isfile(path) or os.path.getsize(path) == 0:
    print("%-34s not started (0/%d)" % (label, target))
    raise SystemExit(0)

try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")

epoch = int(checkpoint.get("next_epoch", 0))
best = float(checkpoint.get("best_loss", float("nan")))
done = bool(checkpoint.get("completed", False)) and epoch >= target
state = "complete" if done else "resume"
print("%-34s %-8s (%d/%d), best valid %.6e" %
      (label, state, epoch, target, best))
PY
}

is_complete() {
  local checkpoint="$1"
  "${PYTHON_BIN}" - "${checkpoint}" "${TARGET_EPOCHS}" <<'PY'
import os
import sys
import torch

path, target_text = sys.argv[1:]
if not os.path.isfile(path) or os.path.getsize(path) == 0:
    raise SystemExit(1)
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")
done = (bool(checkpoint.get("completed", False)) and
        int(checkpoint.get("next_epoch", 0)) >= int(target_text))
raise SystemExit(0 if done else 1)
PY
}

print_all_statuses() {
  local index
  for index in 0 1 2 3 4 5; do
    checkpoint_status "${LABELS[$index]}" "${CHECKPOINTS[$index]}"
  done
}

finish() {
  status=$?
  elapsed=$((SECONDS - START_SECONDS))
  printf '\n============================================================\n'
  if [[ "${status}" -eq 0 ]]; then
    echo "ALL SIX TRAINING JOBS COMPLETED"
  else
    echo "TRAINING SEQUENCE STOPPED (exit ${status})"
    echo "Interrupted/current model: ${CURRENT_MODEL}"
  fi
  printf 'Elapsed this session: %02dh:%02dm:%02ds\n' \
    $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
  echo "Aggregate log: ${LOG_FILE}"
  echo "Resume with: ${RESUME_COMMAND}"
  printf '============================================================\n'
}
trap finish EXIT

echo "============================================================"
echo "SIX-MODEL FINAL TRAINING SEQUENCE"
echo "Target: ${TARGET_EPOCHS} total epochs per model"
echo "Order: Phase 1 then matching Phase 0 for each object"
echo "Checkpoint: saved after every completed epoch"
echo "Completed models are skipped automatically on every restart."
echo "Log: ${LOG_FILE}"
echo "Resume after interruption with:"
echo "${RESUME_COMMAND}"
echo "============================================================"
print_all_statuses

for index in 0 1 2 3 4 5; do
  CURRENT_MODEL="${LABELS[$index]}"
  if is_complete "${CHECKPOINTS[$index]}"; then
    echo ""
    echo "[$((index + 1))/6] SKIP complete: ${CURRENT_MODEL}"
    continue
  fi

  echo ""
  echo "[$((index + 1))/6] START/RESUME: ${CURRENT_MODEL}"
  checkpoint_status "${CURRENT_MODEL}" "${CHECKPOINTS[$index]}"
  PYTHON_BIN="${PYTHON_BIN}" "${ROOT}/scripts/run_final_model.sh" \
    "${STYLES[$index]}" "${PHASES[$index]}"

  if ! is_complete "${CHECKPOINTS[$index]}"; then
    echo "Model returned without reaching ${TARGET_EPOCHS} epochs: ${CURRENT_MODEL}"
    exit 4
  fi
  echo "[$((index + 1))/6] COMPLETE: ${CURRENT_MODEL}"
done

CURRENT_MODEL="none"
echo ""
echo "Final checkpoint status:"
print_all_statuses
