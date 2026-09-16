#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURAL="${ROOT}/external/neuralSubdiv"
DATA_ROOT="${NEURAL}/data_meshes"
PKL_ROOT="${NEURAL}/data_PKL"
LOG_ROOT="${ROOT}/logs"

NUM_TRAIN="${NUM_TRAIN:-200}"
NUM_VALID="${NUM_VALID:-20}"
NUM_TEST="${NUM_TEST:-20}"
TARGET_FACES="${TARGET_FACES:-500}"
NUM_SUBD="${NUM_SUBD:-2}"
SMOKE_MESHES="${SMOKE_MESHES:-3}"
MAX_ATTEMPT_MULTIPLIER="${MAX_ATTEMPT_MULTIPLIER:-20}"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
if [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
fi

mkdir -p "${LOG_ROOT}" "${DATA_ROOT}" "${PKL_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/context_dataset_generation_${STAMP}.log"
START_SECONDS="${SECONDS}"

exec > >(tee -a "${LOG_FILE}") 2>&1

finish() {
  status=$?
  elapsed=$((SECONDS - START_SECONDS))
  printf '\n============================================================\n'
  if [[ "${status}" -eq 0 ]]; then
    echo "ALL DATASETS COMPLETED"
  else
    echo "DATASET PIPELINE FAILED (exit ${status})"
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

level_count() {
  local folder="$1"
  local level="$2"
  if [[ ! -d "${folder}/subd${level}" ]]; then
    echo 0
    return
  fi
  find "${folder}/subd${level}" -maxdepth 1 -type f -name '*.obj' 2>/dev/null |
    wc -l | tr -d ' '
}

split_complete() {
  local folder="$1"
  local expected="$2"
  local level
  [[ -d "${folder}" ]] || return 1
  for level in $(seq 0 "${NUM_SUBD}"); do
    [[ "$(level_count "${folder}" "${level}")" -eq "${expected}" ]] || return 1
  done
}

dataset_complete() {
  local style="$1"
  split_complete "${DATA_ROOT}/style_${style}_${NUM_TRAIN}" "${NUM_TRAIN}" &&
    split_complete "${DATA_ROOT}/style_${style}_valid_${NUM_VALID}" "${NUM_VALID}" &&
    split_complete "${DATA_ROOT}/style_${style}_test_${NUM_TEST}" "${NUM_TEST}" &&
    [[ -f "${DATA_ROOT}/style_${style}_seeds.tsv" ]]
}

dataset_artifact_exists() {
  local style="$1"
  [[ -e "${DATA_ROOT}/style_${style}_${NUM_TRAIN}" ||
     -e "${DATA_ROOT}/style_${style}_valid_${NUM_VALID}" ||
     -e "${DATA_ROOT}/style_${style}_test_${NUM_TEST}" ||
     -e "${DATA_ROOT}/style_${style}_seeds.tsv" ]]
}

pkl_complete() {
  local style="$1"
  local split
  for split in train valid test; do
    local pkl="${PKL_ROOT}/style_${style}_${split}.pkl"
    local validation="${PKL_ROOT}/style_${style}_${split}_validation.json"
    local smoke="${PKL_ROOT}/style_${style}_${split}_smoke.json"
    [[ -s "${pkl}" && -s "${validation}" && -s "${smoke}" ]] || return 1
    "${PYTHON_BIN}" - "${validation}" "${smoke}" <<'PY' || return 1
import json
import sys

with open(sys.argv[1]) as handle:
    validation = json.load(handle)
with open(sys.argv[2]) as handle:
    smoke = json.load(handle)
if validation.get("passed") is not True or smoke.get("status") != "passed":
    raise SystemExit(1)
PY
  done
}

print_summary() {
  local style="$1"
  local manifest="${DATA_ROOT}/style_${style}_seeds.tsv"
  local split folder expected accepted rejected
  echo "Status for ${style}:"
  for split in train valid test; do
    case "${split}" in
      train)
        folder="${DATA_ROOT}/style_${style}_${NUM_TRAIN}"
        expected="${NUM_TRAIN}"
        ;;
      valid)
        folder="${DATA_ROOT}/style_${style}_valid_${NUM_VALID}"
        expected="${NUM_VALID}"
        ;;
      test)
        folder="${DATA_ROOT}/style_${style}_test_${NUM_TEST}"
        expected="${NUM_TEST}"
        ;;
    esac
    accepted="$(level_count "${folder}" 0)"
    rejected=0
    if [[ -f "${manifest}" ]]; then
      rejected="$(awk -F '\t' -v target_split="${split}" \
        '$1 == target_split && $4 == "rejected" {count++} END {print count+0}' \
        "${manifest}")"
    fi
    echo "  ${split}: ${accepted}/${expected} accepted, ${rejected} rejected"
  done
}

run_object() {
  local position="$1"
  local total="$2"
  local style="$3"
  local source="$4"

  printf '\n============================================================\n'
  echo "OBJECT ${position}/${total}: ${style}"
  echo "Source: ${source}"
  echo "Hierarchy: ${TARGET_FACES} faces, ${NUM_SUBD} subdivision steps"
  echo "Splits: ${NUM_TRAIN} train / ${NUM_VALID} valid / ${NUM_TEST} test"
  printf '============================================================\n'

  [[ -f "${source}" ]] || {
    echo "Missing source: ${source}"
    return 10
  }

  if dataset_complete "${style}"; then
    echo "Mesh dataset is already complete; skipping generation."
  elif dataset_artifact_exists "${style}"; then
    echo "Incomplete final dataset artifacts exist for ${style}."
    echo "Move or remove those artifacts before rerunning; nothing was overwritten."
    return 11
  else
    PYTHON="${PYTHON_BIN}" \
    MAX_ATTEMPT_MULTIPLIER="${MAX_ATTEMPT_MULTIPLIER}" \
      run_awake "${ROOT}/scripts/generate_style_dataset.sh" \
        "${style}" "${source}" "${NUM_TRAIN}" "${NUM_VALID}" \
        "${TARGET_FACES}" "${NUM_SUBD}" "${NUM_TEST}"
  fi
  print_summary "${style}"

  if pkl_complete "${style}"; then
    echo "PKLs and verification reports already exist; skipping conversion."
  else
    echo "Building PKLs and running Phase 0/Phase 1 smoke tests..."
    run_awake "${PYTHON_BIN}" "${ROOT}/scripts/make_style_pkls.py" \
      "${style}" --num-train "${NUM_TRAIN}" --num-valid "${NUM_VALID}" \
      --num-test "${NUM_TEST}" --smoke-meshes "${SMOKE_MESHES}"
  fi
  echo "OBJECT ${position}/${total} COMPLETE: ${style}"
}

source_for_style() {
  case "$1" in
    bimba_head)
      echo "${ROOT}/data_candidates/compact_context/01_bimba_head.obj"
      ;;
    fat_dragon)
      echo "${ROOT}/data_candidates/compact_context/02_fat_dragon.obj"
      ;;
    gear16)
      echo "${ROOT}/data_candidates/compact_context/03_gear16.obj"
      ;;
    *)
      echo "Unknown object '$1'. Use: bimba_head, fat_dragon, or gear16." >&2
      return 2
      ;;
  esac
}

if (( $# == 0 )); then
  STYLES=(bimba_head fat_dragon gear16)
else
  STYLES=("$@")
fi

for style in "${STYLES[@]}"; do
  source_for_style "${style}" >/dev/null
done

echo "Context dataset pipeline started: $(date)"
echo "Log: ${LOG_FILE}"
df -h "${ROOT}" | tail -1

run_awake "${ROOT}/scripts/build_cxx_generator.sh"

position=0
total="${#STYLES[@]}"
for style in "${STYLES[@]}"; do
  position=$((position + 1))
  run_object "${position}" "${total}" "${style}" "$(source_for_style "${style}")"
done

printf '\nFinal verification artifacts:\n'
for style in "${STYLES[@]}"; do
  print_summary "${style}"
  echo "  PKL reports: ${PKL_ROOT}/style_${style}_*_validation.json"
  echo "  Smoke reports: ${PKL_ROOT}/style_${style}_*_smoke.json"
done
