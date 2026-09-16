#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 STYLE_NAME SOURCE_MESH [NUM_TRAIN=200] [NUM_VALID=20] [TARGET_FACES=500] [NUM_SUBD=2] [NUM_TEST=20]"
  echo "Example: $0 bunny data/style_sources/bunny.obj 200 20 500 2 20"
  exit 2
fi

STYLE="$1"
SOURCE_OBJ="$2"
NUM_TRAIN="${3:-200}"
NUM_VALID="${4:-20}"
TARGET_FACES="${5:-500}"
NUM_SUBD="${6:-2}"
NUM_TEST="${7:-20}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURAL="${ROOT}/external/neuralSubdiv"
GEN_DIR="${ROOT}/external/surface_multigrid_code/09_random_subdiv_remesh"
BIN="${GEN_DIR}/build/random_subdiv_remesh_bin"
SEED_MANIFEST="${NEURAL}/data_meshes/style_${STYLE}_seeds.tsv"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
VALIDATOR="${NEURAL}/validate_generated_chain.py"
MAX_ATTEMPT_MULTIPLIER="${MAX_ATTEMPT_MULTIPLIER:-20}"

if [[ ! "${STYLE}" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "STYLE_NAME may contain only letters, numbers, underscores, and hyphens."
  exit 2
fi
for value in "${NUM_TRAIN}" "${NUM_VALID}" "${NUM_TEST}" \
             "${TARGET_FACES}" "${NUM_SUBD}" "${MAX_ATTEMPT_MULTIPLIER}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Counts, face target, subdivisions, and attempt multiplier must be positive integers."
    exit 2
  fi
done

if [[ ! -x "${BIN}" ]]; then
  echo "Generator binary not found. Run scripts/build_cxx_generator.sh first."
  exit 3
fi

SOURCE_ABS="$(cd "$(dirname "${SOURCE_OBJ}")" && pwd)/$(basename "${SOURCE_OBJ}")"
if [[ ! -f "${SOURCE_ABS}" ]]; then
  echo "Missing source OBJ: ${SOURCE_OBJ}"
  exit 4
fi

TRAIN_ROOT="${NEURAL}/data_meshes/style_${STYLE}_${NUM_TRAIN}"
VALID_ROOT="${NEURAL}/data_meshes/style_${STYLE}_valid_${NUM_VALID}"
TEST_ROOT="${NEURAL}/data_meshes/style_${STYLE}_test_${NUM_TEST}"
for path in "${TRAIN_ROOT}" "${VALID_ROOT}" "${TEST_ROOT}" "${SEED_MANIFEST}"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite existing dataset artifact: ${path}"
    exit 5
  fi
done

mkdir -p "${NEURAL}/data_meshes"
RUN_TMP="$(mktemp -d "${NEURAL}/data_meshes/.style_${STYLE}.XXXXXX")"
MANIFEST_TMP="${RUN_TMP}/seeds.tsv"
trap 'rm -rf "${RUN_TMP}"' EXIT

make_split() {
  local split_name="$1"
  local count="$2"
  local seed_offset="$3"
  local out_root="${RUN_TMP}/${split_name}"
  local accepted=0
  local rejected=0
  local attempted=0
  local split_start="${SECONDS}"
  local max_attempts=$((count * MAX_ATTEMPT_MULTIPLIER + 20))
  local seed elapsed percent idx reason
  local -a paths

  mkdir -p "${out_root}"
  for level in $(seq 0 "${NUM_SUBD}"); do
    mkdir -p "${out_root}/subd${level}"
  done

  echo
  echo "Generating ${split_name}: ${count} chains -> ${out_root}"
  while (( accepted < count )); do
    attempted=$((attempted + 1))
    if (( attempted > max_attempts )); then
      echo "Failed to collect ${count} valid ${split_name} chains after ${max_attempts} attempts"
      exit 6
    fi
    seed=$((seed_offset + attempted))
    elapsed=$((SECONDS - split_start))
    percent=$((accepted * 100 / count))
    printf '[%s] [%s %s] accepted=%d/%d (%d%%) rejected=%d attempt=%d seed=%d elapsed=%02d:%02d:%02d - still generating\n' \
      "$(date '+%H:%M:%S')" "${STYLE}" "${split_name}" \
      "${accepted}" "${count}" "${percent}" "${rejected}" \
      "${attempted}" "${seed}" \
      $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
    (
      cd "${GEN_DIR}/build"
      ./random_subdiv_remesh_bin "${SOURCE_ABS}" "${TARGET_FACES}" "${NUM_SUBD}" "${seed}"
    )
    paths=()
    for level in $(seq 0 "${NUM_SUBD}"); do
      paths+=("${GEN_DIR}/output_s${level}.obj")
    done
    if reason="$("${PYTHON_BIN}" "${VALIDATOR}" \
        --expected-start-faces "${TARGET_FACES}" "${paths[@]}" 2>&1)"; then
      accepted=$((accepted + 1))
      idx="$(printf "%03d" "${accepted}")"
      for level in $(seq 0 "${NUM_SUBD}"); do
        cp "${GEN_DIR}/output_s${level}.obj" "${out_root}/subd${level}/${idx}.obj"
      done
      printf "%s\t%s\t%s\taccepted\t%s\n" \
        "${split_name}" "${idx}" "${seed}" "${reason}" >> "${MANIFEST_TMP}"
      percent=$((accepted * 100 / count))
      echo "accepted seed ${seed}: ${split_name} ${accepted}/${count} (${percent}%)"
    else
      rejected=$((rejected + 1))
      reason="$(printf '%s' "${reason}" | tr '\n\t' '  ')"
      printf "%s\t\t%s\trejected\t%s\n" \
        "${split_name}" "${seed}" "${reason}" >> "${MANIFEST_TMP}"
      echo "rejected seed ${seed}: ${reason}"
    fi
  done
  elapsed=$((SECONDS - split_start))
  printf '[%s] [%s %s] COMPLETE: %d/%d accepted, %d rejected, elapsed=%02d:%02d:%02d\n' \
    "$(date '+%H:%M:%S')" "${STYLE}" "${split_name}" \
    "${accepted}" "${count}" "${rejected}" \
    $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
}

echo "============================================================"
echo "Generating normalized Neural Subdivision style dataset"
echo "Style:        ${STYLE}"
echo "Source OBJ:   ${SOURCE_ABS}"
echo "Train/valid/test: ${NUM_TRAIN}/${NUM_VALID}/${NUM_TEST}"
echo "Target faces: ${TARGET_FACES}"
echo "Subdivisions: ${NUM_SUBD}"
echo "============================================================"

printf "split\tindex\tseed\tstatus\treason\n" > "${MANIFEST_TMP}"
make_split train "${NUM_TRAIN}" 0
make_split valid "${NUM_VALID}" 100000
make_split test "${NUM_TEST}" 200000

mv "${RUN_TMP}/train" "${TRAIN_ROOT}"
mv "${RUN_TMP}/valid" "${VALID_ROOT}"
mv "${RUN_TMP}/test" "${TEST_ROOT}"
mv "${MANIFEST_TMP}" "${SEED_MANIFEST}"

echo
echo "Done. Generated folders:"
echo "- ${TRAIN_ROOT}"
echo "- ${VALID_ROOT}"
echo "- ${TEST_ROOT}"
echo "- ${SEED_MANIFEST}"
