#!/usr/bin/env bash
# Full CoDNaS panel (91 pairs): soft-subspace enrichment with n_null=500.
set -u
export PYTHONUNBUFFERED=1

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "${EXP}/../.." && pwd)"
SRC="${ROOT}/src"
BOLTZ="${ROOT}/boltz/.venv/bin/boltz"
BOLTZ_PY="${ROOT}/boltz/.venv/bin/python"

TAG="soft91"
N_NULL=500
MAX_LEN=1500
PRED_PARENT="${EXP}/outputs/predict_${TAG}"
OUT="${EXP}/outputs/analysis_${TAG}"
FAIL_LOG="${EXP}/outputs/logs/${TAG}_failures.log"
mkdir -p "${EXP}/outputs/logs" "${OUT}" "${PRED_PARENT}"
: > "${FAIL_LOG}"
cd "${ROOT}/boltz"

echo "[$(date -Is)] prepare ${TAG} (all pairs, length < ${MAX_LEN})"
PYTHONPATH="${EXP}/scripts" "${BOLTZ_PY}" "${EXP}/scripts/prepare_inputs.py" \
  || { echo "prepare failed" | tee -a "${FAIL_LOG}"; exit 1; }

PANEL="${EXP}/inputs/panel91.json"
mapfile -t IDS < <("${BOLTZ_PY}" -c "
import json
from pathlib import Path
recs = json.loads(Path('${PANEL}').read_text())
for r in recs:
    if int(r['n_res']) < ${MAX_LEN}:
        print(r['apo_id'])
")
echo "[$(date -Is)] ${TAG} n=${#IDS[@]}  n_null=${N_NULL}"

find_cached_pred() {
  local apo="$1"
  local candidate="${PRED_PARENT}/${apo}"
  if compgen -G "${candidate}/boltz_results_*/processed/structures/${apo}.npz" > /dev/null; then
    echo "${candidate}"
    return 0
  fi
  return 1
}

for apo in "${IDS[@]}"; do
  yaml="${EXP}/inputs/yamls/${apo}.yaml"
  pred="${PRED_PARENT}/${apo}"
  nres="$("${BOLTZ_PY}" -c "import json; recs=json.load(open('${PANEL}')); print(next(r['n_res'] for r in recs if r['apo_id']=='${apo}'))")"
  mkdir -p "${pred}"

  if cached="$(find_cached_pred "${apo}")"; then
    if [[ "${cached}" != "${pred}" ]]; then
      echo "reuse cached predict ${apo} from ${cached}"
      pred="${cached}"
    else
      echo "skip predict ${apo}"
    fi
  else
    echo "[$(date -Is)] predict ${apo} n_res=${nres} msa=1024"
    if ! "${BOLTZ}" predict "${yaml}" \
      --out_dir "${pred}" \
      --devices 1 \
      --seed 0 \
      --sampling_steps 50 \
      --diffusion_samples 1 \
      --use_msa_server \
      --subsample_msa \
      --num_subsampled_msa 1024; then
      echo "predict failed ${apo}" | tee -a "${FAIL_LOG}"
      continue
    fi
  fi

  proc="$(ls -d "${pred}"/boltz_results_*/processed 2>/dev/null | head -n1)"
  if [[ -z "${proc}" || ! -f "${proc}/structures/${apo}.npz" ]]; then
    echo "no processed ${apo}" | tee -a "${FAIL_LOG}"
    continue
  fi

  echo "[$(date -Is)] soft enrichment ${apo} n_null=${N_NULL}"
  if ! PYTHONPATH="${SRC}:${EXP}/scripts" "${BOLTZ_PY}" \
    "${EXP}/scripts/directional_curvature.py" \
    --system "${apo}" \
    --processed_dir "${proc}" \
    --predict_root "${pred}" \
    --out_dir "${OUT}" \
    --n_null "${N_NULL}" \
    --force; then
    echo "soft enrichment failed ${apo}" | tee -a "${FAIL_LOG}"
    continue
  fi
done

PYTHONPATH="${EXP}/scripts" "${BOLTZ_PY}" "${EXP}/scripts/summarize_soft.py" "${OUT}" || true
echo "[$(date -Is)] ${TAG} finished  out=${OUT}"
