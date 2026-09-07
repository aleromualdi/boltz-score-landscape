#!/usr/bin/env bash
# Full PSCDB panel: predict → soft CA modes → per-system AUROC → summarize.
set -u
export PYTHONUNBUFFERED=1

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "${EXP}/../.." && pwd)"
EXP1="${ROOT}/experiments/1_codnas_apo_holo"
BOLTZ="${ROOT}/boltz/.venv/bin/boltz"
BOLTZ_PY="${ROOT}/boltz/.venv/bin/python"

PANEL="${EXP}/inputs/panel_full.json"
PRED_PARENT="${EXP}/outputs/predict"
MODE_DIR="${EXP}/outputs/modes"
PER_SYS="${EXP}/outputs/per_system"
LOG_DIR="${EXP}/outputs/logs"
RUN_LOG="${LOG_DIR}/campaign_run.log"
FAIL_LOG="${LOG_DIR}/campaign_failures.log"

mkdir -p "${LOG_DIR}" "${PRED_PARENT}" "${MODE_DIR}" "${PER_SYS}"
cd "${ROOT}/boltz"

log() { echo "[$(date -Is)] $*" | tee -a "${RUN_LOG}"; }

log "=== Exp 3 campaign start ==="

if [[ ! -f "${PANEL}" ]]; then
  log "curate panel"
  PYTHONPATH="${EXP}/scripts" "${BOLTZ_PY}" "${EXP}/scripts/curate_panel.py" \
    --max_len 1000 --min_len 40 --seed 204 \
    >> "${RUN_LOG}" 2>&1 \
    || { log "curate FAILED"; exit 1; }
else
  log "reuse existing panel ${PANEL}"
fi

log "prepare yamls"
PYTHONPATH="${EXP}/scripts" "${BOLTZ_PY}" "${EXP}/scripts/prepare_inputs.py" \
  >> "${RUN_LOG}" 2>&1 \
  || { log "prepare FAILED"; exit 1; }

mapfile -t IDS < <("${BOLTZ_PY}" -c "
import json
from pathlib import Path
for r in json.loads(Path('${PANEL}').read_text()):
    print(r['system'])
")
log "panel n=${#IDS[@]}"

: > "${FAIL_LOG}"
done_n=0
for system in "${IDS[@]}"; do
  yaml="${EXP}/inputs/yamls/${system}.yaml"
  pred="${PRED_PARENT}/${system}"
  result_json="${PER_SYS}/${system}.json"
  mode_npz="${MODE_DIR}/${system}_ca_modes.npz"
  mkdir -p "${pred}"

  if [[ -f "${result_json}" ]]; then
    log "skip done ${system}"
    done_n=$((done_n + 1))
    continue
  fi

  if ! compgen -G "${pred}/boltz_results_*/processed/structures/${system}.npz" > /dev/null; then
    log "predict ${system}"
    if ! "${BOLTZ}" predict "${yaml}" \
      --out_dir "${pred}" \
      --devices 1 \
      --seed 0 \
      --sampling_steps 50 \
      --diffusion_samples 1 \
      --use_msa_server \
      --subsample_msa \
      --num_subsampled_msa 1024 \
      >> "${RUN_LOG}" 2>&1; then
      echo "predict failed ${system}" | tee -a "${FAIL_LOG}"
      continue
    fi
  else
    log "reuse predict ${system}"
  fi

  if [[ ! -f "${mode_npz}" ]]; then
    log "compute modes ${system}"
    if ! PYTHONPATH="${EXP}/scripts:${EXP1}/scripts:${ROOT}/src" "${BOLTZ_PY}" \
      "${EXP}/scripts/compute_modes.py" \
      --system "${system}" \
      --predict_root "${pred}" \
      >> "${RUN_LOG}" 2>&1; then
      echo "modes failed ${system}" | tee -a "${FAIL_LOG}"
      continue
    fi
  else
    log "reuse modes ${system}"
  fi

  log "evaluate ${system}"
  if ! PYTHONPATH="${EXP}/scripts:${EXP1}/scripts:${ROOT}/src" \
    "${BOLTZ_PY}" "${EXP}/scripts/evaluate_retrieval.py" \
      --system "${system}" \
      --predict_root "${pred}" \
      >> "${RUN_LOG}" 2>&1; then
    echo "eval failed ${system}" | tee -a "${FAIL_LOG}"
    continue
  fi
  done_n=$((done_n + 1))
done

log "summarize (${done_n} systems touched/done)"
PYTHONPATH="${EXP}/scripts" "${BOLTZ_PY}" "${EXP}/scripts/summarize_campaign.py" \
  >> "${RUN_LOG}" 2>&1 || log "summarize failed (partial results?)"

log "=== Exp 3 campaign finished ==="
log "failures: ${FAIL_LOG}"
log "summary: ${EXP}/outputs/analysis/SUMMARY.md"
