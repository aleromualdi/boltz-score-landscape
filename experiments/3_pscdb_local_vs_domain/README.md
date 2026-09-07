# Experiment 3 — PSCDB local vs domain (Boltz vs ANM)

## Claim

Blind retrieval of PSCDB Moving-segment residues from soft-mode participation scores on a single predicted structure. The primary metric is AUROC comparing Boltz score-Hessian soft modes against classical ANM on the same predicted Cα geometry. The scientific contrast is whether Boltz beats ANM on local motions (IL+CL) more than on domain motions (CD+ID).

## Results (AUROC by subclass)

219 OK systems. Tables report median AUROC and Δ = Boltz − ANM with 95% bootstrap CIs.


| subclass                | n   | med AUROC Boltz | med AUROC ANM | med ΔAUROC [95% CI]         |
| ----------------------- | --- | --------------- | ------------- | --------------------------- |
| IL (coupled local)      | 78  | 0.905           | 0.818         | +0.033 [+0.015, +0.074]     |
| CL (independent local)  | 68  | 0.894           | 0.704         | +0.133 [+0.092, +0.184]     |
| CD (coupled domain)     | 31  | 0.539           | 0.571         | −0.049 [−0.137, −0.016]     |
| ID (independent domain) | 42  | 0.624           | 0.668         | −0.046 [−0.138, +0.006]     |
| **IL+CL (local)**       | 146 | 0.901           | 0.771         | **+0.086** [+0.044, +0.129] |
| **CD+ID (domain)**      | 73  | 0.584           | 0.601         | **−0.049** [−0.132, −0.016] |


Local − domain contrast on median ΔAUROC: **+0.136** [+0.077, +0.218]. Panel: `outputs/analysis/retrieval_panel.csv`. Narrative summary: `outputs/analysis/SUMMARY.md`.

## How AUROC is computed

Per system, every Cα is one binary classification example. Ground truth maps PSCDB Moving-segment / moving-domain residue ranges onto the predicted chain by sequence alignment; a residue is labeled mobile (1) if it falls in an annotated moving range, else immobile (0). Scores are Δx-blind and use the same predicted Cα geometry. For Boltz, the residue score S_i is the soft-mode participation: the sum of squared Cα amplitudes of that residue over the ten softest score-Hessian modes (`top10_lambda`, equal weights). For ANM, the same participation is computed on the ten softest non-rigid ANM modes (cutoff 15 Å, γ=1). AUROC treats S_i as a ranking score for retrieving mobile residues (Mann–Whitney U with midranks for ties; chance = 0.5). ΔAUROC is AUROC(Boltz) − AUROC(ANM) per system; tables report subclass medians with bootstrap 95% CIs.

Recipe lock: mode selection `top10_lambda`; Boltz score-ascent pose with autodiff soft CA modes (σ ≈ 0.28 Å); ANM cutoff 15 Å, γ=1 on predicted CA; labels from PSCDB Moving segment / moving domain.

## Reproduce

shared helpers in `../../src/`, and Exp 1 scripts under `../1_codnas_apo_holo/scripts` (pose/Hessian utilities imported by `compute_modes.py`). Network is required for ColabFold MSA download during `boltz predict` and for RCSB mmCIF download during panel curation.

### 1. Get the data

Place the Protein Structural Change Database (PSCDB) dump under `data/raw/`:

```bash
cd experiments/3_pscdb_local_vs_domain
mkdir -p data/raw data/structures

# From the PSCDB distribution (or unpack the local archives if present):
#   data/raw/pscdb.csv
#   data/raw/pscdb_pscid_list.csv
unzip -o data/raw/pscdb.zip -d data/raw/            # if you have the zip
unzip -o data/raw/pscdb_pscid_list.zip -d data/raw/  # if you have the zip
```

Expected: `data/raw/pscdb.csv` (full annotations including Moving segment) and `data/raw/pscdb_pscid_list.csv` (PSCID index used by curation). Free/bound structures are downloaded from RCSB into `data/structures/` by `curate_panel.py` (not checked in).

### 2. Parse / prepare inputs

`curate_panel.py` filters CL/IL/CD/ID entries to single-chain free/bound pairs in a length window, downloads mmCIFs, extracts sequences, and writes the panel. `prepare_inputs.py` then emits Boltz YAMLs:

```bash
cd experiments/3_pscdb_local_vs_domain
python scripts/curate_panel.py --max_len 1000 --min_len 40 --seed 204
```

Writes `inputs/panel_full.json` / `inputs/panel_full.csv` and `inputs/yamls/{system}.yaml`. (`run_campaign.sh` runs these automatically if the panel is missing.)

### 3. Run the experiment

Full campaign (predict → score-ascent soft CA modes → per-system AUROC → subclass aggregate):

```bash
cd experiments/3_pscdb_local_vs_domain
bash scripts/run_campaign.sh
```

Per system the script reuses existing predict/mode/eval artifacts when present; otherwise it runs `boltz predict` (seed 0, 50 steps, MSA server, subsample MSA 1024), `compute_modes.py`, and `evaluate_retrieval.py`. At the end `summarize_campaign.py` writes `outputs/analysis/retrieval_panel.csv` and `outputs/analysis/SUMMARY.md`. Logs: `outputs/logs/campaign_run.log`, failures in `outputs/logs/campaign_failures.log`.

To re-aggregate from existing `outputs/per_system/*.json` without re-running predicts:

```bash
cd experiments/3_pscdb_local_vs_domain
python scripts/summarize_campaign.py
```



### Outputs to check


| Path                                   | Contents                                  |
| -------------------------------------- | ----------------------------------------- |
| `outputs/predict/{system}/`            | Boltz predict + `processed/`              |
| `outputs/modes/{system}_ca_modes.npz`  | Soft CA modes after score ascent          |
| `outputs/per_system/{system}.json`     | Per-system AUROC / ΔAUROC                 |
| `outputs/analysis/retrieval_panel.csv` | Aggregate panel                           |
| `outputs/analysis/SUMMARY.md`          | Subclass medians and hypothesis checklist |


