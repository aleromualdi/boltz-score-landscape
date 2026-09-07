# Experiment 1 — CoDNaS soft-subspace enrichment

## Claim

Local soft subspaces derived from the Boltz score Hessian are enriched for experimentally observed conformational degrees of freedom: the apo↔holo Cα displacement Δx taken from CoDNaS. The primary metric is C_k, the fraction of ‖Δx‖² that lies in the top-k soft Cα modes, compared against matched random-direction nulls. As a secondary check we also report directional curvature κ along Δx versus the same nulls.

## Results (`soft91`)

The panel is 91 pairs. DynDom motion type splits into `domain movements` (n=55) and `loop movements` (n=36).

| | all (91) | domain (55) | loop (36) |
|---|---:|---:|---:|
| C_10 enriched vs null | 91 / 91 | 55 / 55 | 36 / 36 |
| κ softer vs null | 91 / 91 | 55 / 55 | 36 / 36 |
| Median C_10 (exp / null) | 0.256 / 0.018 | 0.296 / 0.015 | 0.235 / 0.024 |
| Median enrichment fold | 13.2× | 17.1× | 11.4× |

Panel: `outputs/analysis_soft91/soft_panel.csv`

## Reproduce

You need a GPU, Boltz installed at `../../boltz/`, and the shared score/Hessian helpers in `../../src/`. Running `boltz predict` also needs network access so ColabFold can download MSAs (`--use_msa_server`).

### 1. Get the data

The benchmark is the apo–holo set from [Saldaño et al., Bioinformatics 2022](https://doi.org/10.1093/bioinformatics/btac202). The pair table and structures are distributed with the paper supplement at [sbgunq/af2confdiv-oct2021](https://gitlab.com/sbgunq/publications/af2confdiv-oct2021) (paths recorded in `data/SOURCE.txt`). Clone or download that repo, then place the CSV and PDBs under this experiment’s `data/` tree:

```bash
cd experiments/1_codnas_apo_holo
mkdir -p data/tables data/pairs

cp /path/to/af2confdiv-oct2021/data/26-01-2022/revision1_86_plus_5.csv data/tables/

unzip /path/to/af2confdiv-oct2021/structures/91_apo_holo_pairs.zip -d /tmp/codnas_pairs
# One PDB per table id: data/pairs/{apo_id}.pdb and data/pairs/{holo_id}.pdb (182 files for 91 pairs).
find /tmp/codnas_pairs -name '*.pdb' -exec cp -n {} data/pairs/ \;
```

Expected layout:

```
data/
  SOURCE.txt
  data_description.md
  tables/revision1_86_plus_5.csv   # 91 rows + header (; -delimited)
  pairs/{apo_id,holo_id}.pdb       # 182 chain PDBs
```

The CSV is `;`-delimited and includes apo/holo ids, published RMSD, length, DynDom `motion_type`, and domain counts; full column notes are in `data/data_description.md`. `scripts/pairs.py` is the loader used everywhere else (`load_pairs()`, `pdb_path()`).

### 2. Parse / prepare inputs

There is no separate format conversion beyond the table and PDBs. `scripts/prepare_inputs.py` reads `revision1_86_plus_5.csv`, pulls the apo sequence from each apo PDB under `data/pairs/`, and writes one Boltz YAML per system plus a panel JSON used by the campaign script:

```bash
cd experiments/1_codnas_apo_holo
PYTHONPATH=scripts ../../boltz/.venv/bin/python scripts/prepare_inputs.py
```

This creates `inputs/yamls/{apo_id}.yaml` (single-chain Boltz inputs) and `inputs/panel91.json` (91 records with ids, RMSD, motion type, paths, and `n_res`). `run_soft91.sh` runs this step automatically before predicting.

### 3. Run the experiment

Boltz soft enrichment (predict → score ascent to a stationary pose → soft Cα modes → C_k / κ vs nulls):

```bash
cd experiments/1_codnas_apo_holo
bash scripts/run_soft91.sh
```

For each apo id, the script runs predict with seed 0, 50 sampling steps, one diffusion sample, MSA server, and MSA subsample depth 1024. It then calls `directional_curvature.py` with `n_null=500` (score ascent + soft-mode enrichment) and finally `summarize_soft.py`, which writes `outputs/analysis_soft91/soft_panel.csv`. Per-system failures are appended to `outputs/logs/soft91_failures.log`.