# Experiment 2 — Measured local score landscape + soft-mode localization

## Claim

Around a Boltz score-stationary prediction x₀, the local score-derived potential U_model is evaluated in the plane spanned by the softest and stiffest Hessian modes. The experimental apo→holo displacement Δx is projected into that plane, and soft-mode morph cartoons are shown together with per-residue experimental motion D_i and Boltz soft-mode participation S_i.

## Output

Main figure: `outputs/score_landscape/trio_3d_2d.png`. Columns are (1) 3D U_model surface, (2) 2D contour with the apo→holo arrow, (3) soft-mode morph, (4) per-residue localization profiles. Demo systems: `1EX6_B`, `1GUD_A`, `1AW2_A`.

## Reproduce

This experiment reuses Exp 1 soft91 Boltz predictions and the CoDNaS pair table/PDBs under `../1_codnas_apo_holo`. You also need the shared score/Hessian code in `../../src/` and a GPU for score-landscape / Hessian steps. There is no separate dataset download for Exp 2.

### 1. Get the data

Complete Exp 1 data setup and at least the three demo predicts (or the full soft91 campaign). Required artifacts per system:

- `../1_codnas_apo_holo/data/tables/revision1_86_plus_5.csv` and matching PDBs in `../1_codnas_apo_holo/data/pairs/`
- `../1_codnas_apo_holo/outputs/predict_soft91/{system}/boltz_results_*/processed/` plus the predicted CIF

See `../1_codnas_apo_holo/README.md` for how to obtain the CoDNaS table/PDBs and run `bash scripts/run_soft91.sh`.

### 2. Parse / prepare inputs

No extra parsing. `score_landscape.py` and `localize_soft_motion.py` load Exp 1 pairs via `pairs.py`, experimental apo/holo PDBs via `geom.py`, and Boltz processed batches / CIFs from the soft91 predict directories. Soft CA modes are computed on the fly (score ascent + autodiff Hessian) or reused from caches written under `outputs/score_landscape/`.

### 3. Run the experiment

```bash
cd experiments/2_x0_landscape

# Landscape panels (score ascent → soft/stiff plane → 3D/2D U_model; also writes a landscape-only crop helper)
PYTHONPATH="scripts:../1_codnas_apo_holo/scripts:../../src" \
  ../../boltz/.venv/bin/python scripts/score_landscape.py

# Soft-mode morph cartoons + per-residue D_i / Boltz S_i profiles
PYTHONPATH="scripts:../1_codnas_apo_holo/scripts:../../src" \
  ../../boltz/.venv/bin/python scripts/localize_soft_motion.py \
  --systems 1EX6_B,1GUD_A,1AW2_A --with_anm

# Merge landscape + localization into the paper figure
../../boltz/.venv/bin/python scripts/compose_trio_figure.py
```

`score_landscape.py` writes intermediates under `outputs/score_landscape/` (including `trio_3d_2d_landscape_only.png`). `localize_soft_motion.py` writes under `outputs/localization/`. `compose_trio_figure.py` merges those into `outputs/score_landscape/trio_3d_2d.png`.
