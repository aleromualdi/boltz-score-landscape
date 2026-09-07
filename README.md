# boltz-score-landscape  — conformational soft modes from the Boltz-2 score

This repo asks whether **local geometry of the Boltz-2 diffusion score** encodes experimentally relevant protein motions, even when standard inference returns a single structure. Around a predicted pose, the score Hessian yields **soft modes**; those modes are tested against *apo↔holo* displacements (CoDNaS), visualized as a local score-derived landscape, and compared to ANM for blind retrieval of PSCDB moving segments. Narrative draft: `[scientific_narrative.md](scientific_narrative.md)`. 

Implementation lives in `[src/](src/)`.

Boltz: clone / install [jwohlwend/boltz](https://github.com/jwohlwend/boltz) under `boltz/` so experiments can import the pretrained model and CLI. Predictions are then refined by **score ascent** to a local stationary pose (projected EDM score; not a second denoiser pass), rigid-body modes are projected out, and soft CA modes are obtained via autodiff Hessian–vector products (Lanczos / soft Ritz). Details per experiment below.

## Setup

1. Clone this repo and place a working Boltz install at `boltz/` (with `.venv` and the `boltz` CLI).
2. Use that env (or the repo-root `.venv` if linked) so `import boltz` and `../../src` resolve.
3. GPU needed; `boltz predict` needs network for ColabFold MSA (`--use_msa_server`).


## Experiments (how to regenerate results)


| Exp | What                                              | Reproduce                                                                                                                         |
| --- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 1   | CoDNaS soft-subspace enrichment (C_10, κ vs null) | `[experiments/1_codnas_apo_holo/README.md](experiments/1_codnas_apo_holo/README.md)` → `bash scripts/run_soft91.sh`               |
| 2   | Score landscape + soft-mode localization figure   | `[experiments/2_x0_landscape/README.md](experiments/2_x0_landscape/README.md)` (needs Exp 1 predicts)                             |
| 3   | PSCDB local vs domain AUROC (Boltz vs ANM)        | `[experiments/3_pscdb_local_vs_domain/README.md](experiments/3_pscdb_local_vs_domain/README.md)` → `bash scripts/run_campaign.sh` |


Each experiment README covers data download, prepare/parse steps, and the exact run commands.

## Example figure (Exp 2)

![Score landscape and soft-mode localization](experiments/2_x0_landscape/outputs/score_landscape/trio_3d_2d.png)

Three CoDNaS systems after score ascent: **1EX6_B** (domain closure), **1GUD_A** (hinge), **1AW2_A** (loop). Columns show the measured score-derived potential U_model in the soft×stiff plane (3D / 2D with apo→holo arrow), a soft-mode morph, and per-residue experimental motion D_i vs Boltz soft participation S_i (ANM for reference). Soft valleys and participation track the experimental rearrangements; the loop case is where Boltz localization is most distinct from a harmonic ANM baseline.