"""Compute Boltz soft CA modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

EXP_ROOT = Path(__file__).resolve().parents[1]
EXP100 = EXP_ROOT.parents[0] / "1_codnas_apo_holo" / "scripts"
SRC = EXP_ROOT.parents[1] / "src"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(EXP100))

from boltz.main import get_cache_path 
from estimate_logdet import (  
    batch_to_device,
    closest_schedule_sigma,
    denoise_coords,
    load_batch,
    load_model,
    make_autodiff_precision_hvp,
    trunk_and_condition,
)
from inject_apo import boltz_residue_table 
from load_pose import coords_from_predicted_cif, find_predicted_cif 
from rigid import projected_hvp, rigid_basis, rigid_projector 
from run_hessian import MSA_SUBSAMPLE, _center, ca_indices, soft_ca_modes 
from score_mode import score_ascent 
from slq import unit_rademacher_probe 

PANEL = EXP_ROOT / "inputs" / "panel_full.json"
MODE_DIR = EXP_ROOT / "outputs" / "modes"


def load_record(system: str) -> dict[str, object]:
    """Load panel record.

    Args:
        system: System stem.

    Returns:
        Panel dict.
    """
    for row in json.loads(PANEL.read_text()):
        if row["system"] == system:
            return row
    raise KeyError(system)


def find_processed(predict_root: Path, system: str) -> Path:
    """Locate processed directory.

    Args:
        predict_root: Predict root.
        system: System stem.

    Returns:
        Processed path.
    """
    matches = sorted(predict_root.glob("boltz_results_*/processed"))
    for proc in matches:
        if (proc / "structures" / f"{system}.npz").exists():
            return proc
    raise FileNotFoundError(f"no processed structure for {system}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--predict_root", type=Path, required=True)
    parser.add_argument("--n_soft_modes", type=int, default=50)
    parser.add_argument("--lanczos_steps", type=int, default=40)
    parser.add_argument("--score_ascent_steps", type=int, default=80)
    parser.add_argument("--sigma_target", type=float, default=0.28)
    parser.add_argument("--num_subsampled_msa", type=int, default=MSA_SUBSAMPLE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    MODE_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = MODE_DIR / f"{args.system}_ca_modes.npz"
    if out_npz.exists() and not args.force:
        print(f"skip modes {args.system}", flush=True)
        return

    record = load_record(args.system)
    processed = find_processed(args.predict_root, args.system)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mol_dir = Path(get_cache_path()) / "mols"
    batch = load_batch(args.system, mol_dir, processed=processed)
    feats = batch_to_device(batch, device)
    atom_mask = feats["atom_pad_mask"].float()
    cif = find_predicted_cif(args.predict_root, args.system)
    coords = coords_from_predicted_cif(cif, atom_mask, processed, args.system)
    feats["coords"] = coords
    torch.manual_seed(0)
    model = load_model(
        device,
        subsample_msa=True,
        use_kernels=False,
        num_subsampled_msa=args.num_subsampled_msa,
    )
    s_inputs, s_trunk, cond = trunk_and_condition(model, feats, recycling_steps=3)
    coords = _center(coords, atom_mask)
    sigma = closest_schedule_sigma(model, args.sigma_target)

    def _denoise(current: Tensor) -> Tensor:
        """Denoise at fixed σ.

        Args:
            current: Pose.

        Returns:
            Denoised coords.
        """
        return denoise_coords(model, current, sigma, s_inputs, s_trunk, feats, cond)

    ascent = score_ascent(
        denoise=_denoise,
        coords=coords,
        sigma=sigma,
        atom_mask=atom_mask,
        n_steps=args.score_ascent_steps,
        lr=0.02,
        score_tol=0.05,
        residual_tol=0.005,
        method="adam",
    )
    pose = ascent.coords
    rigid_modes = rigid_basis(pose, atom_mask)
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    probe = unit_rademacher_probe(
        atom_mask, generator, _center, rigid_projector(rigid_modes, atom_mask)
    )
    matvec = projected_hvp(
        make_autodiff_precision_hvp(
            model=model,
            coords=pose,
            sigma=sigma,
            atom_mask=atom_mask,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            feats=feats,
            diffusion_conditioning=cond,
            ridge=1e-4,
        ),
        rigid_modes,
        atom_mask,
    )
    ca_idx = ca_indices(processed, args.system)
    modes, evals = soft_ca_modes(
        matvec=matvec,
        probe=probe,
        atom_mask=atom_mask,
        rigid_modes=rigid_modes,
        ca_idx=ca_idx,
        n_modes=args.n_soft_modes,
        lanczos_steps=args.lanczos_steps,
    )
    pred_ca = pose[0, ca_idx].detach().cpu().numpy().astype(np.float64)
    boltz_seq, _atoms, _res = boltz_residue_table(processed, args.system)
    np.savez(
        out_npz,
        pred_ca=pred_ca,
        modes=modes,
        evals=evals,
        ca_idx=ca_idx,
        boltz_seq=np.asarray(boltz_seq),
        sigma=np.asarray([sigma], dtype=np.float64),
        ascent_score_rms=np.asarray([ascent.score_rms], dtype=np.float64),
        class_label=np.asarray(str(record["class"])),
        pscid=np.asarray(str(record["pscid"])),
    )
    print(
        f"{args.system}: modes={len(evals)} n_ca={len(ca_idx)} "
        f"λsoft={evals[0]:.4g} ascent_rms={ascent.score_rms:.4g}",
        flush=True,
    )
    print(f"wrote {out_npz}", flush=True)


if __name__ == "__main__":
    main()
