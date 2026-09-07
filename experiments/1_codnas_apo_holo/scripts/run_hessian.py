"""Score-ascent Hessian plus CA-projected soft Ritz modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC))

from boltz.main import get_cache_path
from estimate_logdet import (
    batch_to_device,
    closest_schedule_sigma,
    denoise_coords,
    evaluate_sigma,
    load_batch,
    load_model,
    make_autodiff_precision_hvp,
    trunk_and_condition,
)
from rigid import internal_dimension, project_out_rigid, projected_hvp, rigid_basis, rigid_projector
from score_mode import projected_score, score_ascent
from slq import lanczos_tridiagonal, ritz_pairs, unit_rademacher_probe

from inject_apo import boltz_residue_table, inject_apo_coords
from load_pose import coords_from_predicted_cif, find_predicted_cif
from pairs import pair_by_apo, pdb_path

EXP_ROOT = Path(__file__).resolve().parents[1]
MSA_SUBSAMPLE = 1024


def _center(coords: Tensor, atom_mask: Tensor) -> Tensor:
    """Subtract the masked centroid.

    Args:
        coords: Atom coordinates.
        atom_mask: Real-atom mask.

    Returns:
        Centered coordinates.
    """
    weight = atom_mask.unsqueeze(-1)
    com = (coords * weight).sum(dim=1, keepdim=True) / weight.sum(dim=1, keepdim=True).clamp_min(
        1.0
    )
    return (coords - com) * weight


def ca_indices(processed_dir: Path, system: str) -> np.ndarray:
    """Atom indices of CA in the unpadded processed structure.

    Args:
        processed_dir: Boltz ``processed/`` folder.
        system: YAML stem / NPZ stem.

    Returns:
        Integer index array.
    """
    npz = np.load(processed_dir / "structures" / f"{system}.npz")
    names = np.asarray(npz["atoms"]["name"])
    return np.asarray(np.where(names == "CA")[0], dtype=np.int64)


def soft_ca_modes(
    matvec: Any,
    probe: Tensor,
    atom_mask: Tensor,
    rigid_modes: Tensor,
    ca_idx: np.ndarray,
    n_modes: int,
    lanczos_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest-Ritz CA displacements from one Lanczos factorization.

    Args:
        matvec: Projected Hessian-vector product.
        probe: Unit internal start vector.
        atom_mask: Real-atom mask.
        rigid_modes: Rigid basis at the pose.
        ca_idx: CA atom indices.
        n_modes: How many lowest modes to keep.
        lanczos_steps: Krylov dimension.

    Returns:
        Modes (K, N_ca, 3) and Ritz values (K,).
    """
    lanczos = lanczos_tridiagonal(
        matvec, probe, atom_mask, n_steps=lanczos_steps, keep_basis=True
    )
    pairs = sorted(ritz_pairs(lanczos, atom_mask), key=lambda item: item[0])
    kept: list[tuple[float, Tensor]] = []
    for theta, raw in pairs:
        if abs(theta) < 1e-3:
            continue
        vec = project_out_rigid(raw, rigid_modes, atom_mask)
        kept.append((float(theta), vec))
        if len(kept) >= n_modes:
            break
    if not kept:
        theta, raw = pairs[0]
        kept.append((float(theta), project_out_rigid(raw, rigid_modes, atom_mask)))
    modes = []
    evals = []
    for theta, vec in kept:
        ca_vec = vec[0, ca_idx].detach().cpu().numpy().astype(np.float64)
        nrm = float(np.linalg.norm(ca_vec))
        if nrm > 0:
            ca_vec = ca_vec / nrm
        modes.append(ca_vec)
        evals.append(theta)
    return np.stack(modes, axis=0), np.asarray(evals, dtype=np.float64)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--processed_dir", type=Path, required=True)
    parser.add_argument("--predict_root", type=Path, default=None)
    parser.add_argument("--n_res", type=int, default=0)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sigma_target", type=float, default=0.28)
    parser.add_argument("--n_probes", type=int, default=8)
    parser.add_argument("--n_soft_modes", type=int, default=15)
    parser.add_argument("--score_ascent_steps", type=int, default=80)
    parser.add_argument("--num_subsampled_msa", type=int, default=MSA_SUBSAMPLE)
    parser.add_argument("--lanczos_steps", type=int, default=40)
    parser.add_argument(
        "--pose",
        choices=("predicted", "apo", "holo"),
        default="predicted",
        help=(
            "Hessian site: deposited CIF (+ score-ascent), experimental apo, "
            "or experimental holo (upper-bound control)."
        ),
    )
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir is not None else EXP_ROOT / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = {"predicted": "score_ascent", "apo": "apo", "holo": "holo"}[args.pose]
    json_path = out_dir / f"{args.system}_{suffix}_logdet.json"
    npy_path = out_dir / f"{args.system}_{suffix}_ca_modes.npz"
    if json_path.exists() and npy_path.exists() and not args.force:
        print(f"skip {args.system}: outputs exist", flush=True)
        return

    predict_root = (
        args.predict_root
        if args.predict_root is not None
        else EXP_ROOT / "outputs" / "predict" / args.system
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mol_dir = Path(get_cache_path()) / "mols"
    print(
        f"device={device} system={args.system} processed={args.processed_dir} "
        f"msa={args.num_subsampled_msa}",
        flush=True,
    )
    batch = load_batch(args.system, mol_dir, processed=args.processed_dir)
    feats = batch_to_device(batch, device)
    atom_mask = feats["atom_pad_mask"].float()
    cif_path = find_predicted_cif(predict_root, args.system)
    coords = coords_from_predicted_cif(
        cif_path=cif_path,
        atom_mask=atom_mask,
        processed_dir=args.processed_dir,
        system=args.system,
    )
    # Deposited pose for any path that reads feats['coords'] (B, M, 3).
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
        """Denoiser at the snapped σ.

        Args:
            current: Pose.

        Returns:
            D(current, σ).
        """
        return denoise_coords(model, current, sigma, s_inputs, s_trunk, feats, cond)

    inject_stats: dict[str, float] = {}
    boltz_seq, _atoms, _residues = boltz_residue_table(args.processed_dir, args.system)
    if args.pose in {"apo", "holo"}:
        exp_label = args.system if args.pose == "apo" else pair_by_apo()[args.system]["holo_id"]
        pose, inject_stats = inject_apo_coords(
            predicted=coords,
            atom_mask=atom_mask,
            processed_dir=args.processed_dir,
            system=args.system,
            apo_pdb=pdb_path(exp_label),
            apo_label=exp_label,
        )
        pose = _center(pose, atom_mask)
        rigid_modes = rigid_basis(pose, atom_mask)
        score_state = projected_score(_denoise, pose, sigma, atom_mask, rigid_modes)
        ascent_score_rms = score_state.score_rms
        ascent_converged = 0.0
        print(
            f"{args.pose} inject ca_rmsd={inject_stats['ca_rmsd_to_apo']:.4f}  "
            f"copy={inject_stats['copy_frac']:.3f}  "
            f"score_rms={ascent_score_rms:.4g}  (no ascent)",
            flush=True,
        )
    elif args.score_ascent_steps > 0:
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
        ascent_score_rms = ascent.score_rms
        ascent_converged = float(ascent.converged)
        print(
            f"ascent score_rms {ascent.score_rms0:.4g}→{ascent.score_rms:.4g}  "
            f"conv={ascent.converged}",
            flush=True,
        )
        rigid_modes = rigid_basis(pose, atom_mask)
    else:
        pose = coords
        rigid_modes = rigid_basis(pose, atom_mask)
        ascent_score_rms = projected_score(
            _denoise, pose, sigma, atom_mask, rigid_modes
        ).score_rms
        ascent_converged = 0.0
        print(f"no ascent  score_rms={ascent_score_rms:.4g}", flush=True)
    n_free = internal_dimension(atom_mask, int(rigid_modes.shape[0]))
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    probes = [
        unit_rademacher_probe(
            atom_mask, generator, _center, rigid_projector(rigid_modes, atom_mask)
        )
        for _ in range(args.n_probes)
    ]
    stats = evaluate_sigma(
        model=model,
        coords=pose,
        sigma=sigma,
        atom_mask=atom_mask,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        feats=feats,
        diffusion_conditioning=cond,
        probes=probes,
        n_free=n_free,
        fd_eps=0.02,
        ridge=1e-4,
        lanczos_steps=args.lanczos_steps,
        central=True,
        hvp="autodiff",
        rigid_modes=rigid_modes,
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
    ca_idx = ca_indices(args.processed_dir, args.system)
    modes, evals = soft_ca_modes(
        matvec=matvec,
        probe=probes[0],
        atom_mask=atom_mask,
        rigid_modes=rigid_modes,
        ca_idx=ca_idx,
        n_modes=args.n_soft_modes,
        lanczos_steps=args.lanczos_steps,
    )
    pred_ca = pose[0, ca_idx].detach().cpu().numpy().astype(np.float64)
    np.savez(
        npy_path,
        pred_ca=pred_ca,
        modes=modes,
        evals=evals,
        ca_idx=ca_idx,
        boltz_seq=np.asarray(boltz_seq),
    )
    row: dict[str, Any] = {
        "system": args.system,
        "n_res": args.n_res,
        "n_ca": len(ca_idx),
        "sigma": float(sigma),
        "n_rigid": float(rigid_modes.shape[0]),
        "num_subsampled_msa": float(args.num_subsampled_msa),
        "cif": str(cif_path),
        **stats,
        **inject_stats,
        "pose": args.pose,
        "ascent_score_rms": ascent_score_rms,
        "ascent_converged": ascent_converged,
        "lambda_soft": float(evals[0]),
        "n_soft_modes": float(len(evals)),
    }
    json_path.write_text(json.dumps(row, indent=2))
    print(
        f"S/coord={stats['S_hessian_per_coord_bits']:.4f}  "
        f"gm/am={stats['gm_over_am']:.4f}  λ_soft={evals[0]:.4g}  n_ca={len(ca_idx)}",
        flush=True,
    )
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {npy_path}", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
