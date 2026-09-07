"""Soft-subspace enrichment for experimental conformational displacements.

At a Boltz score stationary point, ask whether experimentally observed
conformational differences (apo↔holo equilibrium pair) lie preferentially
in Boltz's local soft subspace. Primary metric: C_k vs matched random null.
Secondary: directional curvature κ along the experimental displacement.
"""

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
    load_batch,
    load_model,
    make_autodiff_precision_hvp,
    trunk_and_condition,
)
from rigid import project_out_rigid, projected_hvp, rigid_basis, rigid_projector
from score_mode import score_ascent
from slq import masked_dot, masked_norm, unit_rademacher_probe

from geom import (
    apo_holo_delta_aligned,
    cumulative_subspace_overlap,
    kabsch_rotation,
    load_ca_sequence,
    matched_residue_indices,
    parse_label,
)
from inject_apo import boltz_residue_table
from load_pose import coords_from_predicted_cif, find_predicted_cif, load_cif_atoms
from pairs import EXP_ROOT, pair_by_apo, pdb_path
from run_hessian import MSA_SUBSAMPLE, _center, ca_indices, soft_ca_modes


def nearest_experimental_state(pred_apo: float, pred_holo: float) -> str:
    """Label which experimental endpoint is closest to the Boltz pose.

    Args:
        pred_apo: CA RMSD from Boltz pose to experimental state A (apo).
        pred_holo: CA RMSD from Boltz pose to experimental state B (holo).

    Returns:
        ``"apo"`` or ``"holo"``.
    """
    return "apo" if pred_apo <= pred_holo else "holo"


def pred_ca_rmsds(system: str, predict_root: Path) -> tuple[float, float]:
    """CA RMSD of the deposited Boltz pose to experimental apo and holo.

    Args:
        system: Apo table id.
        predict_root: Boltz predict folder.

    Returns:
        ``(pred→apo, pred→holo)`` in Å.
    """
    spec = pair_by_apo()[system]
    cif = find_predicted_cif(predict_root, system)
    names, xyz = load_cif_atoms(cif)
    pred = xyz[[i for i, n in enumerate(names) if n == "CA"]]
    _p, apo_model, _ = parse_label(spec["apo_id"])
    _p, holo_model, _ = parse_label(spec["holo_id"])
    apo, _ = load_ca_sequence(pdb_path(spec["apo_id"]), apo_model)
    holo, _ = load_ca_sequence(pdb_path(spec["holo_id"]), holo_model)
    n = min(len(pred), len(apo), len(holo))

    def _rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
        rot, cm, ct = kabsch_rotation(mobile[:n], target[:n])
        aligned = (mobile[:n] - cm) @ rot + ct
        return float(np.sqrt(((aligned - target[:n]) ** 2).sum(1).mean()))

    return _rmsd(pred, apo), _rmsd(pred, holo)


def aligned_ca_delta(
    system: str,
    processed_dir: Path,
    predict_root: Path,
) -> tuple[np.ndarray, np.ndarray, float, str, np.ndarray]:
    """Experimental Δx on Boltz CA indices in the pose frame.

    Args:
        system: Apo table id.
        processed_dir: Boltz processed folder.
        predict_root: Predict output root.

    Returns:
        Matched CA Δx (N, 3), pred CA (N, 3), apo-holo RMSD, boltz sequence,
        full CA atom index array used.
    """
    spec = pair_by_apo()[system]
    _p, apo_model, _ = parse_label(spec["apo_id"])
    _p, holo_model, _ = parse_label(spec["holo_id"])
    apo_ca, apo_seq = load_ca_sequence(pdb_path(spec["apo_id"]), apo_model)
    holo_ca, holo_seq = load_ca_sequence(pdb_path(spec["holo_id"]), holo_model)
    apo_m, delta, rmsd, apo_match_idx, _holo_idx = apo_holo_delta_aligned(
        apo_ca, apo_seq, holo_ca, holo_seq
    )
    boltz_seq, _atoms, _res = boltz_residue_table(processed_dir, system)
    cif = find_predicted_cif(predict_root, system)
    names, xyz = load_cif_atoms(cif)
    pred_ca_all = xyz[[i for i, n in enumerate(names) if n == "CA"]]
    ca_idx_full = ca_indices(processed_dir, system)
    pred_idx, apo_idx = matched_residue_indices(boltz_seq, apo_seq)
    apo_to_delta = {int(i): k for k, i in enumerate(apo_match_idx)}
    keep_pred: list[int] = []
    keep_delta: list[int] = []
    for p_i, a_i in zip(pred_idx, apo_idx, strict=True):
        if int(a_i) in apo_to_delta:
            keep_pred.append(int(p_i))
            keep_delta.append(apo_to_delta[int(a_i)])
    pred_ca = pred_ca_all[keep_pred]
    delta = delta[keep_delta]
    apo_m = apo_m[keep_delta]
    rot, _cm, _ct = kabsch_rotation(pred_ca, apo_m)
    delta_pose = delta @ rot.T
    ca_idx = ca_idx_full[keep_pred]
    return delta_pose, pred_ca, rmsd, boltz_seq, ca_idx


def embed_ca_displacement(
    delta_ca: np.ndarray,
    ca_idx: np.ndarray,
    atom_mask: Tensor,
    device: torch.device,
) -> Tensor:
    """Place a CA-only displacement into the padded atom tensor.

    Args:
        delta_ca: (N_ca, 3) in the Boltz pose frame.
        ca_idx: Atom indices for each CA row.
        atom_mask: Real-atom mask (B, M).
        device: Torch device.

    Returns:
        Tensor (1, M, 3).
    """
    vec = torch.zeros(
        *atom_mask.shape,
        3,
        dtype=torch.float32,
        device=device,
    )
    ca_tensor = torch.as_tensor(delta_ca, device=device, dtype=torch.float32)
    for row, atom_i in enumerate(ca_idx):
        vec[0, int(atom_i)] = ca_tensor[row]
    return vec


def unit_internal_direction(
    delta_ca: np.ndarray,
    ca_idx: np.ndarray,
    atom_mask: Tensor,
    rigid_modes: Tensor,
    device: torch.device,
) -> Tensor:
    """Rigid-project and normalize a CA displacement in all-atom internal space.

    Args:
        delta_ca: CA displacement (N, 3).
        ca_idx: CA atom indices.
        atom_mask: Real-atom mask.
        rigid_modes: Rigid basis at the pose.
        device: Torch device.

    Returns:
        Unit internal vector (1, M, 3).
    """
    raw = embed_ca_displacement(delta_ca, ca_idx, atom_mask, device)
    internal = project_out_rigid(raw, rigid_modes, atom_mask)
    nrm = masked_norm(internal, atom_mask)
    if float(nrm.item()) < 1e-12:
        msg = "vanishing internal displacement after rigid projection"
        raise ValueError(msg)
    return internal / nrm


def directional_curvature(
    matvec: Any,
    direction: Tensor,
    atom_mask: Tensor,
) -> float:
    """Compute κ = vᵀ H v / vᵀ v for a unit internal direction.

    Args:
        matvec: Projected Hessian-vector product.
        direction: Unit displacement (1, M, 3).
        atom_mask: Real-atom mask.

    Returns:
        Directional curvature (precision units).
    """
    hv = matvec(direction)
    num = float(masked_dot(direction, hv, atom_mask).item())
    den = float(masked_dot(direction, direction, atom_mask).item())
    if den < 1e-18:
        return float("nan")
    return num / den


def null_directional_curvatures(
    matvec: Any,
    atom_mask: Tensor,
    rigid_modes: Tensor,
    ca_idx: np.ndarray,
    n_samples: int,
    generator: torch.Generator,
    device: torch.device,
) -> np.ndarray:
    """Draw matched random unit CA directions and evaluate κ.

    Args:
        matvec: Projected Hessian-vector product.
        atom_mask: Real-atom mask.
        rigid_modes: Rigid basis.
        ca_idx: CA atom indices for embedding.
        n_samples: Null draw count.
        generator: Torch RNG.
        device: Torch device.

    Returns:
        Array of null κ values.
    """
    n_ca = len(ca_idx)
    values: list[float] = []
    log_every = max(1, n_samples // 10)
    for i in range(n_samples):
        noise = torch.randn((n_ca, 3), generator=generator, device=device, dtype=torch.float32)
        noise_np = noise.detach().cpu().numpy()
        unit = unit_internal_direction(noise_np, ca_idx, atom_mask, rigid_modes, device)
        values.append(directional_curvature(matvec, unit, atom_mask))
        if (i + 1) % log_every == 0 or i + 1 == n_samples:
            print(f"  null κ {i + 1}/{n_samples}", flush=True)
    return np.asarray(values, dtype=np.float64)


def null_c10_values(
    modes_apo: np.ndarray,
    ca_idx: np.ndarray,
    atom_mask: Tensor,
    rigid_modes: Tensor,
    n_samples: int,
    generator: torch.Generator,
    device: torch.device,
    k: int = 10,
) -> np.ndarray:
    """Null distribution of C_k for random CA directions on the same support.

    Args:
        modes_apo: Soft modes (K, N_ca, 3) in pose/apo frame.
        ca_idx: CA atom indices.
        atom_mask: Real-atom mask.
        rigid_modes: Rigid basis.
        n_samples: Null draw count.
        generator: Torch RNG.
        device: Torch device.
        k: Subspace dimension for C_k.

    Returns:
        Null C_k samples.
    """
    n_ca = len(ca_idx)
    values: list[float] = []
    n_keep = min(k, len(modes_apo))
    for _ in range(n_samples):
        noise = torch.randn((n_ca, 3), generator=generator, device=device, dtype=torch.float32)
        noise_np = noise.detach().cpu().numpy()
        unit = unit_internal_direction(noise_np, ca_idx, atom_mask, rigid_modes, device)
        delta_ca = unit[0, ca_idx].detach().cpu().numpy().astype(np.float64)
        values.append(cumulative_subspace_overlap(modes_apo, delta_ca, k=n_keep))
    return np.asarray(values, dtype=np.float64)


def evaluate_system(
    system: str,
    processed_dir: Path,
    predict_root: Path,
    *,
    n_null: int = 200,
    n_soft_modes: int = 15,
    lanczos_steps: int = 40,
    score_ascent_steps: int = 80,
    sigma_target: float = 0.28,
    num_subsampled_msa: int = MSA_SUBSAMPLE,
    seed: int = 0,
) -> dict[str, Any]:
    """Run C_k enrichment and κ null tests at the score-ascent stationary point.

    Args:
        system: Apo table id (experimental displacement is apo→holo).
        processed_dir: Boltz processed folder.
        predict_root: Predict output root.
        n_null: Random-direction null samples.
        n_soft_modes: Ritz modes to keep.
        lanczos_steps: Krylov steps for soft modes.
        score_ascent_steps: Score-ascent iterations.
        sigma_target: Target σ in Å.
        num_subsampled_msa: MSA subsample depth.
        seed: RNG seed for null draws.

    Returns:
        Result dictionary (JSON-serializable).
    """
    spec = pair_by_apo()[system]
    pred_apo, pred_holo = pred_ca_rmsds(system, predict_root)
    delta_ca, _pred_ca, rmsd, _seq, ca_idx = aligned_ca_delta(
        system, processed_dir, predict_root
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mol_dir = Path(get_cache_path()) / "mols"
    batch = load_batch(system, mol_dir, processed=processed_dir)
    feats = batch_to_device(batch, device)
    atom_mask = feats["atom_pad_mask"].float()
    cif_path = find_predicted_cif(predict_root, system)
    coords = coords_from_predicted_cif(cif_path, atom_mask, processed_dir, system)
    feats["coords"] = coords
    torch.manual_seed(seed)
    model = load_model(
        device,
        subsample_msa=True,
        use_kernels=False,
        num_subsampled_msa=num_subsampled_msa,
    )
    s_inputs, s_trunk, cond = trunk_and_condition(model, feats, recycling_steps=3)
    coords = _center(coords, atom_mask)
    sigma = closest_schedule_sigma(model, sigma_target)

    def _denoise(current: Tensor) -> Tensor:
        """Denoiser at snapped σ."""
        return denoise_coords(model, current, sigma, s_inputs, s_trunk, feats, cond)

    ascent = score_ascent(
        denoise=_denoise,
        coords=coords,
        sigma=sigma,
        atom_mask=atom_mask,
        n_steps=score_ascent_steps,
        lr=0.02,
        score_tol=0.05,
        residual_tol=0.005,
        method="adam",
    )
    pose = ascent.coords
    rigid_modes = rigid_basis(pose, atom_mask)
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

    exp_dir = unit_internal_direction(delta_ca, ca_idx, atom_mask, rigid_modes, device)
    kappa_exp = directional_curvature(matvec, exp_dir, atom_mask)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1)
    kappa_null = null_directional_curvatures(
        matvec, atom_mask, rigid_modes, ca_idx, n_null, generator, device
    )

    probe = unit_rademacher_probe(
        atom_mask, generator, _center, rigid_projector(rigid_modes, atom_mask)
    )
    ca_idx_full = ca_indices(processed_dir, system)
    modes, evals = soft_ca_modes(
        matvec=matvec,
        probe=probe,
        atom_mask=atom_mask,
        rigid_modes=rigid_modes,
        ca_idx=ca_idx_full,
        n_modes=n_soft_modes,
        lanczos_steps=lanczos_steps,
    )
    pred_ca_pose = pose[0, ca_idx].detach().cpu().numpy().astype(np.float64)
    _p, apo_model, _ = parse_label(spec["apo_id"])
    apo_ca, apo_seq = load_ca_sequence(pdb_path(spec["apo_id"]), apo_model)
    _p, holo_model, _ = parse_label(spec["holo_id"])
    holo_ca, holo_seq = load_ca_sequence(pdb_path(spec["holo_id"]), holo_model)
    apo_m, delta_aligned, _rmsd2, apo_match_idx, _ = apo_holo_delta_aligned(
        apo_ca, apo_seq, holo_ca, holo_seq
    )
    boltz_seq, _, _ = boltz_residue_table(processed_dir, system)
    pred_idx, apo_idx = matched_residue_indices(boltz_seq, apo_seq)
    apo_to_delta = {int(i): k for k, i in enumerate(apo_match_idx)}
    keep_pred = [
        int(p_i)
        for p_i, a_i in zip(pred_idx, apo_idx, strict=True)
        if int(a_i) in apo_to_delta
    ]
    keep_delta = [apo_to_delta[int(a_i)] for a_i in apo_idx if int(a_i) in apo_to_delta]
    modes = modes[:, keep_pred]
    delta_for_c = delta_aligned[keep_delta]
    rot, _cm, _ct = kabsch_rotation(pred_ca_pose, apo_m[keep_delta])
    modes_apo = np.einsum("kni,ij->knj", modes, rot)

    c10_exp = cumulative_subspace_overlap(modes_apo, delta_for_c, k=min(10, len(modes_apo)))
    c5_exp = cumulative_subspace_overlap(modes_apo, delta_for_c, k=min(5, len(modes_apo)))
    c10_null = null_c10_values(
        modes_apo, ca_idx, atom_mask, rigid_modes, n_null, generator, device, k=10
    )
    c5_null = null_c10_values(
        modes_apo, ca_idx, atom_mask, rigid_modes, n_null, generator, device, k=5
    )

    softer_than_null = float(np.mean(kappa_null > kappa_exp))
    c10_percentile = float(np.mean(c10_exp > c10_null))
    c5_percentile = float(np.mean(c5_exp > c5_null))
    nearest = nearest_experimental_state(pred_apo, pred_holo)

    return {
        "system": system,
        "holo_id": spec["holo_id"],
        "motion_type": spec["motion_type"],
        "rmsd_table": float(spec["rmsd_apo_holo"]),
        "rmsd_recomputed": float(rmsd),
        "pred_rmsd_apo": pred_apo,
        "pred_rmsd_holo": pred_holo,
        "pred_nearest_state": nearest,
        "n_ca": float(len(ca_idx)),
        "sigma": float(sigma),
        "ascent_score_rms": float(ascent.score_rms),
        "ascent_converged": float(ascent.converged),
        "C5_exp": c5_exp,
        "C10_exp": c10_exp,
        "C5_null_mean": float(np.mean(c5_null)),
        "C10_null_mean": float(np.mean(c10_null)),
        "C5_enriched_fraction": c5_percentile,
        "C10_enriched_fraction": c10_percentile,
        "kappa_exp": kappa_exp,
        "kappa_null_mean": float(np.mean(kappa_null)),
        "kappa_null_std": float(np.std(kappa_null)),
        "kappa_softer_fraction": softer_than_null,
        "kappa_z": float((kappa_exp - np.mean(kappa_null)) / max(np.std(kappa_null), 1e-12)),
        "n_null": float(n_null),
        "n_soft_modes": float(len(modes_apo)),
        "lambda_soft": float(evals[0]),
        "cif": str(cif_path),
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--processed_dir", type=Path, required=True)
    parser.add_argument("--predict_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--n_null", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out_dir = (
        args.out_dir if args.out_dir is not None else EXP_ROOT / "outputs" / "analysis_soft91"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.system}_soft_enrichment.json"
    if json_path.exists() and not args.force:
        print(f"skip {args.system}: exists", flush=True)
        return
    row = evaluate_system(
        args.system,
        args.processed_dir,
        args.predict_root,
        n_null=args.n_null,
    )
    json_path.write_text(json.dumps(row, indent=2))
    print(
        f"{args.system}  ΔRMSD={row['rmsd_table']:.2f}Å  nearest={row['pred_nearest_state']}  "
        f"C10={row['C10_exp']:.3f}  enriched={row['C10_enriched_fraction']:.3f}  "
        f"κ_softer={row['kappa_softer_fraction']:.3f}",
        flush=True,
    )
    print(f"wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
