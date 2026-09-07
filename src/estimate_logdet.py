"""Shared Boltz score / Hessian helpers for experiment pipelines.

Provides model loading, trunk conditioning, denoiser evaluation, and
precision Hessian-vector products used for soft-mode analyses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
    get_cache_path,
)
from boltz.model.models.boltz2 import Boltz2
from rigid import projected_hvp
from slq import slq_logdet
from torch import Tensor

SKIP_KEYS = {
    "all_coords",
    "all_resolved_mask",
    "crop_to_all_atom_map",
    "chain_symmetries",
    "amino_acids_symmetries",
    "ligand_symmetries",
    "record",
    "affinity_mw",
}


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields of an inference batch to a device.

    Args:
        batch: Collated Boltz2 batch.
        device: Target device.

    Returns:
        Batch with tensors on `device`.
    """
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if key in SKIP_KEYS:
            out[key] = value
        else:
            out[key] = value.to(device)
    return out


def load_model(
    device: torch.device,
    subsample_msa: bool,
    use_kernels: bool = True,
    num_subsampled_msa: int = 32,
) -> Boltz2:
    """Load Boltz2 checkpoint in eval mode.

    Args:
        device: CUDA or CPU device.
        subsample_msa: Whether the trunk should subsample the MSA.
        use_kernels: Pairformer CUDA kernels. Disable for autodiff HVPs.
        num_subsampled_msa: MSA depth when ``subsample_msa`` is on. Exp51
            ensembles historically used 32; Boltz CLI default is 1024.

    Returns:
        Boltz2 module on `device`.
    """
    cache = Path(get_cache_path())
    ckpt = cache / "boltz2_conf.ckpt"
    if not ckpt.exists():
        msg = f"Missing checkpoint {ckpt}"
        raise FileNotFoundError(msg)
    diffusion_params = Boltz2DiffusionParams()
    predict_args = {
        "recycling_steps": 3,
        "sampling_steps": 50,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
        "pair_scale": 0.0,
    }
    model = Boltz2.load_from_checkpoint(
        ckpt,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=use_kernels,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(
            MSAModuleArgs(
                subsample_msa=subsample_msa,
                num_subsampled_msa=num_subsampled_msa,
                use_paired_feature=True,
            )
        ),
        steering_args=asdict(BoltzSteeringParams()),
    )
    model.eval()
    return model.to(device)


def load_batch(system: str, mol_dir: Path, processed: Path) -> dict[str, Any]:
    """Load one inference batch from an existing processed run.

    Args:
        system: System / construct name (for errors only).
        mol_dir: Boltz CCD mol directory.
        processed: Path to Boltz ``processed/`` directory.

    Returns:
        Collated batch on CPU.
    """
    proc = processed
    if not (proc / "manifest.json").exists():
        msg = f"Missing processed data at {proc} (system={system})"
        raise FileNotFoundError(msg)
    manifest = Manifest.load(proc / "manifest.json")
    extra = proc / "mols"
    data_module = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=proc / "structures",
        msa_dir=proc / "msa",
        mol_dir=mol_dir,
        num_workers=0,
        constraints_dir=(proc / "constraints") if (proc / "constraints").exists() else None,
        template_dir=(proc / "templates") if (proc / "templates").exists() else None,
        extra_mols_dir=extra if extra.exists() else None,
    )
    return next(iter(data_module.predict_dataloader()))


def center_coords(coords: Tensor, atom_mask: Tensor) -> Tensor:
    """Subtract the masked-atom centroid (sampler convention).

    Args:
        coords: Atom coordinates, shape (B, M, 3).
        atom_mask: 1 for real atoms, shape (B, M).

    Returns:
        Centered coordinates.
    """
    weight = atom_mask.unsqueeze(-1)
    com = (coords * weight).sum(dim=1, keepdim=True) / weight.sum(dim=1, keepdim=True).clamp_min(
        1.0
    )
    return coords - com


def denoise_coords(
    model: Boltz2,
    coords: Tensor,
    sigma: float,
    s_inputs: Tensor,
    s_trunk: Tensor,
    feats: dict[str, Tensor],
    diffusion_conditioning: dict[str, Tensor],
) -> Tensor:
    """Run the EDM denoiser D_θ(x, σ).

    Args:
        model: Boltz2 module.
        coords: Noisy or clean atom coords (B, M, 3).
        sigma: EDM noise level in the sampler's units.
        s_inputs: Input single embeddings.
        s_trunk: Trunk single embeddings.
        feats: Inference features.
        diffusion_conditioning: Cached diffusion conditioning tensors.

    Returns:
        Denoised coordinates (B, M, 3).
    """
    with torch.no_grad():
        return model.structure_module.preconditioned_network_forward(
            coords,
            sigma,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "feats": feats,
                "multiplicity": 1,
                "diffusion_conditioning": diffusion_conditioning,
            },
        )


def denoise_coords_grad(
    model: Boltz2,
    coords: Tensor,
    sigma: float,
    s_inputs: Tensor,
    s_trunk: Tensor,
    feats: dict[str, Tensor],
    diffusion_conditioning: dict[str, Tensor],
) -> Tensor:
    """Denoise with gradients enabled (for autodiff JVP).

    Args:
        model: Boltz2 module.
        coords: Atom coords (B, M, 3).
        sigma: EDM noise level.
        s_inputs: Input single embeddings.
        s_trunk: Trunk single embeddings.
        feats: Inference features.
        diffusion_conditioning: Cached diffusion conditioning tensors.

    Returns:
        Denoised coordinates (B, M, 3).
    """
    return model.structure_module.preconditioned_network_forward(
        coords,
        sigma,
        network_condition_kwargs={
            "s_inputs": s_inputs,
            "s_trunk": s_trunk,
            "feats": feats,
            "multiplicity": 1,
            "diffusion_conditioning": diffusion_conditioning,
        },
    )



def make_autodiff_precision_hvp(
    model: Boltz2,
    coords: Tensor,
    sigma: float,
    atom_mask: Tensor,
    s_inputs: Tensor,
    s_trunk: Tensor,
    feats: dict[str, Tensor],
    diffusion_conditioning: dict[str, Tensor],
    ridge: float,
) -> Callable[[Tensor], Tensor]:
    """Build v ↦ (H + εI)v with exact JVP of the denoiser.

    Args:
        model: Boltz2 module (kernels should be off).
        coords: Evaluation point.
        sigma: Noise level.
        atom_mask: Real-atom mask.
        s_inputs: Input embeddings.
        s_trunk: Trunk embeddings.
        feats: Features.
        diffusion_conditioning: Cached conditioning.
        ridge: Diagonal regularizer.

    Returns:
        Hessian-vector product on COM-free probes.
    """
    mask = atom_mask.float()
    sigma_sq = float(sigma) ** 2
    base = coords.detach()

    def denoise_x(pose: Tensor) -> Tensor:
        """Denoiser as a map on coordinates.

        Args:
            pose: Atom coords (B, M, 3).

        Returns:
            D_θ(pose, σ).
        """
        return denoise_coords_grad(
            model, pose, sigma, s_inputs, s_trunk, feats, diffusion_conditioning
        )

    def matvec(vec: Tensor) -> Tensor:
        """Apply the autodiff precision Hessian.

        Args:
            vec: Unit COM-free probe.

        Returns:
            (H + εI) vec.
        """
        _, jvp_vec = torch.autograd.functional.jvp(
            denoise_x,
            (base,),
            (vec.detach(),),
            create_graph=False,
            strict=False,
        )
        hv = (vec - jvp_vec) / sigma_sq + ridge * vec
        return center_coords(hv, mask) * mask.unsqueeze(-1)

    return matvec


def _conditioning_from_trunk(
    model: Boltz2,
    feats: dict[str, Tensor],
    s_trunk: Tensor,
    z_trunk: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Build diffusion conditioning tensors from trunk embeddings.

    Args:
        model: Boltz2 module.
        feats: Device batch.
        s_trunk: Single representation from the trunk.
        z_trunk: Pair representation from the trunk.

    Returns:
        ``s_inputs`` and the diffusion conditioning dict.
    """
    s_inputs = model.input_embedder(feats)
    rel_pos = model.rel_pos(feats)
    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
        model.diffusion_conditioning(
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            relative_position_encoding=rel_pos,
            feats=feats,
        )
    )
    conditioning = {
        "q": q,
        "c": c,
        "to_keys": to_keys,
        "atom_enc_bias": atom_enc_bias,
        "atom_dec_bias": atom_dec_bias,
        "token_trans_bias": token_trans_bias,
    }
    return s_inputs.float(), conditioning



def trunk_and_condition(
    model: Boltz2,
    feats: dict[str, Tensor],
    recycling_steps: int = 3,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Run the MSA/pairformer trunk only and rebuild diffusion conditioning.

    Skips structure sampling so a deposited CIF pose can be Hessian'd under
    freshly computed sequence-conditioned embeddings.

    Args:
        model: Boltz2 module.
        feats: Device batch (``feats['coords']`` unused when confidence is off).
        recycling_steps: Trunk recycling depth (Boltz predict default is 3).

    Returns:
        Tuple of (s_inputs, s_trunk, diffusion_conditioning).
    """
    previous_skip = bool(model.skip_run_structure)
    previous_conf = bool(model.confidence_prediction)
    model.skip_run_structure = True
    model.confidence_prediction = False
    try:
        with torch.no_grad():
            out = model(
                feats,
                recycling_steps=recycling_steps,
                num_sampling_steps=1,
                diffusion_samples=1,
                max_parallel_samples=1,
                run_confidence_sequentially=True,
            )
            s_trunk = out["s"]
            s_inputs, conditioning = _conditioning_from_trunk(
                model, feats, s_trunk, out["z"]
            )
    finally:
        model.skip_run_structure = previous_skip
        model.confidence_prediction = previous_conf
    return s_inputs, s_trunk.float(), conditioning


def closest_schedule_sigma(model: Boltz2, target: float) -> float:
    """Pick the Boltz2 sample-schedule σ closest to a target.

    Args:
        model: Boltz2 module (provides AtomDiffusion.sample_schedule).
        target: Desired noise level in Å.

    Returns:
        Schedule σ, excluding the final 0.
    """
    schedule = model.structure_module.sample_schedule(50)[:-1]
    values = [float(sigma.item()) for sigma in schedule]
    return min(values, key=lambda sigma: abs(sigma - target))


def make_precision_hvp(
    model: Boltz2,
    coords: Tensor,
    sigma: float,
    atom_mask: Tensor,
    s_inputs: Tensor,
    s_trunk: Tensor,
    feats: dict[str, Tensor],
    diffusion_conditioning: dict[str, Tensor],
    fd_eps: float,
    ridge: float,
    n_free: float,
    central: bool,
) -> Callable[[Tensor], Tensor]:
    """Build v ↦ (H + εI)v for the precision Hessian H = −∇² log p_σ.

    Uses the EDM identity H v = (v − J_D[v]) / σ². The finite-difference
    probe is scaled to unit RMS so the step ``fd_eps`` is in Å.

    Args:
        model: Boltz2 module.
        coords: Evaluation point (B, M, 3), already centered.
        sigma: Noise level.
        atom_mask: Real-atom mask (B, M).
        s_inputs: Input embeddings.
        s_trunk: Trunk embeddings.
        feats: Features.
        diffusion_conditioning: Cached conditioning.
        fd_eps: Finite-difference step (Å) on a unit-RMS probe.
        ridge: Added to the diagonal before Lanczos.
        n_free: COM-free dimension, used to scale the FD probe.
        central: If True, use central differences (2 denoise calls).

    Returns:
        Hessian-vector product callable on unit COM-free vectors.
    """
    mask = atom_mask.float()
    sigma_sq = float(sigma) ** 2
    fd_scale = n_free**0.5
    d0: Tensor | None = None
    if not central:
        d0 = denoise_coords(
            model, coords, sigma, s_inputs, s_trunk, feats, diffusion_conditioning
        )

    def matvec(vec: Tensor) -> Tensor:
        """Apply the regularized precision Hessian to a unit vector.

        Args:
            vec: Unit COM-free probe (B, M, 3).

        Returns:
            (H + εI) vec, COM-free on real atoms.
        """
        probe = vec * fd_scale
        if central:
            d_plus = denoise_coords(
                model,
                coords + fd_eps * probe,
                sigma,
                s_inputs,
                s_trunk,
                feats,
                diffusion_conditioning,
            )
            d_minus = denoise_coords(
                model,
                coords - fd_eps * probe,
                sigma,
                s_inputs,
                s_trunk,
                feats,
                diffusion_conditioning,
            )
            jvp_probe = (d_plus - d_minus) / (2.0 * fd_eps)
        else:
            if d0 is None:
                msg = "forward finite difference requires a cached D(x,σ)"
                raise RuntimeError(msg)
            d_plus = denoise_coords(
                model,
                coords + fd_eps * probe,
                sigma,
                s_inputs,
                s_trunk,
                feats,
                diffusion_conditioning,
            )
            jvp_probe = (d_plus - d0) / fd_eps
        jvp_vec = jvp_probe / fd_scale
        hv = (vec - jvp_vec) / sigma_sq + ridge * vec
        return center_coords(hv, mask) * mask.unsqueeze(-1)

    return matvec


def evaluate_sigma(
    model: Boltz2,
    coords: Tensor,
    sigma: float,
    atom_mask: Tensor,
    s_inputs: Tensor,
    s_trunk: Tensor,
    feats: dict[str, Tensor],
    diffusion_conditioning: dict[str, Tensor],
    probes: list[Tensor],
    n_free: float,
    fd_eps: float,
    ridge: float,
    lanczos_steps: int,
    central: bool,
    hvp: str,
    rigid_modes: Tensor | None = None,
) -> dict[str, float]:
    """SLQ logdet of the precision Hessian at one noise level.

    Args:
        model: Boltz2 module.
        coords: Centered evaluation pose.
        sigma: EDM noise level (Å).
        atom_mask: Real-atom mask.
        s_inputs: Input embeddings.
        s_trunk: Trunk embeddings.
        feats: Features.
        diffusion_conditioning: Cached conditioning.
        probes: Unit COM-free Hutchinson vectors (reused across σ).
        n_free: COM-free or internal-coordinate dimension.
        fd_eps: Finite-difference step (Å).
        ridge: Logdet floor.
        lanczos_steps: Krylov dimension.
        central: Central vs forward FD.
        hvp: ``fd`` or ``autodiff``.
        rigid_modes: If set, wrap the HVP with P H P.

    Returns:
        SLQ statistics plus denoiser residual.
    """
    d0 = denoise_coords(model, coords, sigma, s_inputs, s_trunk, feats, diffusion_conditioning)
    d0_rms = float(torch.sqrt(((d0 - coords) ** 2 * atom_mask.unsqueeze(-1)).sum() / n_free).item())
    if hvp == "autodiff":
        matvec = make_autodiff_precision_hvp(
            model=model,
            coords=coords,
            sigma=sigma,
            atom_mask=atom_mask,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            feats=feats,
            diffusion_conditioning=diffusion_conditioning,
            ridge=ridge,
        )
    else:
        matvec = make_precision_hvp(
            model=model,
            coords=coords,
            sigma=sigma,
            atom_mask=atom_mask,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            feats=feats,
            diffusion_conditioning=diffusion_conditioning,
            fd_eps=fd_eps,
            ridge=ridge,
            n_free=n_free,
            central=central,
        )
    if rigid_modes is not None:
        matvec = projected_hvp(matvec, rigid_modes, atom_mask)
    stats = slq_logdet(
        matvec=matvec,
        probes=probes,
        atom_mask=atom_mask,
        n_free=n_free,
        n_steps=lanczos_steps,
        ridge=ridge,
    )
    stats["d0_rms_update"] = d0_rms
    stats["sigma"] = float(sigma)
    return stats
