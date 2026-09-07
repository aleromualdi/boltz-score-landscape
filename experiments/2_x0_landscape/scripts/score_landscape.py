"""Measured Boltz score landscape around a score-stationary prediction.

Evaluates the score-derived potential U_model in the plane of the softest
and stiffest Hessian modes, and writes the three-system panel trio_3d_2d.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

EXP_ROOT = Path(__file__).resolve().parents[1]
EXP100 = EXP_ROOT.parents[0] / "1_codnas_apo_holo"
SRC = EXP_ROOT.parents[1] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(EXP100 / "scripts"))

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
from geom import (
    apo_holo_delta_aligned,
    cumulative_subspace_overlap,
    load_ca_sequence,
    parse_label,
    subspace_project_displacement,
)
from inject_apo import boltz_residue_table
from load_pose import coords_from_predicted_cif
from mode_interpret import (
    filter_overlap_sites,
    overlap_pred_indices,
)
from pairs import pair_by_apo, pdb_path
from rigid import (
    internal_dimension,
    project_out_rigid,
    projected_hvp,
    rigid_basis,
    rigid_projector,
)
from run_hessian import MSA_SUBSAMPLE, _center, ca_indices, soft_ca_modes
from score_mode import projected_score, score_ascent
from slq import masked_dot, masked_norm, unit_rademacher_probe

type FloatArr = NDArray[np.float64]

SOFT_COLOR = "#2563eb"
STIFF_COLOR = "#dc2626"
EXP_COLOR = "#ea580c"
DEMO_SYSTEMS: tuple[str, ...] = ("1EX6_B", "1GUD_A", "1AW2_A")


def find_processed_and_predict(system: str) -> tuple[Path, Path, Path]:
    """Locate processed dir, predict root, and CIF for ``system``.

    Args:
        system: Apo table id.

    Returns:
        ``(processed_dir, predict_root, cif_path)``.

    Raises:
        FileNotFoundError: Missing prediction artifacts.
    """
    # Prefer soft91 so landscape modes match Exp 100 C10 enrichment.
    for name in ("predict_soft91",):
        predict_root = EXP100 / "outputs" / name / system
        if not predict_root.is_dir():
            continue
        matches = sorted(predict_root.glob(f"boltz_results_*/predictions/*/{system}_model_0.cif"))
        processed = sorted(predict_root.glob("boltz_results_*/processed"))
        if matches and processed:
            return processed[0], predict_root, matches[0]
    msg = f"no prediction artifacts for {system}"
    raise FileNotFoundError(msg)


def ca_rms(delta: Tensor, ca_idx: np.ndarray, atom_mask: Tensor) -> float:
    """CA RMS of an all-atom displacement.

    Args:
        delta: Displacement (1, M, 3).
        ca_idx: CA atom indices.
        atom_mask: Real-atom mask.

    Returns:
        RMS in Å over CA sites.
    """
    ca = delta[0, ca_idx] * atom_mask[0, ca_idx].unsqueeze(-1)
    return float(torch.sqrt((ca**2).sum() / max(len(ca_idx), 1)).item())


def score_force_along_mode(
    denoise: Any,
    pose: Tensor,
    mode: Tensor,
    sigma: float,
    atom_mask: Tensor,
) -> float:
    """Directional derivative of the score-derived potential along ``mode``.

    With ``U = -log p_σ`` and ``s = ∇ log p_σ``, one has ``dU/dα = -s · v``
    for ``x = x0 + α v``.

    Args:
        denoise: Denoiser map.
        pose: Current coordinates.
        mode: Unit-CA-RMS displacement direction.
        sigma: Noise level.
        atom_mask: Real-atom mask.

    Returns:
        Scalar ``-⟨s, mode⟩``.
    """
    basis = rigid_basis(pose, atom_mask)
    score = projected_score(denoise, pose, sigma, atom_mask, basis).score
    return float((-masked_dot(score, mode, atom_mask)).item())


def integrate_potential(alphas: FloatArr, forces: FloatArr) -> FloatArr:
    """Trapezoidal integral of ``dU/dα`` with ``U(0)=0``.

    Args:
        alphas: Displacement grid (Å CA RMS), must include 0.
        forces: ``dU/dα`` samples.

    Returns:
        Potential relative to α=0.
    """
    order = np.argsort(alphas)
    a = alphas[order]
    f = forces[order]
    zero = int(np.argmin(np.abs(a)))
    pot = np.zeros_like(a)
    for i in range(zero + 1, len(a)):
        pot[i] = pot[i - 1] + 0.5 * (f[i] + f[i - 1]) * (a[i] - a[i - 1])
    for i in range(zero - 1, -1, -1):
        pot[i] = pot[i + 1] - 0.5 * (f[i + 1] + f[i]) * (a[i + 1] - a[i])
    out = np.empty_like(pot)
    out[order] = pot
    return out


def sweep_mode_potential(
    denoise: Any,
    x0: Tensor,
    mode: Tensor,
    sigma: float,
    atom_mask: Tensor,
    alphas: FloatArr,
) -> tuple[FloatArr, FloatArr, FloatArr]:
    """Evaluate the score-derived potential along one eigenmode.

    Args:
        denoise: Denoiser.
        x0: Score-stationary pose.
        mode: CA-RMS-normalized mode.
        sigma: Noise level.
        atom_mask: Real-atom mask.
        alphas: Displacement amplitudes (Å CA RMS).

    Returns:
        ``(alphas, U_model, force)``.
    """
    forces: FloatArr = np.zeros(len(alphas), dtype=np.float64)
    for i, alpha in enumerate(alphas):
        pose = _center(x0 + float(alpha) * mode, atom_mask)
        forces[i] = score_force_along_mode(denoise, pose, mode, sigma, atom_mask)
        print(f"    α={alpha:+.3f} Å  dU/dα={forces[i]:+.4g}", flush=True)
    potential = integrate_potential(alphas, forces)
    # Pin U(0)=0 exactly (numerical guard).
    zero = int(np.argmin(np.abs(alphas)))
    potential = potential - potential[zero]
    return alphas, potential, forces


def embed_ca_mode(
    mode_ca: FloatArr,
    ca_idx: np.ndarray,
    atom_mask: Tensor,
    pose: Tensor,
) -> Tensor:
    """Embed a CA mode into the all-atom tensor and rigid-project.

    Args:
        mode_ca: CA displacement (N_ca, 3).
        ca_idx: CA atom indices (same length as ``mode_ca``).
        atom_mask: Real-atom mask.
        pose: Pose used for the rigid basis.

    Returns:
        CA-RMS-normalized all-atom mode (1, M, 3).
    """
    device = atom_mask.device
    vec = torch.zeros(*atom_mask.shape, 3, dtype=torch.float32, device=device)
    ca = torch.as_tensor(mode_ca, dtype=torch.float32, device=device)
    # Match lengths if filtered.
    n = min(len(ca_idx), len(mode_ca))
    vec[0, ca_idx[:n]] = ca[:n]
    basis = rigid_basis(pose, atom_mask)
    vec = project_out_rigid(vec, basis, atom_mask)
    nrm = masked_norm(vec, atom_mask)
    if float(nrm.item()) < 1e-12:
        msg = "vanishing embedded CA mode"
        raise ValueError(msg)
    vec = vec / nrm
    rms = ca_rms(vec, ca_idx, atom_mask)
    return vec / rms


def load_ca_mode_trio(system: str) -> tuple[FloatArr, FloatArr, Path]:
    """Load soft / mid / stiff CA modes from Exp 100 score-ascent NPZ.

    Prefers soft91 (matches published C10), then local caches.

    Args:
        system: Apo table id.

    Returns:
        ``(modes (K,N,3), evals (K,), npz_path)``.

    Raises:
        FileNotFoundError: Missing modes file.
    """
    candidates = [
        EXP100 / "outputs" / "analysis_soft91" / f"{system}_score_ascent_ca_modes.npz",
        EXP_ROOT / "outputs" / "score_landscape" / f"{system}_score_ascent_ca_modes.npz",
                    ]
    for path in candidates:
        if path.is_file():
            data = np.load(path, allow_pickle=True)
            modes = np.asarray(data["modes"], dtype=np.float64)
            evals = np.asarray(data["evals"], dtype=np.float64)
            return modes, evals, path
    msg = f"no CA modes NPZ for {system}"
    raise FileNotFoundError(msg)


def compute_ca_modes_at_x0(
    model: Any,
    x0: Tensor,
    sigma: Tensor,
    atom_mask: Tensor,
    s_inputs: Tensor,
    s_trunk: Tensor,
    feats: dict[str, Tensor],
    cond: Any,
    ca_idx: np.ndarray,
    *,
    n_modes: int = 15,
    lanczos_steps: int = 40,
    out_npz: Path | None = None,
) -> tuple[FloatArr, FloatArr]:
    """Compute soft CA Hessian modes at the score-stationary pose.

    Matches Exp 100 ``directional_curvature`` / ``soft_ca_modes`` settings so
    soft-subspace fraction of Δx aligns with published ``C10``.

    Args:
        model: Boltz model.
        x0: Stationary coordinates.
        sigma: Noise level.
        atom_mask: Real-atom mask.
        s_inputs: Trunk single inputs.
        s_trunk: Trunk singles.
        feats: Feature dict.
        cond: Diffusion conditioning.
        ca_idx: CA indices.
        n_modes: Softest Ritz modes to keep.
        lanczos_steps: Krylov dimension.
        out_npz: Optional cache path.

    Returns:
        ``(modes (K,N,3), evals (K,))``.
    """
    rigid_modes = rigid_basis(x0, atom_mask)
    matvec = projected_hvp(
        make_autodiff_precision_hvp(
            model=model,
            coords=x0,
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
    generator = torch.Generator(device=x0.device)
    generator.manual_seed(0)
    probe = unit_rademacher_probe(
        atom_mask, generator, _center, rigid_projector(rigid_modes, atom_mask)
    )
    print(
        f"computing CA modes (n={n_modes}, lanczos={lanczos_steps})…",
        flush=True,
    )
    modes, evals = soft_ca_modes(
        matvec=matvec,
        probe=probe,
        atom_mask=atom_mask,
        rigid_modes=rigid_modes,
        ca_idx=ca_idx,
        n_modes=n_modes,
        lanczos_steps=lanczos_steps,
    )
    if out_npz is not None:
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        pred_ca = x0[0, ca_idx].detach().cpu().numpy().astype(np.float64)
        np.savez(out_npz, pred_ca=pred_ca, modes=modes, evals=evals, ca_idx=ca_idx)
        print(f"cached modes → {out_npz}", flush=True)
    return modes, evals


def choose_ca_trio(
    modes: FloatArr,
    evals: FloatArr,
) -> list[tuple[str, float, FloatArr]]:
    """Select soft / mid / stiff CA modes by eigenvalue rank.

    Args:
        modes: CA modes (K, N, 3), softest first.
        evals: Matching Ritz values.

    Returns:
        ``[(name, λ, mode_ca), ...]``.
    """
    order = np.argsort(evals)
    soft_i = int(order[0])
    stiff_i = int(order[-1])
    mid_i = int(order[len(order) // 2])
    trio = [
        ("soft", float(evals[soft_i]), modes[soft_i]),
        ("mid", float(evals[mid_i]), modes[mid_i]),
        ("stiff", float(evals[stiff_i]), modes[stiff_i]),
    ]
    # Normalize each CA mode to unit CA RMS.
    out: list[tuple[str, float, FloatArr]] = []
    for name, lam, mode in trio:
        rms = float(np.sqrt((mode**2).sum() / max(len(mode), 1)))
        out.append((name, lam, mode / max(rms, 1e-12)))
    return out


def experimental_projection_on_mode(
    mode_ca: FloatArr,
    delta_pose: FloatArr,
) -> tuple[float, float]:
    """Project apo→holo Δx onto a CA mode.

    Args:
        mode_ca: Mode (N, 3), approximately unit CA-RMS.
        delta_pose: Experimental displacement (N, 3) in the pose frame.

    Returns:
        ``(α_exp, cosine)`` where α_exp is the CA-RMS coordinate along the mode.
    """
    v = mode_ca.reshape(-1)
    d = delta_pose.reshape(-1)
    v_n = float(np.linalg.norm(v))
    d_n = float(np.linalg.norm(d))
    if v_n < 1e-18 or d_n < 1e-18:
        return 0.0, float("nan")
    alpha = float(d @ v) / (v_n**2)
    cosine = float(d @ v) / (d_n * v_n)
    return alpha, cosine


class DirectionFields(TypedDict):
    """Pose-frame vectors for the collective-motion overlay."""

    pred_ca: FloatArr
    delta_pose: FloatArr
    mode: FloatArr
    mode_index: int
    mode_cos: float
    delta_soft: FloatArr
    c10: float
    proj_cos: float
    evals: FloatArr


def best_mode_aligned(modes: FloatArr, delta: FloatArr) -> tuple[FloatArr, int, float]:
    """Pick the soft mode most aligned with ``delta`` and flip to positive cosine.

    Args:
        modes: Soft modes (K, N, 3).
        delta: Experimental displacement (N, 3).

    Returns:
        ``(aligned_mode, mode_index, cosine)``.
    """
    flat_d = delta.reshape(-1)
    dn = float(np.linalg.norm(flat_d))
    best_i = 0
    best_cos = 0.0
    best_abs = -1.0
    for i, mode in enumerate(modes):
        flat_m = mode.reshape(-1)
        mn = float(np.linalg.norm(flat_m))
        if mn < 1e-12 or dn < 1e-12:
            continue
        cosine = float(flat_m @ flat_d) / (mn * dn)
        if abs(cosine) > best_abs:
            best_abs = abs(cosine)
            best_cos = cosine
            best_i = i
    mode = np.asarray(modes[best_i], dtype=np.float64)
    if best_cos < 0.0:
        mode = -mode
        best_cos = -best_cos
    return mode, best_i, float(best_cos)


def project_xy(ca: FloatArr) -> FloatArr:
    """Project CA coordinates onto the leading principal plane.

    Args:
        ca: Coordinates (N, 3).

    Returns:
        2D points (N, 2).
    """
    xyz = ca - ca.mean(axis=0)
    _, _, vt = np.linalg.svd(xyz, full_matrices=False)
    return np.asarray(xyz @ vt[:2].T, dtype=np.float64)


def project_vectors(ca: FloatArr, vecs: FloatArr) -> FloatArr:
    """Project 3D vectors into the CA principal plane.

    Args:
        ca: CA coordinates (N, 3).
        vecs: Vectors (N, 3).

    Returns:
        Projected vectors (N, 2).
    """
    xyz = ca - ca.mean(axis=0)
    _, _, vt = np.linalg.svd(xyz, full_matrices=False)
    return np.asarray(vecs @ vt[:2].T, dtype=np.float64)


def load_direction_fields(system: str, out_dir: Path) -> DirectionFields:
    """Load overlap-matched CA, Δx, and best-aligned soft mode.

    Args:
        system: Apo table id.
        out_dir: Score-landscape output directory.

    Returns:
        Pose-frame fields for the 2D arrow overlay.

    Raises:
        FileNotFoundError: Missing landscape or modes NPZ.
        ValueError: CA count mismatch between landscape and modes.
    """
    land_path = out_dir / f"{system}_score_landscape.npz"
    if not land_path.is_file():
        msg = f"no saved landscape at {land_path}"
        raise FileNotFoundError(msg)
    land = np.load(land_path)
    pred_ca = np.asarray(land["pred_ca"], dtype=np.float64)
    delta_pose = np.asarray(land["delta_pose"], dtype=np.float64)
    local_modes = out_dir / f"{system}_score_ascent_ca_modes.npz"
    if local_modes.is_file():
        packed = np.load(local_modes, allow_pickle=True)
        ca_modes = np.asarray(packed["modes"], dtype=np.float64)
        evals = np.asarray(packed["evals"], dtype=np.float64)
        modes_path = local_modes
    else:
        ca_modes, evals, modes_path = load_ca_mode_trio(system)
    if ca_modes.shape[1] != pred_ca.shape[0]:
        spec = pair_by_apo()[system]
        processed_dir, _pred_root, _cif = find_processed_and_predict(system)
        boltz_seq, _atoms, _res = boltz_residue_table(processed_dir, system)
        _p, apo_model, _ = parse_label(spec["apo_id"])
        _p, holo_model, _ = parse_label(spec["holo_id"])
        apo_ca, apo_seq = load_ca_sequence(pdb_path(spec["apo_id"]), apo_model)
        holo_ca, holo_seq = load_ca_sequence(pdb_path(spec["holo_id"]), holo_model)
        apo_m, delta_exp, _rmsd, apo_match_idx, _ = apo_holo_delta_aligned(
            apo_ca, apo_seq, holo_ca, holo_seq
        )
        pred_full = np.asarray(np.load(modes_path)["pred_ca"], dtype=np.float64)
        pred_ca, ca_modes, _modes_apo, _delta_apo, delta_pose = filter_overlap_sites(
            boltz_seq,
            apo_seq,
            apo_match_idx,
            pred_full,
            ca_modes,
            delta_exp,
            apo_m,
        )
    mode, mode_i, mode_cos = best_mode_aligned(ca_modes, delta_pose)
    delta_soft = subspace_project_displacement(ca_modes, delta_pose, k=10)
    c10 = cumulative_subspace_overlap(ca_modes, delta_pose, k=min(10, len(ca_modes)))
    return {
        "pred_ca": pred_ca,
        "delta_pose": delta_pose,
        "mode": mode,
        "mode_index": int(mode_i),
        "mode_cos": float(mode_cos),
        "delta_soft": delta_soft,
        "c10": float(c10),
        "proj_cos": float(np.sqrt(max(c10, 0.0))),
        "evals": np.asarray(evals, dtype=np.float64),
    }


def draw_subspace_arrows(ax: Any, fields: DirectionFields) -> None:
    """Draw Δx and its 10-mode soft-subspace reconstruction on the same scale.

    No mode is selected and no eigenvector sign is chosen. Blue arrows are
    ``V_10 V_10^T Δx``, so missing length is motion outside the soft subspace.

    Args:
        ax: Matplotlib axes.
        fields: Pose-frame coordinates and vectors, including ``delta_soft``.
    """
    pred_ca = np.asarray(fields["pred_ca"], dtype=np.float64)
    delta_pose = np.asarray(fields["delta_pose"], dtype=np.float64)
    delta_soft = np.asarray(fields["delta_soft"], dtype=np.float64)
    xy = project_xy(pred_ca)
    ax.plot(xy[:, 0], xy[:, 1], color="0.8", lw=0.9, zorder=0)
    ax.scatter(xy[:, 0], xy[:, 1], s=6, c="0.55", zorder=1)
    d2 = project_vectors(pred_ca, delta_pose)
    s2 = project_vectors(pred_ca, delta_soft)
    span = float(np.ptp(xy[:, 0]) + np.ptp(xy[:, 1])) / 2.0
    target = 0.045 * max(span, 1.0)
    max_len = 0.10 * max(span, 1.0)
    amp_d = np.linalg.norm(d2, axis=1)
    positive = amp_d > 1e-12
    ref = float(np.quantile(amp_d[positive], 0.90)) if np.any(positive) else 1.0
    scale = target / max(ref, 1e-8)
    d2n = d2 * scale
    s2n = s2 * scale
    peak = float(np.max(np.linalg.norm(d2n, axis=1)))
    if peak > max_len:
        shrink = max_len / peak
        d2n = d2n * shrink
        s2n = s2n * shrink
    stride = max(len(xy) // 45, 2)
    idx = np.arange(0, len(xy), max(stride, 1))
    keep = amp_d[idx] > np.quantile(amp_d[idx], 0.55)
    sel = idx[keep]
    ax.quiver(
        xy[sel, 0],
        xy[sel, 1],
        d2n[sel, 0],
        d2n[sel, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=EXP_COLOR,
        width=0.004,
        alpha=0.9,
        label=r"$\Delta x$",
    )
    ax.quiver(
        xy[sel, 0],
        xy[sel, 1],
        s2n[sel, 0],
        s2n[sel, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=SOFT_COLOR,
        width=0.004,
        alpha=0.9,
        label=r"$\Delta x_{\mathrm{soft}}$",
    )
    pad = 0.08 * max(span, 1.0)
    x0 = float(xy[:, 0].min()) - pad
    x1 = float(xy[:, 0].max()) + pad
    y0 = float(xy[:, 1].min()) - pad
    y1 = float(xy[:, 1].max()) + pad
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    half = 0.5 * max(x1 - x0, y1 - y0)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=7, frameon=False)


def draw_eigenmode_softness(
    ax: Any,
    evals: FloatArr,
    *,
    k_soft: int = 10,
) -> None:
    """Boxplot of C_k eigenvalues versus the remaining Ritz values.

    Lower ``lambda`` is softer. The left box is the first ``k_soft`` modes
    used for ``C_10``; the right box is the rest of the Lanczos stack.

    Args:
        ax: Matplotlib axes.
        evals: Ritz eigenvalues, any order.
        k_soft: Number of lowest modes in the soft subspace.
    """
    lam = np.asarray(evals, dtype=np.float64)
    sorted_lam = np.sort(lam)
    k_keep = min(k_soft, len(sorted_lam))
    groups = [sorted_lam[:k_keep]]
    labels = [rf"$C_{{{k_keep}}}$"]
    colors = [SOFT_COLOR]
    if k_keep < len(sorted_lam):
        groups.append(sorted_lam[k_keep:])
        labels.append("rest")
        colors.append("#94a3b8")
    boxes = ax.boxplot(
        groups,
        tick_labels=labels,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#0f172a", "lw": 1.6},
        whiskerprops={"color": "#334155", "lw": 1.1},
        capprops={"color": "#334155", "lw": 1.1},
        zorder=2,
    )
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.4)
    ax.set_xlim(0.55, len(groups) + 0.45)
    ax.set_ylabel(r"$\lambda$  (smaller = softer)")


def plane_potential_grid(
    denoise: Any,
    x0: Tensor,
    mode1: Tensor,
    mode2: Tensor,
    sigma: float,
    atom_mask: Tensor,
    *,
    alpha_max: float,
    n_grid: int,
) -> tuple[FloatArr, FloatArr, FloatArr]:
    """Evaluate score-derived potential on a (q1, q2) grid around x0.

    At each grid point ``Δ = q1 v1 + q2 v2`` uses the two-point trapezoid

    ``U(q) ≈ -½ ⟨s(x0) + s(x0+Δ), Δ⟩``

    so anharmonicity is visible (not a Hessian ellipse).

    Args:
        denoise: Denoiser.
        x0: Stationary pose.
        mode1: Soft mode (CA-RMS unit).
        mode2: Second soft mode (CA-RMS unit).
        sigma: Noise level.
        atom_mask: Real-atom mask.
        alpha_max: Half-range in Å CA RMS.
        n_grid: Points per axis.

    Returns:
        ``(Q1, Q2, U)`` meshgrids.
    """
    axis: FloatArr = np.linspace(-alpha_max, alpha_max, n_grid, dtype=np.float64)
    q1_grid: FloatArr
    q2_grid: FloatArr
    q1_grid, q2_grid = np.meshgrid(axis, axis)
    u_grid = np.zeros_like(q1_grid)
    basis0 = rigid_basis(x0, atom_mask)
    s0 = projected_score(denoise, x0, sigma, atom_mask, basis0).score
    total = n_grid * n_grid
    done = 0
    for i in range(n_grid):
        for j in range(n_grid):
            q1 = float(q1_grid[i, j])
            q2 = float(q2_grid[i, j])
            delta = q1 * mode1 + q2 * mode2
            if abs(q1) < 1e-12 and abs(q2) < 1e-12:
                u_grid[i, j] = 0.0
            else:
                pose = _center(x0 + delta, atom_mask)
                basis = rigid_basis(pose, atom_mask)
                score = projected_score(denoise, pose, sigma, atom_mask, basis).score
                # Trapezoid ∫ -s·dx from 0→Δ.
                u_grid[i, j] = float((-0.5 * masked_dot(s0 + score, delta, atom_mask)).item())
            done += 1
            if done % max(n_grid, 1) == 0 or done == total:
                print(f"    plane grid {done}/{total}", flush=True)
    # Pin U(0,0)=0 exactly.
    mid = n_grid // 2
    u_grid = u_grid - u_grid[mid, mid]
    return q1_grid, q2_grid, u_grid


def fit_local_hessian(
    q1: FloatArr,
    q2: FloatArr,
    energy: FloatArr,
    *,
    radius: float = 0.40,
) -> tuple[float, float, float]:
    """Least-squares local Hessian of ``U`` in a disk around the origin.

    Fits ``U ≈ ½ κ₁ q₁² + ½ κ₂ q₂² + κ₁₂ q₁ q₂``, i.e. the local Hessian
    of the measured score-derived potential (same units as ``U_model``).

    Args:
        q1: Meshgrid along v1 (Å CA RMS).
        q2: Meshgrid along v2 (Å CA RMS).
        energy: Measured ``U_model`` on the mesh.
        radius: Fit disk radius in Å CA RMS.

    Returns:
        ``(κ1, κ2, κ12)``.
    """
    rad = np.sqrt(q1**2 + q2**2)
    mask = (rad > 1e-12) & (rad <= radius)
    if int(np.count_nonzero(mask)) < 4:
        spacing = float(np.min(rad[rad > 1e-12])) if np.any(rad > 1e-12) else radius
        mask = (rad > 1e-12) & (rad <= max(radius, 2.2 * spacing))
    x1 = q1[mask].reshape(-1)
    x2 = q2[mask].reshape(-1)
    y = energy[mask].reshape(-1)
    design = np.column_stack((0.5 * x1**2, 0.5 * x2**2, x1 * x2))
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


def build_soft_stiff_exp_plane(
    ca_modes: FloatArr,
    ca_evals: FloatArr,
    delta_pose: FloatArr,
    keep_pred: list[int] | np.ndarray,
    *,
    k_soft: int = 10,
) -> tuple[FloatArr, FloatArr, float, float, float]:
    """Softest × stiffest plane with experimental Δx projected as arrow.

    Contours of measured ``U`` elongate along the softest Hessian mode (see
    1D soft/stiff sweeps). The apo→holo arrow is the projection of Δx into
    this plane; ``soft_frac`` is the soft-subspace fraction (same as ``C_k``).

    Args:
        ca_modes: CA modes (K, N, 3), softest-first Ritz order.
        ca_evals: Matching eigenvalues.
        delta_pose: Experimental Δx on overlap sites (n_ov, 3).
        keep_pred: Overlap indices into the full construct.
        k_soft: Soft-subspace size for ``soft_frac``.

    Returns:
        ``(v_soft, v_stiff, q1_exp, q2_exp, soft_frac)``.
    """
    keep: NDArray[np.int64] = np.asarray(keep_pred, dtype=np.int64)
    n_ca = int(ca_modes.shape[1])
    order = np.argsort(ca_evals)
    if len(order) < 2:
        msg = "need at least two CA modes"
        raise ValueError(msg)
    soft_i = int(order[0])
    stiff_i = int(order[-1])
    v1 = np.asarray(ca_modes[soft_i], dtype=np.float64)
    v2 = np.asarray(ca_modes[stiff_i], dtype=np.float64)
    rms1 = float(np.sqrt((v1**2).sum() / max(n_ca, 1)))
    rms2 = float(np.sqrt((v2**2).sum() / max(n_ca, 1)))
    v1 = v1 / max(rms1, 1e-12)
    # Orthogonalize stiff against soft in CA Frobenius inner product.
    proj = float((v2 * v1).sum()) / float((v1 * v1).sum())
    v2 = v2 - proj * v1
    rms2 = float(np.sqrt((v2**2).sum() / max(n_ca, 1)))
    if rms2 < 1e-8:
        msg = "stiff mode parallel to softest mode"
        raise ValueError(msg)
    v2 = v2 / rms2
    delta_full: FloatArr = np.zeros((n_ca, 3), dtype=np.float64)
    delta_full[keep] = np.asarray(delta_pose, dtype=np.float64)
    q1_exp = float((delta_full * v1).sum()) / float((v1 * v1).sum())
    q2_exp = float((delta_full * v2).sum()) / float((v2 * v2).sum())
    if q1_exp < 0.0:
        v1 = -v1
        q1_exp = -q1_exp
    if q2_exp < 0.0:
        v2 = -v2
        q2_exp = -q2_exp
    # Soft-subspace fraction (Exp 100 C_k), including near-zero / negative λ.
    soft_idx = order[: min(k_soft, len(order))]
    basis: list[np.ndarray] = []
    for i in soft_idx:
        vec = np.asarray(ca_modes[int(i)][keep], dtype=np.float64).reshape(-1)
        for b in basis:
            vec = vec - float(vec @ b) * b
        nrm = float(np.linalg.norm(vec))
        if nrm > 1e-10:
            basis.append(vec / nrm)
    d = np.asarray(delta_pose, dtype=np.float64).reshape(-1)
    d_n2 = float(d @ d)
    if not basis or d_n2 < 1e-18:
        soft_frac = float("nan")
    else:
        bmat = np.column_stack(basis)
        coeffs = bmat.T @ d
        soft_frac = float((coeffs @ coeffs) / d_n2)
    return v1, v2, q1_exp, q2_exp, soft_frac


def attach_delta_u_colorbar(
    fig: Any,
    mappable: Any,
    ax: Any,
    *,
    fraction: float = 0.046,
    pad: float = 0.04,
    cax: Any | None = None,
) -> None:
    """Attach a relative-potential colorbar with no magnitude ticks.

    ``U_model`` is pinned at the score-stationary pose, so only the sign of
    the scale (zero at ``x_0``) is meaningful.

    Args:
        fig: Matplotlib figure.
        mappable: Contour or image mappable.
        ax: Axes the colorbar is attached to (ignored if ``cax`` is set).
        fraction: Colorbar width as a fraction of the parent axes.
        pad: Padding between the axes and the colorbar.
        cax: Optional pre-allocated colorbar axes.
    """
    if cax is not None:
        cbar = fig.colorbar(mappable, cax=cax)
    else:
        cbar = fig.colorbar(mappable, ax=ax, fraction=fraction, pad=pad)
    cbar.set_label(r"$\Delta U_{\mathrm{model}}$")
    cbar.ax.yaxis.set_major_locator(plt.NullLocator())
    cbar.ax.yaxis.set_minor_locator(plt.NullLocator())
    cbar.ax.text(
        0.5,
        0.02,
        r"$0$",
        transform=cbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#0f172a",
    )


def place_apo_holo_label(ax: Any, q1_m: float, q2_m: float, half: float) -> None:
    """Write the apo→holo label in black, clipped inside the axes.

    Args:
        ax: Matplotlib axes.
        q1_m: Arrow tip q1 (already clipped).
        q2_m: Arrow tip q2 (already clipped).
        half: Axes half-range.
    """
    pad = 0.18 * max(half, 1e-6)
    length = float(np.hypot(q1_m, q2_m))
    if length < 0.28 * half:
        scale = 0.42 * half / max(length, 1e-12)
        lx = q1_m * scale
        ly = q2_m * scale
    else:
        lx = 0.38 * q1_m
        ly = 0.38 * q2_m
    if abs(q1_m) >= abs(q2_m):
        ly += 0.12 * half if q2_m >= 0.0 else -0.12 * half
    else:
        lx += 0.12 * half if q1_m >= 0.0 else -0.12 * half
    lx = float(np.clip(lx, -half + pad, half - pad))
    ly = float(np.clip(ly, -half + pad, half - pad))
    ax.text(
        lx,
        ly,
        r"apo$\rightarrow$holo",
        color="#0f172a",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=10,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )


def _draw_contour_panel(
    ax: Any,
    q1: FloatArr,
    q2: FloatArr,
    energy: FloatArr,
    *,
    q1_exp: float,
    q2_exp: float,
    cmap: str = "YlOrRd",
) -> Any:
    """Draw the 2D measured landscape with apo→holo overlay.

    Args:
        ax: Matplotlib axes.
        q1: Meshgrid along softest.
        q2: Meshgrid along stiffest.
        energy: Measured ``U_model``.
        q1_exp: Experimental q1.
        q2_exp: Experimental q2.
        cmap: Colormap name.

    Returns:
        ContourSet from ``contourf`` (for a colorbar).
    """
    e_hi = float(np.nanmax(energy))
    filled = ax.contourf(
        q1,
        q2,
        energy,
        levels=np.linspace(0.0, max(e_hi, 1e-6), 18),
        cmap=cmap,
    )
    ax.contour(
        q1,
        q2,
        energy,
        levels=np.linspace(0.12 * max(e_hi, 1e-6), 0.75 * max(e_hi, 1e-6), 6),
        colors="#0f172a",
        linewidths=1.2,
    )
    half = float(max(np.max(np.abs(q1)), np.max(np.abs(q2))))
    q1_m = float(np.clip(q1_exp, -0.92 * half, 0.92 * half))
    q2_m = float(np.clip(q2_exp, -0.92 * half, 0.92 * half))
    ax.annotate(
        "",
        xy=(0.50 * half, 0.0),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "color": SOFT_COLOR, "lw": 1.6, "mutation_scale": 11},
    )
    ax.annotate(
        "",
        xy=(0.0, 0.50 * half),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "color": STIFF_COLOR, "lw": 1.6, "mutation_scale": 11},
    )
    ax.text(0.52 * half, -0.12 * half, "soft", color=SOFT_COLOR, fontsize=9, fontweight="bold")
    ax.text(0.04 * half, 0.52 * half, "stiff", color=STIFF_COLOR, fontsize=9, fontweight="bold")
    ax.scatter([0.0], [0.0], s=36, c="#0f172a", zorder=7)
    ax.annotate(r"$x_0$", xy=(0.0, 0.0), xytext=(-0.22 * half, -0.18 * half), color="#0f172a")
    ax.annotate(
        "",
        xy=(q1_m, q2_m),
        xytext=(0.0, 0.0),
        arrowprops={
            "arrowstyle": "-|>",
            "color": EXP_COLOR,
            "lw": 3.0,
            "mutation_scale": 16,
        },
        zorder=8,
    )
    ax.scatter(
        [q1_m],
        [q2_m],
        s=64,
        c=EXP_COLOR,
        zorder=9,
        edgecolors="white",
        linewidths=0.8,
    )
    place_apo_holo_label(ax, q1_m, q2_m, half)
    ax.set_aspect("equal")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_xlabel(r"$q_1$ along softest (Å CA RMS)")
    ax.set_ylabel(r"$q_2$ along stiffest (Å CA RMS)")
    ax.axhline(0.0, color="0.55", lw=0.5)
    ax.axvline(0.0, color="0.55", lw=0.5)
    return filled


def _draw_surface_panel(
    ax: Any,
    q1: FloatArr,
    q2: FloatArr,
    energy: FloatArr,
    *,
    q1_exp: float,
    q2_exp: float,
    cmap: str = "YlOrRd",
) -> Any:
    """Draw a 3D surface of measured ``U`` (no apo→holo overlay).

    Args:
        ax: 3D matplotlib axes.
        q1: Meshgrid along softest.
        q2: Meshgrid along stiffest.
        energy: Measured ``U_model``.
        q1_exp: Experimental q1.
        q2_exp: Experimental q2.
        cmap: Colormap name.

    Returns:
        The ``Poly3DCollection`` from ``plot_surface``.
    """
    e_hi = float(np.nanmax(energy))
    half = float(max(np.max(np.abs(q1)), np.max(np.abs(q2))))
    stride = max(1, q1.shape[0] // 24)
    surf = ax.plot_surface(
        q1,
        q2,
        energy,
        cmap=cmap,
        linewidth=0.15,
        edgecolor="#7c2d12",
        antialiased=True,
        rstride=stride,
        cstride=stride,
        alpha=0.93,
        vmin=0.0,
        vmax=max(e_hi, 1e-6),
    )
    ax.contour(
        q1,
        q2,
        energy,
        levels=np.linspace(0.12 * max(e_hi, 1e-6), 0.75 * max(e_hi, 1e-6), 6),
        colors="#0f172a",
        linewidths=0.6,
        zdir="z",
        offset=0.0,
    )
    ax.plot([0.0], [0.0], [0.0], "o", color="#0f172a", markersize=6, zorder=6)
    _ = q1_exp, q2_exp
    ax.set_xlabel(r"$q_1$ soft")
    ax.set_ylabel(r"$q_2$ stiff")
    ax.set_zlabel(r"$\Delta U_{\mathrm{model}}$")
    ax.view_init(elev=28, azim=-60)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(0.0, max(e_hi, 1e-6))
    return surf


def plot_trio_surface_panel(
    systems: list[str],
    out_dir: Path,
    out_path: Path,
) -> None:
    """Three-system 3D surface + 2D contour + collective-direction composite.

    Args:
        systems: Apo table ids with saved NPZs.
        out_dir: Score-landscape output directory.
        out_path: Combined PNG path.
    """
    n = len(systems)
    fig = plt.figure(figsize=(19.2, 3.7 * n))
    for i, system in enumerate(systems):
        data = np.load(out_dir / f"{system}_score_landscape.npz")
        summary = json.loads((out_dir / f"{system}_score_landscape.json").read_text())
        q1 = np.asarray(data["plane_q1"], dtype=np.float64)
        q2 = np.asarray(data["plane_q2"], dtype=np.float64)
        energy = np.asarray(data["plane_u"], dtype=np.float64)
        q1_exp = float(np.asarray(data["plane_q1_exp"]))
        q2_exp = float(np.asarray(data["plane_q2_exp"]))
        ax3 = fig.add_subplot(n, 4, 4 * i + 1, projection="3d")
        ax2 = fig.add_subplot(n, 4, 4 * i + 2)
        axd = fig.add_subplot(n, 4, 4 * i + 3)
        axs = fig.add_subplot(n, 4, 4 * i + 4)
        _draw_surface_panel(ax3, q1, q2, energy, q1_exp=q1_exp, q2_exp=q2_exp)
        filled = _draw_contour_panel(ax2, q1, q2, energy, q1_exp=q1_exp, q2_exp=q2_exp)
        attach_delta_u_colorbar(fig, filled, ax2)
        fields = load_direction_fields(system, out_dir)
        draw_eigenmode_softness(
            axd,
            fields["evals"],
            k_soft=10,
        )
        draw_subspace_arrows(axs, fields)
        ax3.set_title(
            rf"{system}  $C_{{10}}={summary['C10']:.2f}$, $\Delta={summary['rmsd']:.2f}$ Å",
            fontsize=10,
        )
        ax2.set_title("2D contour", fontsize=10)
        axd.set_title(r"$C_{10}$ $\lambda$ vs rest", fontsize=10)
        axs.set_title(
            rf"$C_{{10}}={fields['c10']:.2f}$  (proj. cos $={fields['proj_cos']:.2f}$)",
            fontsize=10,
        )
    fig.suptitle(
        r"Measured $U_{\mathrm{model}}$, Hessian spectrum, and soft-subspace capture "
        r"(orange $\Delta x$; blue $\Delta x_{\mathrm{soft}}$)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_enrichment(system: str) -> dict[str, float]:
    """Load Exp 100 soft-enrichment scalars if present.

    Args:
        system: Apo table id.

    Returns:
        Dict with C10 / null means (NaN if missing).
    """
    for folder in ("analysis_soft91",):
        path = EXP100 / "outputs" / folder / f"{system}_soft_enrichment.json"
        if path.is_file():
            row = json.loads(path.read_text())
            return {
                "C10": float(row.get("C10_exp", float("nan"))),
                "C10_null": float(row.get("C10_null_mean", float("nan"))),
            }
    return {"C10": float("nan"), "C10_null": float("nan")}


def run_system(
    system: str,
    *,
    out_dir: Path,
    alpha_max: float = 2.0,
    n_alpha: int = 17,
    n_grid: int = 25,
    plane_skip: int = 0,
    lanczos_steps: int = 40,
    score_ascent_steps: int = 80,
    sigma_target: float = 0.28,
    num_subsampled_msa: int = MSA_SUBSAMPLE,
) -> dict[str, object]:
    """Compute actual score landscapes along soft/mid/stiff modes.

    Args:
        system: Apo table id.
        out_dir: Output directory.
        alpha_max: Half-range of CA-RMS displacement (Å).
        n_alpha: Grid points for 1D mode sweeps.
        n_grid: Points per axis for the 2D (v1, v2) landscape (odd; default 25).
        plane_skip: Unused (kept for CLI compatibility).
        lanczos_steps: Unused (CA modes reused from Exp 100).
        score_ascent_steps: Ascent iterations to x0.
        sigma_target: Target EDM σ.
        num_subsampled_msa: MSA depth.

    Returns:
        Summary metrics.
    """
    _ = plane_skip
    spec = pair_by_apo()[system]
    processed_dir, _predict_root, cif_path = find_processed_and_predict(system)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mol_dir = Path(get_cache_path()) / "mols"
    print(f"device={device} system={system} predict={_predict_root.name}", flush=True)

    batch = load_batch(system, mol_dir, processed=processed_dir)
    feats = batch_to_device(batch, device)
    atom_mask = feats["atom_pad_mask"].float()
    coords = coords_from_predicted_cif(
        cif_path=cif_path,
        atom_mask=atom_mask,
        processed_dir=processed_dir,
        system=system,
    )
    feats["coords"] = coords
    torch.manual_seed(0)
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
        """Denoiser at fixed σ.

        Args:
            current: Pose.

        Returns:
            Denoised coordinates.
        """
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
    x0 = ascent.coords
    print(
        f"ascent score_rms {ascent.score_rms0:.4g}→{ascent.score_rms:.4g}  "
        f"conv={ascent.converged}",
        flush=True,
    )
    rigid_modes = rigid_basis(x0, atom_mask)
    ca_idx = ca_indices(processed_dir, system)
    modes_cache = out_dir / f"{system}_score_ascent_ca_modes.npz"
    recompute_modes = False
    try:
        ca_modes, ca_evals, modes_npz = load_ca_mode_trio(system)
        # Prefer freshly computed soft91-matched cache when present.
        if modes_cache.is_file() and modes_npz != modes_cache:
            data = np.load(modes_cache, allow_pickle=True)
            ca_modes = np.asarray(data["modes"], dtype=np.float64)
            ca_evals = np.asarray(data["evals"], dtype=np.float64)
            modes_npz = modes_cache
        # Soft91 enrichment uses modes at soft91 x0 — recompute unless cache
        # already lives under score_landscape / analysis_soft91.
        use_cached = "analysis_soft91" in str(modes_npz) or modes_npz == modes_cache
        recompute_modes = not use_cached
    except FileNotFoundError:
        recompute_modes = True
    if recompute_modes:
        ca_modes, ca_evals = compute_ca_modes_at_x0(
            model,
            x0,
            sigma,
            atom_mask,
            s_inputs,
            s_trunk,
            feats,
            cond,
            ca_idx,
            n_modes=15,
            lanczos_steps=lanczos_steps,
            out_npz=modes_cache,
        )
        modes_npz = modes_cache
    # Align NPZ CA count with current ca_idx (same construct).
    if ca_modes.shape[1] != len(ca_idx):
        msg = (
            f"CA count mismatch: modes {ca_modes.shape[1]} vs structure {len(ca_idx)}"
        )
        raise ValueError(msg)
    trio = choose_ca_trio(ca_modes, ca_evals)
    print(f"CA modes from {modes_npz}:", flush=True)
    for name, lam, _mode in trio:
        print(f"  {name}: λ={lam:.4g}", flush=True)

    n_int = internal_dimension(atom_mask, int(rigid_modes.shape[0]))
    alphas: FloatArr = np.linspace(-alpha_max, alpha_max, n_alpha, dtype=np.float64)
    sweeps: dict[str, dict[str, FloatArr | float]] = {}
    mode_tensors: dict[str, Tensor] = {}
    named_meta: list[tuple[str, float, FloatArr]] = []
    for name, lam, mode_ca in trio:
        mode = embed_ca_mode(mode_ca, ca_idx, atom_mask, x0)
        print(f"sweep {name} (λ={lam:.4g})…", flush=True)
        internal_norm = float(masked_norm(mode, atom_mask).item())
        a, pot, force = sweep_mode_potential(_denoise, x0, mode, sigma, atom_mask, alphas)
        # Local curvature from central force slope (dashed overlay in CA-RMS).
        zero = int(np.argmin(np.abs(a)))
        i0 = max(zero - 1, 0)
        i1 = min(zero + 1, len(a) - 1)
        kappa = float((force[i1] - force[i0]) / max(a[i1] - a[i0], 1e-12))
        sweeps[name] = {
            "alphas": a,
            "potential": pot,
            "force": force,
            "lambda": lam,
            "kappa_local": kappa,
            "internal_per_ca_rms": internal_norm,
        }
        mode_tensors[name] = mode
        named_meta.append((name, lam, mode_ca))

    # Experimental Δx in pose frame on overlap sites.
    boltz_seq, _atoms, _res = boltz_residue_table(processed_dir, system)
    _p, apo_model, _ = parse_label(spec["apo_id"])
    _p, holo_model, _ = parse_label(spec["holo_id"])
    apo_ca, apo_seq = load_ca_sequence(pdb_path(spec["apo_id"]), apo_model)
    holo_ca, holo_seq = load_ca_sequence(pdb_path(spec["holo_id"]), holo_model)
    apo_m, delta_exp, rmsd_gt, apo_match_idx, _ = apo_holo_delta_aligned(
        apo_ca, apo_seq, holo_ca, holo_seq
    )
    pred_ca_full = x0[0, ca_idx].detach().cpu().numpy().astype(np.float64)
    modes_stack = np.stack([m for _n, _lam, m in named_meta], axis=0)
    pred_ca, modes_pose, _modes_apo_trio, _delta_apo, delta_pose = filter_overlap_sites(
        boltz_seq,
        apo_seq,
        apo_match_idx,
        pred_ca_full,
        modes_stack,
        delta_exp,
        apo_m,
    )
    soft_pose = modes_pose[0]
    alpha_exp, cosine_soft = experimental_projection_on_mode(soft_pose, delta_pose)
    # C10 from the full soft-mode stack (same basis as soft_frac / Exp 100).
    _pred, modes_pose_full, modes_apo_full, delta_apo_c, _dp = filter_overlap_sites(
        boltz_seq,
        apo_seq,
        apo_match_idx,
        pred_ca_full,
        ca_modes,
        delta_exp,
        apo_m,
    )
    _aligned, mode_i, mode_cos = best_mode_aligned(modes_pose_full, delta_pose)
    c10 = cumulative_subspace_overlap(
        modes_apo_full, delta_apo_c, k=min(10, len(modes_apo_full))
    )
    enrich = load_enrichment(system)
    c10_null = float(enrich["C10_null"])
    c10_pub = float(enrich["C10"])
    print(
        f"C10(modes)={c10:.3f}  C10(soft91 json)={c10_pub:.3f}  null={c10_null:.3f}",
        flush=True,
    )

    # --- Softest x stiffest landscape; apo->holo projects along soft ---
    keep_pred = overlap_pred_indices(boltz_seq, apo_seq, apo_match_idx)
    v1_ca, v2_ca, q1_exp, q2_exp, soft_frac = build_soft_stiff_exp_plane(
        ca_modes,
        ca_evals,
        delta_pose,
        keep_pred,
        k_soft=10,
    )
    print(
        f"plane axes: softest (α={q1_exp:.3f} Å) × stiffest (q2_exp={q2_exp:.3f} Å); "
        f"soft_frac={soft_frac:.2f}",
        flush=True,
    )
    mode1 = embed_ca_mode(v1_ca, ca_idx, atom_mask, x0)
    mode2 = embed_ca_mode(v2_ca, ca_idx, atom_mask, x0)
    if n_grid % 2 == 0:
        n_grid = n_grid + 1
    # Fixed local window so soft/stiff elongation is readable; arrow may clamp.
    plane_max = float(alpha_max)
    print(f"2D plane grid {n_grid}×{n_grid} (±{plane_max:.2f} Å)…", flush=True)
    q1_mesh, q2_mesh, u_mesh = plane_potential_grid(
        _denoise,
        x0,
        mode1,
        mode2,
        sigma,
        atom_mask,
        alpha_max=plane_max,
        n_grid=n_grid,
    )
    spacing = float(np.min(np.abs(np.unique(q1_mesh[0, :]))))
    kappa1, kappa2, kappa12 = fit_local_hessian(
        q1_mesh,
        q2_mesh,
        u_mesh,
        radius=max(0.40, 2.2 * spacing),
    )
    kappa_ratio = float(kappa2 / max(abs(kappa1), 1e-18))
    # Finite-amplitude elongation diagnostic along the plotted axes.
    mid = u_mesh.shape[0] // 2
    axis = q1_mesh[0, :]
    i1 = int(np.argmin(np.abs(axis - min(1.5, 0.75 * plane_max))))
    i2 = int(np.argmin(np.abs(q2_mesh[:, 0] - min(1.5, 0.75 * plane_max))))
    u_soft = float(u_mesh[mid, i1])
    u_stiff = float(u_mesh[i2, mid])
    print(
        f"measured κ_soft={kappa1:.4g} κ_stiff={kappa2:.4g} (ratio {kappa_ratio:.3g}); "
        f"U_stiff/U_soft at ~{axis[i1]:.2f} Å = {u_stiff / max(u_soft, 1e-9):.2f}",
        flush=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist raw curves + 2D plane.
    save: dict[str, Any] = {
        "alphas": alphas,
        "pred_ca": pred_ca,
        "soft_ca": soft_pose,
        "delta_pose": delta_pose,
        "alpha_exp": np.asarray(alpha_exp),
        "cosine_soft": np.asarray(cosine_soft),
        "plane_q1": q1_mesh,
        "plane_q2": q2_mesh,
        "plane_u": u_mesh,
        "plane_q1_exp": np.asarray(q1_exp),
        "plane_q2_exp": np.asarray(q2_exp),
        "plane_v1_ca": v1_ca,
        "plane_v2_ca": v2_ca,
        "plane_kappa1": np.asarray(kappa1),
        "plane_kappa2": np.asarray(kappa2),
        "plane_kappa12": np.asarray(kappa12),
        "plane_kappa_ratio": np.asarray(kappa_ratio),
        "plane_max": np.asarray(plane_max),
        "plane_soft_frac": np.asarray(soft_frac),
        "plane_u_soft_at_diag": np.asarray(u_soft),
        "plane_u_stiff_at_diag": np.asarray(u_stiff),
    }
    for name, data in sweeps.items():
        save[f"{name}_potential"] = np.asarray(data["potential"])
        save[f"{name}_force"] = np.asarray(data["force"])
        save[f"{name}_lambda"] = np.asarray(float(data["lambda"]))
        save[f"{name}_kappa_local"] = np.asarray(float(data["kappa_local"]))
        save[f"{name}_internal_per_ca_rms"] = np.asarray(float(data["internal_per_ca_rms"]))
    np.savez_compressed(out_dir / f"{system}_score_landscape.npz", **save)

    summary: dict[str, object] = {
        "system": system,
        "motion_type": spec["motion_type"],
        "rmsd": float(rmsd_gt),
        "sigma": float(sigma),
        "ascent_score_rms": float(ascent.score_rms),
        "ascent_converged": float(ascent.converged),
        "alpha_max": alpha_max,
        "plane_max": plane_max,
        "n_alpha": n_alpha,
        "n_grid": n_grid,
        "modes_npz": str(modes_npz),
        "modes": {
            name: {
                "lambda": float(data["lambda"]),
                "kappa_local": float(data["kappa_local"]),
            }
            for name, data in sweeps.items()
        },
        "alpha_exp_soft": alpha_exp,
        "cosine_soft": cosine_soft,
        "mode_index": int(mode_i),
        "mode_cos": float(mode_cos),
        "plane_q1_exp": q1_exp,
        "plane_q2_exp": q2_exp,
        "plane_kappa_exp": kappa1,
        "plane_kappa_perp": kappa2,
        "plane_kappa_ratio": kappa_ratio,
        "plane_soft_frac": soft_frac,
        "plane_u_ratio_at_1p5": float(u_stiff / max(u_soft, 1e-9)),
        "C10": c10,
        "C10_null": c10_null,
        "n_int": float(n_int),
    }
    (out_dir / f"{system}_score_landscape.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {out_dir / f'{system}_score_landscape.npz'}", flush=True)
    return summary


def main() -> None:
    """CLI entrypoint: compute missing landscapes, then write the trio figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=EXP_ROOT / "outputs" / "score_landscape",
    )
    parser.add_argument(
        "--systems",
        default=",".join(DEMO_SYSTEMS),
        help="Comma-separated apo ids (default: demo trio).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute landscapes even when NPZ files exist.",
    )
    args = parser.parse_args()
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for system in systems:
        npz_path = args.out_dir / f"{system}_score_landscape.npz"
        if npz_path.is_file() and not args.force:
            print(f"skip compute {system}: {npz_path} exists", flush=True)
            continue
        print(f"compute landscape {system}…", flush=True)
        run_system(system, out_dir=args.out_dir)
    plot_trio_surface_panel(
        systems,
        args.out_dir,
        args.out_dir / "trio_3d_2d.png",
    )
    print(f"wrote {args.out_dir / 'trio_3d_2d.png'}", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
