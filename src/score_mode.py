"""Projected EDM score and score-ascent to a stationary pose.

Iterating the denoiser x ← D(x,σ) is not mode-seeking at finite σ: D is a
conditional mean. The EDM score s = (D−x)/σ² is ∇ log p_σ, so projected
ascent on s walks toward a stationary point used for Hessian soft modes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor

from rigid import internal_dimension, kabsch_align, project_out_rigid, rigid_basis
from slq import masked_norm

DenoiseFn = Callable[[Tensor], Tensor]
MatVec = Callable[[Tensor], Tensor]


class ScoreState(NamedTuple):
    """Projected score and residual at one pose.

    Attributes:
        score: Internal-coordinate score (B, M, 3).
        score_rms: RMS of the projected score (Å⁻¹).
        residual_rms: RMS of D(x,σ)−x on real atoms (Å).
    """

    score: Tensor
    score_rms: float
    residual_rms: float


class AscentResult(NamedTuple):
    """Outcome of projected score ascent.

    Attributes:
        coords: Final pose.
        n_steps: Updates applied.
        score_rms: Final projected score RMS.
        residual_rms: Final RMS(D−x).
        score_rms0: Score RMS at the start.
        residual_rms0: RMS(D−x) at the start.
        displacement_rms: RMS move from the starting pose after Kabsch.
        converged: Whether a tolerance was met.
        n_rigid: Rigid modes projected out.
        history: Per-step diagnostics.
    """

    coords: Tensor
    n_steps: int
    score_rms: float
    residual_rms: float
    score_rms0: float
    residual_rms0: float
    displacement_rms: float
    converged: bool
    n_rigid: int
    history: list[dict[str, float]]


def projected_score(
    denoise: DenoiseFn,
    coords: Tensor,
    sigma: float,
    atom_mask: Tensor,
    basis: Tensor,
) -> ScoreState:
    """Evaluate s = (D−x)/σ² and remove rigid-body components.

    Args:
        denoise: Map x ↦ D(x, σ).
        coords: Current pose.
        sigma: EDM noise level (Å).
        atom_mask: Real-atom mask.
        basis: Orthonormal rigid modes at ``coords``.

    Returns:
        Projected score and RMS diagnostics.
    """
    residual = (denoise(coords) - coords) * atom_mask.unsqueeze(-1)
    n_int = internal_dimension(atom_mask, int(basis.shape[0]))
    residual_rms = float(torch.sqrt((residual**2).sum() / max(n_int, 1.0)).item())
    score = project_out_rigid(residual / (float(sigma) ** 2), basis, atom_mask)
    score_rms = float(torch.sqrt((score**2).sum() / max(n_int, 1.0)).item())
    return ScoreState(score=score, score_rms=score_rms, residual_rms=residual_rms)


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


def _rms_displacement(left: Tensor, right: Tensor, atom_mask: Tensor, n_int: float) -> float:
    """RMS of a Kabsch-aligned displacement.

    Args:
        left: First pose.
        right: Second pose.
        atom_mask: Real-atom mask.
        n_int: Internal dimension used as the RMS denominator.

    Returns:
        RMS in Å.
    """
    aligned = kabsch_align(left, right, atom_mask)
    delta = (aligned - right) * atom_mask.unsqueeze(-1)
    return float(torch.sqrt((delta**2).sum() / max(n_int, 1.0)).item())


def score_ascent(
    denoise: DenoiseFn,
    coords: Tensor,
    sigma: float,
    atom_mask: Tensor,
    n_steps: int,
    lr: float,
    score_tol: float,
    residual_tol: float,
    method: str = "adam",
) -> AscentResult:
    """Walk toward ∇ log p_σ = 0 with a projected score.

    Args:
        denoise: Map x ↦ D(x, σ).
        coords: Starting pose (Boltz prediction).
        sigma: EDM noise level (Å).
        atom_mask: Real-atom mask.
        n_steps: Maximum updates.
        lr: Adam step size (Å²-like) or initial line-search step (Å²).
        score_tol: Stop when RMS of the projected score is below this (Å⁻¹).
        residual_tol: Stop when RMS(D−x) is below this (Å).
        method: ``adam`` or ``linesearch``.

    Returns:
        Final pose and convergence diagnostics.
    """
    pose = _center(coords, atom_mask)
    start = pose.clone()
    basis = rigid_basis(pose, atom_mask)
    n_int = internal_dimension(atom_mask, int(basis.shape[0]))
    state = projected_score(denoise, pose, sigma, atom_mask, basis)
    score_rms0 = state.score_rms
    residual_rms0 = state.residual_rms
    history: list[dict[str, float]] = [
        {"step": 0.0, "score_rms": score_rms0, "residual_rms": residual_rms0, "lr": lr}
    ]
    adam_m = torch.zeros_like(pose)
    adam_v = torch.zeros_like(pose)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    eta = lr
    used = 0
    for step_i in range(1, max(n_steps, 0) + 1):
        if state.score_rms < score_tol or state.residual_rms < residual_tol:
            break
        used = step_i
        direction = state.score
        if method == "adam":
            adam_m = beta1 * adam_m + (1.0 - beta1) * direction
            adam_v = beta2 * adam_v + (1.0 - beta2) * direction.square()
            m_hat = adam_m / (1.0 - beta1**step_i)
            v_hat = adam_v / (1.0 - beta2**step_i)
            trial = _center(pose + eta * m_hat / (v_hat.sqrt() + eps), atom_mask)
            trial_basis = rigid_basis(trial, atom_mask)
            trial_state = projected_score(denoise, trial, sigma, atom_mask, trial_basis)
            if trial_state.score_rms > 1.25 * state.score_rms:
                eta *= 0.5
                adam_m = torch.zeros_like(pose)
                adam_v = torch.zeros_like(pose)
                if eta < 1e-10:
                    break
                continue
            pose, basis, state = trial, trial_basis, trial_state
            eta = min(eta * 1.05, lr)
        else:
            accepted = False
            eta_try = eta
            dir_sq = float(masked_dot(direction, direction, atom_mask).item())
            for _ in range(8):
                trial = _center(pose + eta_try * direction, atom_mask)
                trial_basis = rigid_basis(trial, atom_mask)
                trial_state = projected_score(denoise, trial, sigma, atom_mask, trial_basis)
                disp = (trial - pose) * atom_mask.unsqueeze(-1)
                dlogp = 0.5 * float(
                    masked_dot(direction + trial_state.score, disp, atom_mask).item()
                )
                if trial_state.score_rms < state.score_rms or dlogp > 0.05 * eta_try * dir_sq:
                    pose, basis, state = trial, trial_basis, trial_state
                    eta = min(eta_try * 1.2, 10.0 * lr)
                    accepted = True
                    break
                eta_try *= 0.5
            if not accepted:
                break
        history.append(
            {
                "step": float(step_i),
                "score_rms": state.score_rms,
                "residual_rms": state.residual_rms,
                "lr": eta,
            }
        )
    pose = kabsch_align(pose, start, atom_mask)
    displacement_rms = _rms_displacement(pose, start, atom_mask, n_int)
    converged = state.score_rms < score_tol or state.residual_rms < residual_tol
    return AscentResult(
        coords=pose,
        n_steps=used,
        score_rms=state.score_rms,
        residual_rms=state.residual_rms,
        score_rms0=score_rms0,
        residual_rms0=residual_rms0,
        displacement_rms=displacement_rms,
        converged=converged,
        n_rigid=int(basis.shape[0]),
        history=history,
    )
