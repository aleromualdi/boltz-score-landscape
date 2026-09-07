"""Stochastic Lanczos quadrature for Tr[log(H + εI)].

Given Hessian-vector products v ↦ Hv, Hutchinson probes plus an m-step
Lanczos tridiagonalization estimate Tr[f(H)] for f = log. See Ubaru, Chen,
and Saad, SIAM J. Matrix Anal. Appl. 38 (2017).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import math

import numpy as np
import torch
from torch import Tensor

MatVec = Callable[[Tensor], Tensor]


class LanczosResult(NamedTuple):
    """Tridiagonal Lanczos factorization of one probe.

    Attributes:
        alpha: Diagonal of T, length m_eff.
        beta: Off-diagonal of T, length m_eff - 1.
        n_steps: Number of completed Lanczos steps.
        basis: Krylov basis, empty unless ``keep_basis`` was set.
    """

    alpha: np.ndarray
    beta: np.ndarray
    n_steps: int
    basis: list[Tensor]


class ProbeQuadrature(NamedTuple):
    """Gauss quadrature of f(H) along one Hutchinson probe.

    Attributes:
        tau_log: e1ᵀ log(T) e1 with Ritz values clamped to ``ridge``.
        tau_log_plus: Same using only strictly positive Ritz values.
        tau_lin: e1ᵀ T e1 ≈ vᵀ H v for unit v.
        min_ritz: Smallest Ritz value of T.
        max_ritz: Largest Ritz value of T.
        n_clamped: How many Ritz values were clamped to ``ridge``.
        n_negative: How many Ritz values were ≤ 0.
        n_steps: Lanczos steps used.
    """

    tau_log: float
    tau_log_plus: float
    tau_lin: float
    min_ritz: float
    max_ritz: float
    n_clamped: int
    n_negative: int
    n_steps: int


def masked_dot(left: Tensor, right: Tensor, atom_mask: Tensor) -> Tensor:
    """Inner product on real atoms.

    Args:
        left: Vector, shape (B, M, 3).
        right: Vector, shape (B, M, 3).
        atom_mask: 1 for real atoms, shape (B, M).

    Returns:
        Scalar inner product.
    """
    return (left * right * atom_mask.unsqueeze(-1)).sum()


def masked_norm(vec: Tensor, atom_mask: Tensor) -> Tensor:
    """Euclidean norm on real atoms.

    Args:
        vec: Vector, shape (B, M, 3).
        atom_mask: 1 for real atoms, shape (B, M).

    Returns:
        Scalar L2 norm.
    """
    return masked_dot(vec, vec, atom_mask).sqrt().clamp_min(1e-12)


def lanczos_tridiagonal(
    matvec: MatVec,
    start: Tensor,
    atom_mask: Tensor,
    n_steps: int,
    breakdown: float = 1e-8,
    keep_basis: bool = False,
) -> LanczosResult:
    """m-step Lanczos with full reorthogonalization.

    Args:
        matvec: Map v ↦ (H + εI)v, COM-free on real atoms.
        start: Starting vector, normalized internally.
        atom_mask: Real-atom mask (B, M).
        n_steps: Maximum Krylov dimension.
        breakdown: Stop if the next off-diagonal is below this.
        keep_basis: If True, retain the Krylov vectors for Ritz reconstruction.

    Returns:
        Diagonal and off-diagonal of the Jacobi matrix T.
    """
    vec = start / masked_norm(start, atom_mask)
    basis: list[Tensor] = [vec]
    alphas: list[float] = []
    betas: list[float] = []
    w_vec = matvec(vec)
    alpha = float(masked_dot(vec, w_vec, atom_mask).item())
    alphas.append(alpha)
    w_vec = w_vec - alpha * vec
    prev = vec
    for _ in range(1, n_steps):
        beta = float(masked_norm(w_vec, atom_mask).item())
        if beta < breakdown:
            break
        betas.append(beta)
        vec = w_vec / beta
        for old in basis:
            vec = vec - masked_dot(vec, old, atom_mask) * old
        vec = vec / masked_norm(vec, atom_mask)
        basis.append(vec)
        w_vec = matvec(vec)
        alpha = float(masked_dot(vec, w_vec, atom_mask).item())
        alphas.append(alpha)
        w_vec = w_vec - alpha * vec - beta * prev
        prev = vec
    return LanczosResult(
        alpha=np.asarray(alphas, dtype=np.float64),
        beta=np.asarray(betas, dtype=np.float64),
        n_steps=len(alphas),
        basis=basis if keep_basis else [],
    )


def ritz_pairs(lanczos: LanczosResult, atom_mask: Tensor) -> list[tuple[float, Tensor]]:
    """Reconstruct Ritz pairs (θ, y) from a Lanczos factorization.

    Args:
        lanczos: Output of ``lanczos_tridiagonal(..., keep_basis=True)``.
        atom_mask: Real-atom mask (B, M).

    Returns:
        Eigenpairs of the Jacobi matrix, with y in the original space.
    """
    if not lanczos.basis:
        msg = "ritz_pairs requires lanczos_tridiagonal(..., keep_basis=True)"
        raise ValueError(msg)
    dim = int(lanczos.alpha.shape[0])
    tri = np.diag(lanczos.alpha)
    if dim > 1 and lanczos.beta.size:
        off = np.asarray(lanczos.beta[: dim - 1], dtype=np.float64)
        tri += np.diag(off, k=1) + np.diag(off, k=-1)
    evals, evecs = np.linalg.eigh(tri)
    pairs: list[tuple[float, Tensor]] = []
    for index, theta in enumerate(evals):
        vec = torch.zeros_like(lanczos.basis[0])
        for step in range(dim):
            vec = vec + float(evecs[step, index]) * lanczos.basis[step]
        vec = vec / masked_norm(vec, atom_mask)
        pairs.append((float(theta), vec))
    return pairs


def gauss_quadrature_log(alpha: np.ndarray, beta: np.ndarray, ridge: float) -> ProbeQuadrature:
    """Evaluate e1ᵀ log(T) e1 and e1ᵀ T e1 from a Jacobi matrix.

    Args:
        alpha: Diagonal of T.
        beta: Off-diagonal of T.
        ridge: Clamp Ritz values to this floor before taking log.

    Returns:
        Quadrature scalars and Ritz diagnostics.
    """
    dim = int(alpha.shape[0])
    tri = np.diag(alpha)
    if dim > 1 and beta.size:
        off = np.asarray(beta[: dim - 1], dtype=np.float64)
        tri += np.diag(off, k=1) + np.diag(off, k=-1)
    evals, evecs = np.linalg.eigh(tri)
    floor = float(ridge)
    n_clamped = int(np.sum(evals < floor))
    n_negative = int(np.sum(evals <= 0.0))
    clamped = np.maximum(evals, floor)
    weights = evecs[0, :] ** 2
    tau_log = float(np.dot(weights, np.log(clamped)))
    tau_lin = float(np.dot(weights, evals))
    positive = evals > 0.0
    if bool(positive.any()) and float(weights[positive].sum()) > 0.0:
        plus_w = weights[positive]
        plus_w = plus_w / plus_w.sum()
        tau_log_plus = float(np.dot(plus_w, np.log(evals[positive])))
    else:
        tau_log_plus = float("nan")
    return ProbeQuadrature(
        tau_log=tau_log,
        tau_log_plus=tau_log_plus,
        tau_lin=tau_lin,
        min_ritz=float(evals.min()),
        max_ritz=float(evals.max()),
        n_clamped=n_clamped,
        n_negative=n_negative,
        n_steps=dim,
    )


def slq_logdet(
    matvec: MatVec,
    probes: list[Tensor],
    atom_mask: Tensor,
    n_free: float,
    n_steps: int,
    ridge: float,
) -> dict[str, float]:
    """Hutchinson + Lanczos estimate of Tr[log(H+εI)] and Tr[H].

    Each probe must be a unit vector in the COM-free subspace. Then
    Tr[f(H)] ≈ n_free * mean_i e1ᵀ f(T_i) e1.

    Args:
        matvec: Map v ↦ (H + εI)v.
        probes: Unit COM-free Hutchinson vectors.
        atom_mask: Real-atom mask (B, M).
        n_free: Subspace dimension (typically 3N − 3).
        n_steps: Lanczos steps per probe.
        ridge: Floor for log and the ε already in ``matvec``.

    Returns:
        Trace estimates, S_hessian, isotropic comparison, Ritz stats.
    """
    quads: list[ProbeQuadrature] = []
    for probe in probes:
        lanczos = lanczos_tridiagonal(matvec, probe, atom_mask, n_steps)
        quads.append(gauss_quadrature_log(lanczos.alpha, lanczos.beta, ridge=ridge))
    tau_log = np.array([q.tau_log for q in quads], dtype=np.float64)
    tau_log_plus = np.array([q.tau_log_plus for q in quads], dtype=np.float64)
    tau_lin = np.array([q.tau_lin for q in quads], dtype=np.float64)
    logdet = float(n_free * tau_log.mean())
    plus_ok = np.isfinite(tau_log_plus)
    logdet_plus = (
        float(n_free * tau_log_plus[plus_ok].mean()) if bool(plus_ok.any()) else float("nan")
    )
    trace = float(n_free * tau_lin.mean())
    mean_prec = trace / n_free
    s_hess_nats = -0.5 * logdet
    s_plus_nats = -0.5 * logdet_plus
    s_iso_nats = -0.5 * n_free * float(np.log(max(mean_prec, ridge)))
    ln2 = float(np.log(2.0))
    two_pi_e = 2.0 * float(np.pi) * float(np.e)
    s_gauss_nats = 0.5 * n_free * float(np.log(two_pi_e)) + s_hess_nats
    s_iso_gauss_nats = 0.5 * n_free * float(np.log(two_pi_e)) + s_iso_nats
    gm_over_am = float(np.exp(logdet / n_free - np.log(max(mean_prec, ridge))))
    n_steps_tot = float(sum(q.n_steps for q in quads))
    n_negative = float(sum(q.n_negative for q in quads))
    return {
        "n_probes": float(len(quads)),
        "n_free": n_free,
        "logdet_H": logdet,
        "logdet_H_plus": logdet_plus,
        "trace_H": trace,
        "mean_precision": mean_prec,
        "width_iso": float("inf") if mean_prec <= 0 else 1.0 / mean_prec,
        "gm_over_am": gm_over_am,
        "S_hessian_nats": s_hess_nats,
        "S_hessian_bits": s_hess_nats / ln2,
        "S_hessian_per_coord_nats": s_hess_nats / n_free,
        "S_hessian_per_coord_bits": (s_hess_nats / ln2) / n_free,
        "S_plus_nats": s_plus_nats,
        "S_plus_per_coord_bits": (
            (s_plus_nats / ln2) / n_free if math.isfinite(s_plus_nats) else float("nan")
        ),
        "S_iso_nats": s_iso_nats,
        "S_iso_bits": s_iso_nats / ln2,
        "S_iso_per_coord_bits": (s_iso_nats / ln2) / n_free,
        "S_gaussian_bits": s_gauss_nats / ln2,
        "S_iso_gaussian_bits": s_iso_gauss_nats / ln2,
        "std_tau_log": float(tau_log.std(ddof=1) if len(tau_log) > 1 else 0.0),
        "std_tau_lin": float(tau_lin.std(ddof=1) if len(tau_lin) > 1 else 0.0),
        "min_ritz": float(min(q.min_ritz for q in quads)),
        "max_ritz": float(max(q.max_ritz for q in quads)),
        "n_clamped": float(sum(q.n_clamped for q in quads)),
        "n_negative": n_negative,
        "f_neg": n_negative / n_steps_tot if n_steps_tot else float("nan"),
        "mean_lanczos_steps": float(np.mean([q.n_steps for q in quads])),
    }


def unit_rademacher_probe(
    atom_mask: Tensor,
    generator: torch.Generator,
    center_fn: Callable[[Tensor, Tensor], Tensor],
    project_fn: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Draw a unit COM-free Rademacher vector on real atoms.

    Args:
        atom_mask: 1 for real atoms, shape (B, M).
        generator: RNG on the same device as ``atom_mask``.
        center_fn: Subtracts the masked centroid.
        project_fn: Optional extra projector (e.g. remove rotations).

    Returns:
        Unit probe, shape (B, M, 3).
    """
    signs = torch.randint(
        0,
        2,
        (*atom_mask.shape, 3),
        device=atom_mask.device,
        generator=generator,
        dtype=torch.int64,
    )
    probe = (2.0 * signs.float() - 1.0) * atom_mask.unsqueeze(-1)
    probe = center_fn(probe, atom_mask)
    if project_fn is not None:
        probe = project_fn(probe)
    return probe / masked_norm(probe, atom_mask)
