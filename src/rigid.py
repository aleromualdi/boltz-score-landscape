"""Translation/rotation subspace for Cartesian Hessians.

A 3N-dimensional atom coordinate Hessian always has six rigid-body
near-null modes (three translations, three infinitesimal rotations
about the centroid). Those are not conformational. Projecting them
out leaves the internal-coordinate block whose logdet is the local
Gaussian volume.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from slq import masked_dot, masked_norm


def _masked_com(coords: Tensor, atom_mask: Tensor) -> Tensor:
    """Masked centroid, shape (B, 1, 3).

    Args:
        coords: Atom coordinates (B, M, 3).
        atom_mask: Real-atom mask (B, M).

    Returns:
        Centroid of real atoms.
    """
    weight = atom_mask.unsqueeze(-1)
    return (coords * weight).sum(dim=1, keepdim=True) / weight.sum(dim=1, keepdim=True).clamp_min(
        1.0
    )


def rigid_basis(coords: Tensor, atom_mask: Tensor, min_norm: float = 1e-6) -> Tensor:
    """Orthonormal translation and rotation modes at a pose.

    Args:
        coords: Atom coordinates (B, M, 3).
        atom_mask: Real-atom mask (B, M).
        min_norm: Drop a candidate mode if its residual norm is below this.

    Returns:
        Stacked modes of shape (K, B, M, 3) with K ≤ 6.
    """
    mask = atom_mask.float()
    zeros = torch.zeros_like(coords)
    rel = (coords - _masked_com(coords, mask)) * mask.unsqueeze(-1)
    candidates = []
    for axis in range(3):
        trans = zeros.clone()
        trans[..., axis] = mask
        candidates.append(trans)
    candidates.append(torch.stack([zeros[..., 0], -rel[..., 2], rel[..., 1]], dim=-1))
    candidates.append(torch.stack([rel[..., 2], zeros[..., 0], -rel[..., 0]], dim=-1))
    candidates.append(torch.stack([-rel[..., 1], rel[..., 0], zeros[..., 0]], dim=-1))

    basis: list[Tensor] = []
    for raw in candidates:
        vec = raw * mask.unsqueeze(-1)
        for old in basis:
            vec = vec - masked_dot(vec, old, atom_mask) * old
        nrm = masked_norm(vec, atom_mask)
        if float(nrm.item()) < min_norm:
            continue
        basis.append(vec / nrm)
    if not basis:
        msg = "rigid-body basis is empty"
        raise RuntimeError(msg)
    return torch.stack(basis, dim=0)


def project_out_rigid(vec: Tensor, basis: Tensor, atom_mask: Tensor) -> Tensor:
    """Remove the rigid-body components of a displacement.

    Args:
        vec: Vector (B, M, 3).
        basis: Orthonormal rigid modes (K, B, M, 3).
        atom_mask: Real-atom mask (B, M).

    Returns:
        Internal-coordinate component of ``vec``.
    """
    out = vec * atom_mask.unsqueeze(-1)
    for mode in basis:
        out = out - masked_dot(out, mode, atom_mask) * mode
    return out


def internal_dimension(atom_mask: Tensor, n_rigid: int) -> float:
    """Dimension of the internal (rigid-free) subspace.

    Args:
        atom_mask: Real-atom mask (B, M).
        n_rigid: Number of retained rigid modes (usually 6).

    Returns:
        3N − K.
    """
    return float(atom_mask.sum().item()) * 3.0 - float(n_rigid)


def kabsch_align(mobile: Tensor, target: Tensor, atom_mask: Tensor) -> Tensor:
    """Rigidly align ``mobile`` onto ``target`` (Kabsch).

    Args:
        mobile: Coordinates to rotate/translate (B, M, 3).
        target: Reference coordinates (B, M, 3).
        atom_mask: Real-atom mask (B, M).

    Returns:
        Aligned copy of ``mobile``, padded atoms zeroed.
    """
    mask = atom_mask.unsqueeze(-1)
    com_m = _masked_com(mobile, atom_mask)
    com_t = _masked_com(target, atom_mask)
    centered = (mobile - com_m) * mask
    cov = torch.matmul(centered.transpose(-1, -2), (target - com_t) * mask)
    u_mat, _, vh_mat = torch.linalg.svd(cov)
    rot = torch.matmul(u_mat, vh_mat)
    reflect = torch.ones(mobile.shape[0], 3, device=mobile.device, dtype=mobile.dtype)
    reflect[:, -1] = torch.where(
        torch.det(rot) < 0,
        torch.full((), -1.0, device=mobile.device, dtype=mobile.dtype),
        reflect[:, -1],
    )
    rot = torch.matmul(u_mat, torch.matmul(torch.diag_embed(reflect), vh_mat))
    aligned = torch.matmul(mobile - com_m, rot) + com_t
    return aligned * mask


def projected_hvp(
    matvec: Callable[[Tensor], Tensor],
    basis: Tensor,
    atom_mask: Tensor,
) -> Callable[[Tensor], Tensor]:
    """Wrap a Hessian-vector product with P H P.

    Args:
        matvec: Unprojected map v ↦ Hv.
        basis: Orthonormal rigid modes.
        atom_mask: Real-atom mask.

    Returns:
        Internal-coordinate Hessian-vector product.
    """

    def matvec_internal(vec: Tensor) -> Tensor:
        """Apply P H P.

        Args:
            vec: Probe.

        Returns:
            Projected Hessian-vector product.
        """
        projected = project_out_rigid(vec, basis, atom_mask)
        return project_out_rigid(matvec(projected), basis, atom_mask)

    return matvec_internal


def rigid_projector(basis: Tensor, atom_mask: Tensor) -> Callable[[Tensor], Tensor]:
    """Build v ↦ P v with a frozen rigid basis.

    Args:
        basis: Orthonormal rigid modes.
        atom_mask: Real-atom mask.

    Returns:
        Projector callable.
    """

    def _project(vec: Tensor) -> Tensor:
        """Remove rigid components.

        Args:
            vec: Displacement.

        Returns:
            Internal component.
        """
        return project_out_rigid(vec, basis, atom_mask)

    return _project
