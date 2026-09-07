"""ANM participation helpers for PSCDB retrieval."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type FloatArr = NDArray[np.float64]


def _anm_hessian(ca: FloatArr, cutoff: float, gamma: float) -> FloatArr:
    """Build a simple C-alpha ANM Hessian.

    Args:
        ca: CA coordinates ``(N, 3)``.
        cutoff: Contact cutoff in Angstrom.
        gamma: Uniform spring constant.

    Returns:
        ANM Hessian matrix ``(3N, 3N)``.
    """
    n_ca = ca.shape[0]
    hess = np.zeros((3 * n_ca, 3 * n_ca), dtype=np.float64)
    for i_site in range(n_ca):
        for j_site in range(i_site + 1, n_ca):
            diff = ca[j_site] - ca[i_site]
            dist = float(np.linalg.norm(diff))
            if dist <= 1e-8 or dist > cutoff:
                continue
            unit = diff / dist
            block = -gamma * np.outer(unit, unit)
            i_slice = slice(3 * i_site, 3 * i_site + 3)
            j_slice = slice(3 * j_site, 3 * j_site + 3)
            hess[i_slice, j_slice] += block
            hess[j_slice, i_slice] += block
            hess[i_slice, i_slice] -= block
            hess[j_slice, j_slice] -= block
    return hess


def anm_participation(ca: FloatArr, k_soft: int, cutoff: float, gamma: float) -> FloatArr:
    """Compute ANM participation score analogous to Boltz ``S_i``.

    Args:
        ca: CA coordinates ``(N, 3)``.
        k_soft: Number of non-rigid low modes to include.
        cutoff: ANM contact cutoff in Angstrom.
        gamma: Uniform spring constant.

    Returns:
        Per-residue ANM participation vector.
    """
    n_ca = ca.shape[0]
    if n_ca < 4:
        return np.zeros(n_ca, dtype=np.float64)
    hess = _anm_hessian(ca, cutoff=cutoff, gamma=gamma)
    evals, evecs = np.linalg.eigh(hess)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    keep = np.where(evals > 1e-8)[0]
    if len(keep) == 0:
        return np.zeros(n_ca, dtype=np.float64)
    take = keep[:k_soft]
    modes = evecs[:, take].T.reshape(len(take), n_ca, 3)
    return np.sum(np.linalg.norm(modes, axis=2) ** 2, axis=0)
