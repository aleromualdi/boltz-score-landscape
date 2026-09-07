"""Overlap filtering between Boltz poses and experimental apo↔holo sites.

Slim library kept for Exp 2 landscape analysis (formerly full soft-mode
interpretation helpers).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from geom import kabsch_rotation, matched_residue_indices

type FloatArr = NDArray[np.float64]


def overlap_pred_indices(
    boltz_seq: str,
    apo_seq: str,
    apo_match_idx: np.ndarray,
) -> np.ndarray:
    """Boltz residue indices overlapping experimental apo→holo sites.

    Args:
        boltz_seq: Boltz construct sequence.
        apo_seq: Experimental apo sequence.
        apo_match_idx: Apo indices used in ``apo_holo_delta_aligned``.

    Returns:
        0-based Boltz indices kept for overlap analysis.
    """
    pred_idx, apo_idx = matched_residue_indices(boltz_seq, apo_seq)
    apo_to_delta = {int(i): k for k, i in enumerate(apo_match_idx)}
    keep_pred = [
        int(p_i)
        for p_i, a_i in zip(pred_idx, apo_idx, strict=True)
        if int(a_i) in apo_to_delta
    ]
    return np.asarray(keep_pred, dtype=np.int64)


def filter_overlap_sites(
    boltz_seq: str,
    apo_seq: str,
    apo_match_idx: np.ndarray,
    pred_ca: FloatArr,
    modes: FloatArr,
    delta_aligned: FloatArr,
    apo_matched: FloatArr,
) -> tuple[FloatArr, FloatArr, FloatArr, FloatArr, FloatArr]:
    """Restrict modes and displacements to sequence-matched overlap sites.

    Matches the residue filtering used in ``directional_curvature.evaluate_system``.
    Returns pose-frame geometry for morphing and apo-frame geometry for ``C_k``.

    Args:
        boltz_seq: Boltz construct sequence.
        apo_seq: Experimental apo sequence.
        apo_match_idx: Apo indices used in ``apo_holo_delta_aligned``.
        pred_ca: Boltz CA at the stationary point (N_boltz, 3).
        modes: Soft modes (K, N_boltz, 3).
        delta_aligned: Experimental displacement on apo matched sites.
        apo_matched: Matched apo CA coordinates.

    Returns:
        ``pred_ca, modes_pose, modes_apo, delta_apo, delta_pose``.
    """
    pred_idx, apo_idx = matched_residue_indices(boltz_seq, apo_seq)
    apo_to_delta = {int(i): k for k, i in enumerate(apo_match_idx)}
    keep_pred = overlap_pred_indices(boltz_seq, apo_seq, apo_match_idx)
    keep_delta = [
        apo_to_delta[int(a_i)]
        for p_i, a_i in zip(pred_idx, apo_idx, strict=True)
        if int(a_i) in apo_to_delta
    ]
    pred_keep = pred_ca[keep_pred]
    delta_apo = delta_aligned[keep_delta]
    apo_keep = apo_matched[keep_delta]
    rot, _cm, _ct = kabsch_rotation(pred_keep, apo_keep)
    modes_pose = modes[:, keep_pred]
    modes_apo = np.einsum("kni,ij->knj", modes_pose, rot)
    delta_pose = delta_apo @ rot.T
    return pred_keep, modes_pose, modes_apo, delta_apo, delta_pose
