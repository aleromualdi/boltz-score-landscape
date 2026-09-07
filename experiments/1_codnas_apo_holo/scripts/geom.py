"""CA geometry: Kabsch, apo→holo Δx, and mode overlap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from numpy.typing import NDArray

type FloatArr = NDArray[np.float64]

AA3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def parse_label(label: str) -> tuple[str, int | None, str]:
    """Split a table id into PDB, optional MODEL, and chain.

    Args:
        label: ``1HSI_B`` or ``1AEL-12_A``.

    Returns:
        PDB code, model number or None, chain id.
    """
    body, chain = label.rsplit("_", 1)
    if "-" in body:
        pdb, maybe_model = body.rsplit("-", 1)
        if maybe_model.isdigit():
            return pdb, int(maybe_model), chain
    return body, None, chain


def load_ca_sequence(pdb_path: Path, model_number: int | None = None) -> tuple[FloatArr, str]:
    """Load CA coordinates and one-letter sequence from a chain-split PDB.

    Args:
        pdb_path: Benchmark PDB (already one chain).
        model_number: NMR MODEL number from the table id, or None.

    Returns:
        CA array (N, 3) and sequence.

    Raises:
        ValueError: No CA atoms.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    models = list(structure)
    chosen = models[0]
    if model_number is not None:
        for model in models:
            if int(model.id) == model_number or int(model.id) == model_number - 1:
                chosen = model
                break
    coords: list[list[float]] = []
    seq: list[str] = []
    for residue in chosen.get_residues():
        hetflag, _resseq, _icode = residue.id
        if hetflag != " ":
            continue
        resname = residue.get_resname().upper()
        if resname not in AA3TO1 or "CA" not in residue:
            continue
        coords.append([float(x) for x in residue["CA"].coord])
        seq.append(AA3TO1[resname])
    if len(coords) < 10:
        msg = f"too few CA atoms in {pdb_path}"
        raise ValueError(msg)
    return np.asarray(coords, dtype=np.float64), "".join(seq)


def kabsch_rotation(mobile: FloatArr, target: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:
    """Kabsch rotation mapping centered mobile onto centered target.

    Args:
        mobile: (N, 3).
        target: (N, 3).

    Returns:
        Rotation (3, 3), mobile centroid, target centroid.
    """
    n_atoms = min(len(mobile), len(target))
    mob = mobile[:n_atoms]
    tgt = target[:n_atoms]
    com_m = mob.mean(axis=0)
    com_t = tgt.mean(axis=0)
    # C = target^T mobile so that mobile @ R ≈ target (same as Bio.PDB Superimposer).
    cov = (tgt - com_t).T @ (mob - com_m)
    u_mat, _, vt_mat = np.linalg.svd(cov)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0:
        vt_mat[-1] *= -1
        rot = vt_mat.T @ u_mat.T
    return rot, com_m, com_t


def align_onto(mobile: FloatArr, target: FloatArr) -> FloatArr:
    """Return mobile coordinates superimposed onto target.

    Args:
        mobile: (N, 3).
        target: (N, 3).

    Returns:
        Aligned mobile, truncated to min length.
    """
    n_atoms = min(len(mobile), len(target))
    rot, com_m, com_t = kabsch_rotation(mobile, target)
    return (mobile[:n_atoms] - com_m) @ rot + com_t


def matched_residue_indices(seq_a: str, seq_b: str) -> tuple[np.ndarray, np.ndarray]:
    """Global-align two sequences and return matched residue index pairs.

    CoDNaS pairs are 100% identical after curation, but termini can differ
    between apo, holo, and the Boltz construct. Never truncate with ``min(N)``.

    Args:
        seq_a: First one-letter sequence.
        seq_b: Second one-letter sequence.

    Returns:
        Index arrays ``idx_a``, ``idx_b`` of equal length.

    Raises:
        ValueError: Alignment yields too few matched residues.
    """
    # Needleman-Wunsch with identity scoring (no external aligner dependency).
    n_a = len(seq_a)
    n_b = len(seq_b)
    score = np.zeros((n_a + 1, n_b + 1), dtype=np.float64)
    pointer = np.zeros((n_a + 1, n_b + 1), dtype=np.int8)  # 1=diag, 2=up, 3=left
    gap = -1.0
    for i in range(1, n_a + 1):
        score[i, 0] = i * gap
        pointer[i, 0] = 2
    for j in range(1, n_b + 1):
        score[0, j] = j * gap
        pointer[0, j] = 3
    for i in range(1, n_a + 1):
        for j in range(1, n_b + 1):
            match = score[i - 1, j - 1] + (1.0 if seq_a[i - 1] == seq_b[j - 1] else -1.0)
            delete = score[i - 1, j] + gap
            insert = score[i, j - 1] + gap
            best = match
            ptr = 1
            if delete > best:
                best = delete
                ptr = 2
            if insert > best:
                best = insert
                ptr = 3
            score[i, j] = best
            pointer[i, j] = ptr
    idx_a: list[int] = []
    idx_b: list[int] = []
    i, j = n_a, n_b
    while i > 0 or j > 0:
        ptr = int(pointer[i, j])
        if ptr == 1:
            idx_a.append(i - 1)
            idx_b.append(j - 1)
            i -= 1
            j -= 1
        elif ptr == 2:
            i -= 1
        elif ptr == 3:
            j -= 1
        elif i > 0:
            i -= 1
        else:
            j -= 1
    idx_a.reverse()
    idx_b.reverse()
    if len(idx_a) < 5:
        msg = f"alignment too short: {len(idx_a)} matched residues"
        raise ValueError(msg)
    return np.asarray(idx_a, dtype=np.int64), np.asarray(idx_b, dtype=np.int64)


def apo_holo_delta_aligned(
    apo: FloatArr,
    apo_seq: str,
    holo: FloatArr,
    holo_seq: str,
) -> tuple[FloatArr, FloatArr, float, np.ndarray, np.ndarray]:
    """Sequence-aligned apo→holo Δx after Kabsch on matched CA.

    Args:
        apo: Apo CA (full chain).
        apo_seq: Apo sequence.
        holo: Holo CA (full chain).
        holo_seq: Holo sequence.

    Returns:
        Matched apo CA, Δx on those sites, RMSD, apo indices, holo indices.
    """
    apo_idx, holo_idx = matched_residue_indices(apo_seq, holo_seq)
    apo_m = apo[apo_idx]
    holo_m = holo[holo_idx]
    aligned = align_onto(holo_m, apo_m)
    delta = aligned - apo_m
    rmsd = float(np.sqrt((delta**2).sum(axis=1).mean()))
    return apo_m, delta, rmsd, apo_idx, holo_idx


def cumulative_subspace_overlap(modes: FloatArr, delta: FloatArr, k: int | None = None) -> float:
    r"""Fraction of Δx captured by the first ``k`` soft modes after orthonormalization.

    Computes \(C_k = \|V_k V_k^\top \Delta x\|^2 / \|\Delta x\|^2\) where the
    columns of \(V_k\) are an orthonormal basis for the CA-projected soft
    subspace (QR on flattened modes). CA slices of all-atom Ritz vectors are
    not orthonormal, so this is the correct subspace measure.

    Args:
        modes: Soft modes (K, N, 3), softest first.
        delta: Experimental apo→holo displacement (N, 3).
        k: Number of modes to keep. ``None`` uses all rows of ``modes``.

    Returns:
        Cumulative overlap in [0, 1], or NaN if undefined.
    """
    if modes.size == 0:
        return float("nan")
    n_keep = len(modes) if k is None else min(k, len(modes))
    if n_keep < 1:
        return float("nan")
    n_atoms = min(modes.shape[1], len(delta))
    flat = modes[:n_keep, :n_atoms].reshape(n_keep, -1).T  # (3N, k)
    target = delta[:n_atoms].reshape(-1)
    denom = float(target @ target)
    if denom < 1e-18:
        return float("nan")
    # Thin QR; drop numerically zero columns.
    q_mat, r_mat = np.linalg.qr(flat, mode="reduced")
    rank = int(np.sum(np.abs(np.diag(r_mat)) > 1e-8))
    if rank < 1:
        return float("nan")
    basis = q_mat[:, :rank]
    proj = basis @ (basis.T @ target)
    return float((proj @ proj) / denom)


def subspace_project_displacement(
    modes: FloatArr,
    delta: FloatArr,
    k: int | None = None,
) -> FloatArr:
    """Return the component of ``delta`` in the top-``k`` soft subspace.

    Args:
        modes: Soft modes (K, N, 3), softest first.
        delta: Target displacement (N, 3).
        k: Number of modes. ``None`` uses all rows of ``modes``.

    Returns:
        Reconstructed displacement (N, 3) equal to ``V_k V_k^T delta`` after QR.
    """
    if modes.size == 0:
        msg = "cannot project with zero modes"
        raise ValueError(msg)
    n_keep = len(modes) if k is None else min(k, len(modes))
    if n_keep < 1:
        msg = "k must be at least 1"
        raise ValueError(msg)
    n_atoms = min(modes.shape[1], len(delta))
    flat = modes[:n_keep, :n_atoms].reshape(n_keep, -1).T
    target = delta[:n_atoms].reshape(-1)
    q_mat, r_mat = np.linalg.qr(flat, mode="reduced")
    rank = int(np.sum(np.abs(np.diag(r_mat)) > 1e-8))
    if rank < 1:
        msg = "soft subspace rank is zero"
        raise ValueError(msg)
    basis = q_mat[:, :rank]
    proj = basis @ (basis.T @ target)
    return proj.reshape(n_atoms, 3)
