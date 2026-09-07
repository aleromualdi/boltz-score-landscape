"""Load a deposited Boltz CIF into the padded atom coordinate tensor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from Bio.PDB import MMCIFParser
from torch import Tensor


def find_predicted_cif(predict_root: Path, system: str) -> Path:
    """Locate ``{system}_model_0.cif`` under a predict output folder.

    Args:
        predict_root: Directory that contains ``boltz_results_*/``.
        system: Job stem.

    Returns:
        CIF path.

    Raises:
        FileNotFoundError: No model_0 CIF.
    """
    matches = sorted(predict_root.glob(f"boltz_results_*/predictions/*/{system}_model_0.cif"))
    if not matches:
        msg = f"no predicted CIF for {system} under {predict_root}"
        raise FileNotFoundError(msg)
    return matches[0]


def load_cif_atoms(cif_path: Path) -> tuple[list[str], np.ndarray]:
    """Read heavy-atom names and coordinates from a Boltz mmCIF in file order.

    Args:
        cif_path: Predicted structure.

    Returns:
        Atom names and array (N, 3).

    Raises:
        ValueError: Too few atoms.
    """
    structure = MMCIFParser(QUIET=True).get_structure(cif_path.stem, str(cif_path))
    names: list[str] = []
    coords: list[np.ndarray] = []
    for atom in structure.get_atoms():
        names.append(atom.get_name().strip().upper())
        coords.append(np.asarray(atom.coord, dtype=np.float64))
    if len(coords) < 10:
        msg = f"too few atoms in {cif_path}"
        raise ValueError(msg)
    return names, np.stack(coords, axis=0)


def coords_from_predicted_cif(
    cif_path: Path,
    atom_mask: Tensor,
    processed_dir: Path,
    system: str,
) -> Tensor:
    """Map a deposited CIF onto the padded Boltz atom tensor.

    Boltz writes CIF atoms in the same order as the unpadded processed NPZ.
    Padded slots stay zero under ``atom_mask``.

    Args:
        cif_path: ``*_model_0.cif``.
        atom_mask: Real-atom mask (B, M).
        processed_dir: Boltz ``processed/`` folder.
        system: NPZ stem.

    Returns:
        Coordinate tensor (B, M, 3) on ``atom_mask``'s device.

    Raises:
        ValueError: Atom-count or name mismatch with the processed table.
    """
    npz = np.load(processed_dir / "structures" / f"{system}.npz")
    atoms = npz["atoms"]
    cif_names, cif_xyz = load_cif_atoms(cif_path)
    n_real = len(atoms)
    if len(cif_xyz) != n_real:
        msg = f"{cif_path}: {len(cif_xyz)} atoms vs processed {n_real}"
        raise ValueError(msg)
    names = [str(name).strip().upper() for name in atoms["name"]]
    if names != cif_names:
        msg = f"atom-name order mismatch for {system}"
        raise ValueError(msg)
    pad = int(atom_mask.shape[1])
    out = torch.zeros(1, pad, 3, device=atom_mask.device, dtype=torch.float32)
    out[0, :n_real] = torch.as_tensor(cif_xyz, device=atom_mask.device, dtype=torch.float32)
    return out * atom_mask.unsqueeze(-1)
