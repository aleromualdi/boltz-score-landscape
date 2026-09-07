"""Copy experimental apo heavy atoms into a Boltz all-atom pose."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from Bio.PDB import PDBParser
from torch import Tensor

from geom import AA3TO1, FloatArr, kabsch_rotation, matched_residue_indices, parse_label

type AtomMap = dict[str, FloatArr]


def _chosen_model(pdb_path: Path, model_number: int | None) -> Any:
    """Pick the Bio.PDB model for a chain-split benchmark file.

    Args:
        pdb_path: PDB path.
        model_number: Optional MODEL serial from the table id.

    Returns:
        Bio.PDB model.
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
    return chosen


def load_exp_residues(pdb_path: Path, model_number: int | None = None) -> tuple[str, list[AtomMap]]:
    """Load standard residues as atom-name → xyz maps.

    Args:
        pdb_path: Chain-split apo PDB.
        model_number: Optional NMR MODEL number.

    Returns:
        One-letter sequence and per-residue heavy-atom maps.

    Raises:
        ValueError: Too few residues.
    """
    chosen = _chosen_model(pdb_path, model_number)
    seq: list[str] = []
    residues: list[AtomMap] = []
    for residue in chosen.get_residues():
        hetflag, _resseq, _icode = residue.id
        if hetflag != " ":
            continue
        resname = residue.get_resname().upper()
        if resname not in AA3TO1 or "CA" not in residue:
            continue
        atoms: AtomMap = {}
        for atom in residue.get_atoms():
            name = atom.get_name().strip().upper()
            if atom.element == "H":
                continue
            atoms[name] = np.asarray(atom.coord, dtype=np.float64)
        seq.append(AA3TO1[resname])
        residues.append(atoms)
    if len(residues) < 10:
        msg = f"too few residues in {pdb_path}"
        raise ValueError(msg)
    return "".join(seq), residues


def boltz_residue_table(processed_dir: Path, system: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Boltz sequence and atom tables.

    Args:
        processed_dir: ``processed/`` folder.
        system: NPZ stem.

    Returns:
        Sequence, atoms structured array, residues structured array.
    """
    npz = np.load(processed_dir / "structures" / f"{system}.npz")
    atoms = npz["atoms"]
    residues = npz["residues"]
    seq = "".join(AA3TO1.get(str(row["name"]).upper(), "X") for row in residues)
    return seq, atoms, residues


def inject_apo_coords(
    predicted: Tensor,
    atom_mask: Tensor,
    processed_dir: Path,
    system: str,
    apo_pdb: Path,
    apo_label: str,
) -> tuple[Tensor, dict[str, float]]:
    """Replace Boltz coordinates with experimental apo heavy atoms.

    Predicted coordinates are Kabsch-superimposed onto apo CA, then every
    matching heavy atom is overwritten with the experimental position.
    Unmatched Boltz atoms keep the superimposed predicted coordinates.

    Args:
        predicted: Denoiser pose (B, M, 3).
        atom_mask: Real-atom mask.
        processed_dir: Boltz processed folder.
        system: Job stem.
        apo_pdb: Experimental apo PDB.
        apo_label: Table id (for MODEL parsing).

    Returns:
        Injected pose and mapping diagnostics.

    Raises:
        ValueError: Sequence mismatch that cannot be zipped.
    """
    _pdb, model_number, _chain = parse_label(apo_label)
    exp_seq, exp_res = load_exp_residues(apo_pdb, model_number)
    boltz_seq, atoms, residues = boltz_residue_table(processed_dir, system)
    boltz_idx, exp_idx = matched_residue_indices(boltz_seq, exp_seq)
    n_match = len(boltz_idx)
    ca_idx = np.asarray([i for i, name in enumerate(atoms["name"]) if name == "CA"], dtype=np.int64)
    pred_np = predicted[0].detach().cpu().numpy().astype(np.float64)
    n_real = len(atoms)
    pred_ca = pred_np[ca_idx[boltz_idx]]
    exp_ca = np.stack([exp_res[int(i)]["CA"] for i in exp_idx])
    rot, com_m, com_t = kabsch_rotation(pred_ca, exp_ca)
    superimposed = pred_np.copy()
    superimposed[:n_real] = (pred_np[:n_real] - com_m) @ rot + com_t
    n_copied = 0
    n_slots = 0
    for pair_i, res_i in enumerate(boltz_idx):
        start = int(residues[int(res_i)]["atom_idx"])
        n_atoms = int(residues[int(res_i)]["atom_num"])
        exp_atoms = exp_res[int(exp_idx[pair_i])]
        for offset in range(n_atoms):
            atom_i = start + offset
            n_slots += 1
            name = str(atoms[atom_i]["name"]).strip().upper()
            if name in exp_atoms:
                superimposed[atom_i] = exp_atoms[name]
                n_copied += 1
    injected = predicted.clone()
    injected[0] = torch.as_tensor(superimposed, device=predicted.device, dtype=predicted.dtype)
    injected = injected * atom_mask.unsqueeze(-1)
    ca_rmsd = float(np.sqrt(((superimposed[ca_idx[boltz_idx]] - exp_ca) ** 2).sum(1).mean()))
    stats = {
        "n_res_matched": float(n_match),
        "n_atoms_copied": float(n_copied),
        "n_atom_slots": float(n_slots),
        "copy_frac": float(n_copied / max(n_slots, 1)),
        "ca_rmsd_to_apo": ca_rmsd,
    }
    return injected, stats
