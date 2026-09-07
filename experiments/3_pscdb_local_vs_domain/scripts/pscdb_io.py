"""PSCDB identifier parsing and structure I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
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

CLASSES = ("CL", "IL", "CD", "ID")


def parse_pscdb_id(label: str) -> tuple[str, list[str]]:
    """Split a PSCDB FreeID/BoundID into PDB code and chain list.

    Args:
        label: e.g. ``2gg4_A`` or ``5csc_AB``.

    Returns:
        Lowercase PDB code and ordered chain ids.
    """
    pdb, chains = label.split("_", 1)
    return pdb.lower(), list(chains)


def is_single_chain(label: str) -> bool:
    """Return True if the PSCDB id names exactly one chain.

    Args:
        label: FreeID or BoundID.

    Returns:
        Whether exactly one chain is listed.
    """
    _pdb, chains = parse_pscdb_id(label)
    return len(chains) == 1


def system_id(pscid: str, free_id: str) -> str:
    """Stable filesystem-safe system stem.

    Args:
        pscid: e.g. ``CL.57``.
        free_id: e.g. ``1a48_A``.

    Returns:
        Stem like ``CL57_1a48_A``.
    """
    return f"{pscid.replace('.', '')}_{free_id}"


def structure_path(structures_dir: Path, pdb_code: str) -> Path | None:
    """Resolve a downloaded mmCIF or PDB path for a PDB code.

    Args:
        structures_dir: Directory of downloaded files.
        pdb_code: Four-letter PDB code.

    Returns:
        Path if present, else None.
    """
    code = pdb_code.lower()
    for suffix in (".cif", ".mmcif", ".pdb"):
        path = structures_dir / f"{code}{suffix}"
        if path.exists():
            return path
    return None


def load_ca_chain(path: Path, chain_id: str) -> tuple[FloatArr, str]:
    """Load CA coordinates and one-letter sequence for one chain.

    Args:
        path: mmCIF or PDB file.
        chain_id: Chain identifier.

    Returns:
        CA array ``(N, 3)`` and sequence.

    Raises:
        ValueError: Chain missing or no CA atoms.
    """
    if path.suffix.lower() in {".cif", ".mmcif"}:
        parser: PDBParser | MMCIFParser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    model = next(iter(structure))
    if chain_id not in model:
        # Some mmCIFs use author vs asym ids; try case variants.
        candidates = [c.id for c in model]
        raise ValueError(f"chain {chain_id!r} not in {path.name}; have {candidates}")
    chain = model[chain_id]
    coords: list[list[float]] = []
    seq: list[str] = []
    for residue in chain.get_residues():
        hetflag, _resseq, _icode = residue.id
        if hetflag != " ":
            continue
        if "CA" not in residue:
            continue
        name = residue.get_resname().upper()
        if name not in AA3TO1:
            continue
        coords.append(residue["CA"].get_coord().tolist())
        seq.append(AA3TO1[name])
    if not coords:
        raise ValueError(f"no CA atoms for chain {chain_id} in {path}")
    return np.asarray(coords, dtype=np.float64), "".join(seq)
