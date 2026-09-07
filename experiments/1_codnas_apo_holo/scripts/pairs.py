"""Saldaño 2022 CoDNaS apo–holo pair table (91 pairs)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypedDict

EXP_ROOT = Path(__file__).resolve().parents[1]
PAIR_TABLE = EXP_ROOT / "data" / "tables" / "revision1_86_plus_5.csv"
PAIR_PDB_DIR = EXP_ROOT / "data" / "pairs"


class PairRow(TypedDict):
    """One curated apo–holo pair.

    Attributes:
        apo_id: Table / PDB-file stem for the apo chain.
        holo_id: Table / PDB-file stem for the holo chain.
        rmsd_apo_holo: Published CA RMSD (Å).
        length: Residue count.
        motion_type: DynDom class (``domain movements`` or ``loop movements``).
        n_domains: DynDom domain count.
    """

    apo_id: str
    holo_id: str
    rmsd_apo_holo: float
    length: int
    motion_type: str
    n_domains: int


def load_pairs() -> list[PairRow]:
    """Load the 91-pair journal table.

    Returns:
        One row per protein.

    Raises:
        FileNotFoundError: Missing CSV.
    """
    if not PAIR_TABLE.exists():
        msg = f"missing {PAIR_TABLE}; copy the GitLab benchmark into data/"
        raise FileNotFoundError(msg)
    rows: list[PairRow] = []
    with PAIR_TABLE.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for raw in reader:
            rows.append(
                {
                    "apo_id": raw["apo_id"],
                    "holo_id": raw["holo_id"],
                    "rmsd_apo_holo": float(raw["rmsd_apo_holo"]),
                    "length": int(raw["length"]),
                    "motion_type": raw["motion_type"],
                    "n_domains": int(raw["domains_quant"]),
                }
            )
    return rows


def pair_by_apo() -> dict[str, PairRow]:
    """Index pairs by apo id.

    Returns:
        Apo id to row.
    """
    return {row["apo_id"]: row for row in load_pairs()}


def pdb_path(label: str) -> Path:
    """Path to a chain-split benchmark PDB.

    Args:
        label: Apo or holo id from the table.

    Returns:
        ``data/pairs/{label}.pdb``.
    """
    return PAIR_PDB_DIR / f"{label}.pdb"
