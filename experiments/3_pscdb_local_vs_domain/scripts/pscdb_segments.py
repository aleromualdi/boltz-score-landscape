"""Parse PSCDB Moving segment annotations and map them onto residue indices."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from numpy.typing import NDArray

from pscdb_io import AA3TO1

type FloatArr = NDArray[np.float64]
type BoolArr = NDArray[np.bool_]
type IntArr = NDArray[np.int64]

_RANGE = re.compile(r"(\d+)([A-Za-z])-(\d+)([A-Za-z])")
_LABEL = re.compile(r"[A-Za-z0-9]+\(([^)]*)\)")


def load_pscdb_rows(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    """Index full PSCDB rows by PSCID.

    Args:
        csv_path: Path to ``pscdb.csv``.

    Returns:
        Map PSCID → component rows.
    """
    by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            by_id[row["PSCID"]].append(row)
    return by_id


def parse_moving_ranges(moving_segment: str) -> list[tuple[int, str, int, str]]:
    """Parse residue ranges from a Moving segment string.

    Args:
        moving_segment: e.g. ``L2(56A-58A,64A-70A)`` or ``D2(33A-87A)``.

    Returns:
        List of ``(start, start_chain, end, end_chain)``.
    """
    text = moving_segment.strip()
    if not text:
        return []
    bodies = _LABEL.findall(text)
    chunks = bodies if bodies else [text]
    ranges: list[tuple[int, str, int, str]] = []
    for chunk in chunks:
        for start, c_a, end, c_b in _RANGE.findall(chunk):
            ranges.append((int(start), c_a, int(end), c_b))
    return ranges


def load_ca_resseq(path: Path, chain_id: str) -> tuple[FloatArr, str, IntArr]:
    """Load CA coords, sequence, and author residue numbers for one chain.

    Args:
        path: mmCIF or PDB.
        chain_id: Chain id.

    Returns:
        ``(ca, sequence, resseq)``.
    """
    if path.suffix.lower() in {".cif", ".mmcif"}:
        parser: PDBParser | MMCIFParser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    model = next(iter(structure))
    if chain_id not in model:
        candidates = [c.id for c in model]
        raise ValueError(f"chain {chain_id!r} not in {path.name}; have {candidates}")
    chain = model[chain_id]
    coords: list[list[float]] = []
    seq: list[str] = []
    resseqs: list[int] = []
    for residue in chain.get_residues():
        hetflag, resseq, _icode = residue.id
        if hetflag != " ":
            continue
        if "CA" not in residue:
            continue
        name = residue.get_resname().upper()
        if name not in AA3TO1:
            continue
        coords.append(residue["CA"].get_coord().tolist())
        seq.append(AA3TO1[name])
        resseqs.append(int(resseq))
    if not coords:
        raise ValueError(f"no CA atoms for chain {chain_id} in {path}")
    return (
        np.asarray(coords, dtype=np.float64),
        "".join(seq),
        np.asarray(resseqs, dtype=np.int64),
    )


def mobile_mask_from_resseq(
    resseqs: IntArr, ranges: list[tuple[int, str, int, str]], chain: str
) -> BoolArr:
    """Label residues whose author resseq falls in PSCDB ranges on ``chain``.

    Args:
        resseqs: Per-CA residue numbers.
        ranges: Parsed moving ranges.
        chain: Chain letter to keep.

    Returns:
        Boolean mobile mask.
    """
    mask = np.zeros(len(resseqs), dtype=bool)
    for start, c_a, end, c_b in ranges:
        if c_a != chain or c_b != chain:
            continue
        lo, hi = (start, end) if start <= end else (end, start)
        mask |= (resseqs >= lo) & (resseqs <= hi)
    return mask


def collect_moving_ranges(
    pscid_rows: list[dict[str, str]],
    *,
    motion_types: frozenset[str] | None = None,
) -> list[tuple[int, str, int, str]]:
    """Union Moving segment ranges across PSCDB components for one PSCID.

    Args:
        pscid_rows: All CSV rows for one PSCID.
        motion_types: Allowed ``Type of motion`` values (lowercased).
            ``None`` keeps Local and Domain.

    Returns:
        Deduplicated residue ranges.
    """
    allowed = motion_types if motion_types is not None else frozenset({"local", "domain"})
    seen: set[tuple[int, str, int, str]] = set()
    out: list[tuple[int, str, int, str]] = []
    for row in pscid_rows:
        motion = row.get("Type of motion", "").strip().lower()
        if motion and motion not in allowed:
            continue
        for item in parse_moving_ranges(row.get("Moving segment", "")):
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def transfer_mask_by_alignment(
    src_mask: BoolArr,
    src_seq: str,
    dst_seq: str,
    matched_src: IntArr,
    matched_dst: IntArr,
    n_dst: int,
) -> BoolArr:
    """Copy mobile labels from src to dst along a sequence alignment.

    Args:
        src_mask: Labels on the source chain.
        src_seq: Unused; kept for API clarity.
        dst_seq: Unused.
        matched_src: Aligned source indices.
        matched_dst: Aligned destination indices.
        n_dst: Destination length.

    Returns:
        Destination mobile mask.
    """
    del src_seq, dst_seq
    out = np.zeros(n_dst, dtype=bool)
    for i_src, i_dst in zip(matched_src.tolist(), matched_dst.tolist(), strict=True):
        if 0 <= i_src < len(src_mask) and 0 <= i_dst < n_dst and bool(src_mask[i_src]):
            out[i_dst] = True
    return out
