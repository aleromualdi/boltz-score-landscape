"""Curate the full usable PSCDB CL/IL/CD/ID panel and download structures."""

from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

from pscdb_io import (
    is_single_chain,
    load_ca_chain,
    parse_pscdb_id,
    structure_path,
    system_id,
)
from pscdb_segments import collect_moving_ranges, load_pscdb_rows

EXP_ROOT = Path(__file__).resolve().parents[1]
RAW_LIST = EXP_ROOT / "data" / "raw" / "pscdb_pscid_list.csv"
PSCDB_CSV = EXP_ROOT / "data" / "raw" / "pscdb.csv"
STRUCT_DIR = EXP_ROOT / "data" / "structures"
OUT_PANEL = EXP_ROOT / "inputs" / "panel_full.json"
OUT_CSV = EXP_ROOT / "inputs" / "panel_full.csv"

RCSB_CIF = "https://files.rcsb.org/download/{code}.cif"
CLASSES = ("CL", "IL", "CD", "ID")


def download_pdb(pdb_code: str) -> Path:
    """Download an mmCIF from RCSB if missing.

    Args:
        pdb_code: Four-letter PDB code.

    Returns:
        Local mmCIF path.
    """
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    existing = structure_path(STRUCT_DIR, pdb_code)
    if existing is not None:
        return existing
    dest = STRUCT_DIR / f"{pdb_code.lower()}.cif"
    url = RCSB_CIF.format(code=pdb_code.upper())
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310
    except Exception as exc:  # noqa: BLE001
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"download failed {pdb_code}: {exc}") from exc
    if dest.stat().st_size < 100:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"empty download {pdb_code}")
    return dest


def try_measure(free_id: str, bound_id: str) -> dict[str, object] | None:
    """Download and measure chain lengths for a candidate pair.

    Args:
        free_id: PSCDB FreeID.
        bound_id: PSCDB BoundID.

    Returns:
        Geometry metadata, or None if unusable.
    """
    if not (is_single_chain(free_id) and is_single_chain(bound_id)):
        return None
    free_pdb, free_chains = parse_pscdb_id(free_id)
    bound_pdb, bound_chains = parse_pscdb_id(bound_id)
    free_chain = free_chains[0]
    bound_chain = bound_chains[0]
    try:
        free_path = download_pdb(free_pdb)
        bound_path = download_pdb(bound_pdb)
        free_ca, free_seq = load_ca_chain(free_path, free_chain)
        bound_ca, bound_seq = load_ca_chain(bound_path, bound_chain)
    except Exception as exc:  # noqa: BLE001
        print(f"  skip {free_id}/{bound_id}: {exc}", flush=True)
        return None
    n_free = len(free_seq)
    n_bound = len(bound_seq)
    if n_free < 40 or n_bound < 40:
        return None
    return {
        "free_pdb": free_pdb,
        "bound_pdb": bound_pdb,
        "free_chain": free_chain,
        "bound_chain": bound_chain,
        "free_path": str(free_path),
        "bound_path": str(bound_path),
        "n_res_free": n_free,
        "n_res_bound": n_bound,
        "n_res": n_free,
        "sequence": free_seq,
        "bound_sequence": bound_seq,
        "n_ca_free": int(free_ca.shape[0]),
        "n_ca_bound": int(bound_ca.shape[0]),
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--min_len", type=int, default=40)
    parser.add_argument("--seed", type=int, default=204)
    parser.add_argument(
        "--classes",
        default=",".join(CLASSES),
        help="Comma-separated classes.",
    )
    args = parser.parse_args()
    classes = tuple(c.strip().upper() for c in args.classes.split(",") if c.strip())

    list_rows = list(csv.DictReader(RAW_LIST.open()))
    by_pscid: dict[str, dict[str, str]] = {}
    for row in list_rows:
        by_pscid[row["PSCID"]] = row
    pscdb = load_pscdb_rows(PSCDB_CSV)

    candidates = [
        row
        for pscid, row in by_pscid.items()
        if pscid.split(".", 1)[0] in classes
        and is_single_chain(row["FreeID"])
        and is_single_chain(row["BoundID"])
        and len(collect_moving_ranges(pscdb.get(pscid, []))) > 0
    ]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    print(
        f"candidates with single-chain + moving segment: {len(candidates)} "
        f"(classes={classes})",
        flush=True,
    )

    selected: list[dict[str, object]] = []
    skipped = Counter()
    for idx, row in enumerate(candidates, start=1):
        pscid = row["PSCID"]
        cls = pscid.split(".", 1)[0]
        print(f"[{idx}/{len(candidates)}] {pscid} {row['FreeID']}", flush=True)
        geom = try_measure(row["FreeID"], row["BoundID"])
        if geom is None:
            skipped["measure_fail"] += 1
            continue
        n_res = int(geom["n_res"])
        if n_res < args.min_len or n_res > args.max_len:
            skipped["length"] += 1
            continue
        record = {
            "system": system_id(pscid, row["FreeID"]),
            "pscid": pscid,
            "class": cls,
            "protein_name": row.get("Protein name", ""),
            "free_id": row["FreeID"],
            "bound_id": row["BoundID"],
            "ligands": row.get("Ligands", ""),
            "n_moving_ranges": len(collect_moving_ranges(pscdb.get(pscid, []))),
            **geom,
        }
        selected.append(record)
        skipped[f"ok_{cls}"] += 1

    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_PANEL.write_text(json.dumps(selected, indent=2))
    with OUT_CSV.open("w", newline="") as handle:
        fields = [
            "system",
            "pscid",
            "class",
            "protein_name",
            "free_id",
            "bound_id",
            "n_res",
            "n_moving_ranges",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    counts = Counter(str(r["class"]) for r in selected)
    print(f"selected n={len(selected)} by class={dict(counts)}", flush=True)
    print(f"skip tallies={dict(skipped)}", flush=True)
    print(f"wrote {OUT_PANEL}", flush=True)
    print(f"wrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
