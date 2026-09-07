"""Write Boltz YAMLs and the soft91 panel JSON from the pair table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from geom import load_ca_sequence, parse_label
from pairs import EXP_ROOT, PairRow, load_pairs, pdb_path


def sequence_of(label: str) -> tuple[str, int]:
    """Canonical sequence from a benchmark PDB.

    Args:
        label: Apo or holo table id.

    Returns:
        Sequence and length.
    """
    _pdb, model, _chain = parse_label(label)
    _coords, seq = load_ca_sequence(pdb_path(label), model)
    return seq, len(seq)


def write_yaml(path: Path, sequence: str) -> None:
    """Write a single-chain Boltz protein YAML.

    Args:
        path: Destination.
        sequence: One-letter sequence.
    """
    path.write_text(
        f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {sequence}\n"
    )


def records_for(pairs: list[PairRow]) -> list[dict[str, object]]:
    """Build JSON records, writing YAMLs from the apo sequence.

    Args:
        pairs: Table rows.

    Returns:
        Serializable panel records.
    """
    yaml_dir = EXP_ROOT / "inputs" / "yamls"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for row in pairs:
        seq, n_res = sequence_of(row["apo_id"])
        yaml_path = yaml_dir / f"{row['apo_id']}.yaml"
        write_yaml(yaml_path, seq)
        records.append(
            {
                **row,
                "n_res": n_res,
                "sequence_len": n_res,
                "yaml": str(yaml_path),
                "apo_pdb": str(pdb_path(row["apo_id"])),
                "holo_pdb": str(pdb_path(row["holo_id"])),
            }
        )
        print(
            f"{row['apo_id']}  n={n_res}  rmsd={row['rmsd_apo_holo']:.2f}  {row['motion_type']}",
            flush=True,
        )
    return records


def main() -> None:
    """CLI: write ``inputs/panel91.json`` (and YAMLs)."""
    pairs = load_pairs()
    records = records_for(pairs)
    out_path = EXP_ROOT / "inputs" / "panel91.json"
    out_path.write_text(json.dumps(records, indent=2))
    print(f"wrote {out_path}  n={len(records)}", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
