"""Aggregate soft-subspace enrichment JSON outputs into one CSV."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """CLI: summarize ``*_soft_enrichment.json`` in a directory."""
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP_ROOT / "outputs" / "analysis_soft91"
    rows: list[dict[str, float | str | bool]] = []
    for path in sorted(out_dir.glob("*_soft_enrichment.json")):
        rows.append(json.loads(path.read_text()))
    if not rows:
        print(f"no soft enrichment results in {out_dir}", flush=True)
        return
    csv_path = out_dir / "soft_panel.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}  n={len(rows)}", flush=True)
    for row in rows:
        print(
            f"  {row['system']}  Δ={float(row['rmsd_table']):.2f}Å  "
            f"nearest={row['pred_nearest_state']}  "
            f"C10={float(row['C10_exp']):.3f}  enriched={float(row['C10_enriched_fraction']):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
    sys.exit(0)
