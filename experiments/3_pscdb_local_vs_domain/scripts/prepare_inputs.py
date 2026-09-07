"""Write Boltz single-chain YAMLs for the full PSCDB panel."""

from __future__ import annotations

import json
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
PANEL = EXP_ROOT / "inputs" / "panel_full.json"
YAML_DIR = EXP_ROOT / "inputs" / "yamls"


def write_yaml(path: Path, sequence: str) -> None:
    """Write a single-protein Boltz YAML.

    Args:
        path: Destination path.
        sequence: One-letter amino-acid sequence.
    """
    path.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {sequence}\n"
    )


def main() -> None:
    """CLI entrypoint."""
    records = json.loads(PANEL.read_text())
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    for row in records:
        system = str(row["system"])
        path = YAML_DIR / f"{system}.yaml"
        write_yaml(path, str(row["sequence"]))
        row["yaml"] = str(path)
    PANEL.write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} yamls under {YAML_DIR}", flush=True)


if __name__ == "__main__":
    main()
