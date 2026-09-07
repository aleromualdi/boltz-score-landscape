"""Blind retrieval for one system vs PSCDB Moving segment (frozen recipe)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

EXP_ROOT = Path(__file__).resolve().parents[1]
EXP100 = EXP_ROOT.parents[0] / "1_codnas_apo_holo" / "scripts"

sys.path.insert(0, str(EXP100))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geom import AA3TO1, matched_residue_indices
from soft_anm import anm_participation
from pscdb_io import structure_path
from pscdb_segments import (
    collect_moving_ranges,
    load_ca_resseq,
    load_pscdb_rows,
    mobile_mask_from_resseq,
    transfer_mask_by_alignment,
)
from retrieval_metrics import (
    auprc,
    auroc,
    select_mode_indices,
    softness_score,
    top_m_precision,
)

type BoolArr = NDArray[np.bool_]

PANEL = EXP_ROOT / "inputs" / "panel_full.json"
PSCDB_CSV = EXP_ROOT / "data" / "raw" / "pscdb.csv"
STRUCT_DIR = EXP_ROOT / "data" / "structures"
MODE_DIR = EXP_ROOT / "outputs" / "modes"
OUT_DIR = EXP_ROOT / "outputs" / "per_system"


def boltz_sequence(processed_dir: Path, system: str) -> str:
    """Boltz construct sequence.

    Args:
        processed_dir: Processed folder.
        system: System stem.

    Returns:
        Sequence.
    """
    npz = np.load(processed_dir / "structures" / f"{system}.npz")
    return "".join(AA3TO1.get(str(name).strip().upper(), "X") for name in npz["residues"]["name"])


def find_processed(predict_root: Path, system: str) -> Path:
    """Find processed dir.

    Args:
        predict_root: Predict root.
        system: System stem.

    Returns:
        Processed path.
    """
    for proc in sorted(predict_root.glob("boltz_results_*/processed")):
        if (proc / "structures" / f"{system}.npz").exists():
            return proc
    raise FileNotFoundError(system)


def build_pscdb_mobile_mask(
    record: dict[str, Any],
    pscid_rows: list[dict[str, str]],
    boltz_seq: str,
    n_mode_ca: int,
) -> tuple[BoolArr, dict[str, Any]]:
    """Map PSCDB Moving segments onto mode CA indices.

    Args:
        record: Panel record.
        pscid_rows: PSCDB rows.
        boltz_seq: Boltz sequence.
        n_mode_ca: Mode CA count.

    Returns:
        ``(mask, diagnostics)``.
    """
    ranges = collect_moving_ranges(pscid_rows)
    free_path = Path(str(record["free_path"]))
    bound_path = Path(str(record["bound_path"]))
    if not free_path.exists():
        alt = structure_path(STRUCT_DIR, str(record["free_pdb"]))
        if alt is None:
            raise FileNotFoundError(free_path)
        free_path = alt
    if not bound_path.exists():
        alt = structure_path(STRUCT_DIR, str(record["bound_pdb"]))
        if alt is None:
            raise FileNotFoundError(bound_path)
        bound_path = alt

    free_chain = str(record["free_chain"])
    bound_chain = str(record["bound_chain"])
    _fc, free_seq, free_resseq = load_ca_resseq(free_path, free_chain)
    _bc, bound_seq, bound_resseq = load_ca_resseq(bound_path, bound_chain)

    free_mask = mobile_mask_from_resseq(free_resseq, ranges, free_chain)
    source = "free_resseq"
    if int(free_mask.sum()) == 0:
        bound_mask = mobile_mask_from_resseq(bound_resseq, ranges, bound_chain)
        if int(bound_mask.sum()) == 0:
            letters = {c for _a, c, _b, _d in ranges} | {d for _a, _c, _b, d in ranges}
            for chain in letters:
                bound_mask |= mobile_mask_from_resseq(bound_resseq, ranges, chain)
        src, dst = matched_residue_indices(bound_seq, free_seq)
        free_mask = transfer_mask_by_alignment(
            bound_mask,
            bound_seq,
            free_seq,
            np.asarray(src, dtype=np.int64),
            np.asarray(dst, dtype=np.int64),
            n_dst=len(free_seq),
        )
        source = "bound_resseq→free_align"

    mask = np.zeros(n_mode_ca, dtype=bool)
    n_mapped = 0
    if int(free_mask.sum()) > 0:
        boltz_idx, free_idx = matched_residue_indices(boltz_seq, free_seq)
        for i_b, i_f in zip(boltz_idx, free_idx, strict=True):
            if i_b < n_mode_ca and 0 <= i_f < len(free_mask) and bool(free_mask[i_f]):
                mask[i_b] = True
                n_mapped += 1
    if int(mask.sum()) == 0:
        bound_mask = mobile_mask_from_resseq(bound_resseq, ranges, bound_chain)
        if int(bound_mask.sum()) == 0:
            letters = {c for _a, c, _b, _d in ranges} | {d for _a, _c, _b, d in ranges}
            for chain in letters:
                bound_mask |= mobile_mask_from_resseq(bound_resseq, ranges, chain)
        boltz_idx, bound_idx = matched_residue_indices(boltz_seq, bound_seq)
        for i_b, i_bd in zip(boltz_idx, bound_idx, strict=True):
            if i_b < n_mode_ca and 0 <= i_bd < len(bound_mask) and bool(bound_mask[i_bd]):
                mask[i_b] = True
                n_mapped += 1
        source = "bound_resseq→boltz_align"

    diag = {
        "n_ranges": len(ranges),
        "ranges": [f"{a}{ca}-{b}{cb}" for a, ca, b, cb in ranges],
        "label_source": source,
        "n_mobile_boltz": int(mask.sum()),
        "frac_mobile": float(mask.mean()) if len(mask) else float("nan"),
        "n_mapped": n_mapped,
    }
    return mask, diag


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--predict_root", type=Path, required=True)
    parser.add_argument("--k_soft", type=int, default=10)
    parser.add_argument("--anm_cutoff", type=float, default=15.0)
    parser.add_argument("--anm_gamma", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"{args.system}.json"
    if out_json.exists() and not args.force:
        print(f"skip eval {args.system}", flush=True)
        return

    record = next(r for r in json.loads(PANEL.read_text()) if r["system"] == args.system)
    mode_npz = MODE_DIR / f"{args.system}_ca_modes.npz"
    if not mode_npz.exists():
        raise FileNotFoundError(mode_npz)
    pscdb = load_pscdb_rows(PSCDB_CSV)
    packed = np.load(mode_npz)
    modes = np.asarray(packed["modes"], dtype=np.float64)
    evals = np.asarray(packed["evals"], dtype=np.float64)
    pred_ca = np.asarray(packed["pred_ca"], dtype=np.float64)
    processed = find_processed(args.predict_root, args.system)
    boltz_seq = boltz_sequence(processed, args.system)
    if "boltz_seq" in packed.files:
        value = packed["boltz_seq"]
        boltz_seq = str(value.item()) if value.shape == () else str(value)

    labels, diag = build_pscdb_mobile_mask(
        record, pscdb[str(record["pscid"])], boltz_seq, n_mode_ca=modes.shape[1]
    )
    result: dict[str, Any] = {
        "system": args.system,
        "class": record["class"],
        "pscid": record["pscid"],
        "ok": False,
        **diag,
    }
    if int(labels.sum()) == 0:
        out_json.write_text(json.dumps(result, indent=2))
        print(f"{args.system}: zero mobile labels", flush=True)
        return

    n = min(modes.shape[1], len(labels), len(pred_ca))
    modes = modes[:, :n]
    pred_ca = pred_ca[:n]
    labels = labels[:n]
    idx = select_mode_indices(
        evals, np.ones(len(evals)), "top10_lambda", pr_cut=0.15, n_cand=50
    )
    s_boltz = softness_score(modes, evals, idx, inv_lambda=False)
    s_anm = anm_participation(
        pred_ca, k_soft=args.k_soft, cutoff=args.anm_cutoff, gamma=args.anm_gamma
    )
    m_pos = max(1, int(labels.sum()))
    result.update(
        {
            "ok": True,
            "n_ca": n,
            "n_mobile": int(labels.sum()),
            "frac_mobile": float(labels.mean()),
            "auroc_boltz": auroc(s_boltz, labels),
            "auroc_anm": auroc(s_anm, labels),
            "auprc_boltz": auprc(s_boltz, labels),
            "auprc_anm": auprc(s_anm, labels),
            "topm_boltz": top_m_precision(s_boltz, labels, m=m_pos),
            "topm_anm": top_m_precision(s_anm, labels, m=m_pos),
        }
    )
    result["delta_auroc"] = float(result["auroc_boltz"]) - float(result["auroc_anm"])
    result["delta_auprc"] = float(result["auprc_boltz"]) - float(result["auprc_anm"])
    result["delta_topm"] = float(result["topm_boltz"]) - float(result["topm_anm"])
    result["boltz_wins_auroc"] = bool(result["delta_auroc"] > 0)
    if isinstance(result.get("ranges"), list):
        result["ranges"] = ";".join(result["ranges"])
    out_json.write_text(json.dumps(result, indent=2))
    print(
        f"{args.system} {record['class']}: mobile={result['n_mobile']}/{n} "
        f"ΔAUROC={result['delta_auroc']:+.3f} ΔAUPRC={result['delta_auprc']:+.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
