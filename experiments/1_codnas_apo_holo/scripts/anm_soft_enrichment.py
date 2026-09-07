"""ANM soft-subspace enrichment for CoDNaS apo↔holo displacements.

Baseline counterpart to Boltz ``directional_curvature.py``: build a Cα ANM on
the predicted structure, measure how much of experimental ``Δx`` sits in the
top-``k`` soft modes (``C_k``), and compare to random-direction nulls.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import MMCIFParser
from numpy.typing import NDArray

from geom import (
    AA3TO1,
    apo_holo_delta_aligned,
    cumulative_subspace_overlap,
    kabsch_rotation,
    load_ca_sequence,
    matched_residue_indices,
    parse_label,
)
from pairs import EXP_ROOT, load_pairs, pair_by_apo, pdb_path

type FloatArr = NDArray[np.float64]


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


def load_pred_ca_sequence(cif_path: Path) -> tuple[FloatArr, str]:
    """Load CA coordinates and one-letter sequence from a Boltz CIF.

    Args:
        cif_path: Predicted ``*_model_0.cif``.

    Returns:
        CA array ``(N, 3)`` and sequence.

    Raises:
        ValueError: Too few CA atoms.
    """
    structure = MMCIFParser(QUIET=True).get_structure(cif_path.stem, str(cif_path))  # type: ignore[no-untyped-call]
    model = next(structure.get_models())
    coords: list[list[float]] = []
    seq: list[str] = []
    for residue in model.get_residues():
        hetflag, _resseq, _icode = residue.id
        if hetflag != " ":
            continue
        resname = residue.get_resname().upper()
        if resname not in AA3TO1 or "CA" not in residue:
            continue
        coords.append([float(x) for x in residue["CA"].coord])
        seq.append(AA3TO1[resname])
    if len(coords) < 10:
        msg = f"too few CA atoms in {cif_path}"
        raise ValueError(msg)
    return np.asarray(coords, dtype=np.float64), "".join(seq)


def anm_hessian(ca: FloatArr, cutoff: float = 15.0, gamma: float = 1.0) -> FloatArr:
    """Build a uniform-spring Cα ANM Hessian.

    Args:
        ca: CA coordinates ``(N, 3)``.
        cutoff: Contact cutoff in Å.
        gamma: Spring constant.

    Returns:
        Hessian ``(3N, 3N)``.
    """
    n_ca = ca.shape[0]
    hess = np.zeros((3 * n_ca, 3 * n_ca), dtype=np.float64)
    for i_site in range(n_ca):
        for j_site in range(i_site + 1, n_ca):
            diff = ca[j_site] - ca[i_site]
            dist = float(np.linalg.norm(diff))
            if dist <= 1e-8 or dist > cutoff:
                continue
            unit = diff / dist
            block = -gamma * np.outer(unit, unit)
            i_slice = slice(3 * i_site, 3 * i_site + 3)
            j_slice = slice(3 * j_site, 3 * j_site + 3)
            hess[i_slice, j_slice] += block
            hess[j_slice, i_slice] += block
            hess[i_slice, i_slice] -= block
            hess[j_slice, j_slice] -= block
    return hess


def anm_soft_modes(
    ca: FloatArr,
    n_modes: int = 15,
    cutoff: float = 15.0,
    gamma: float = 1.0,
) -> FloatArr:
    """Return the softest non-rigid ANM eigenvectors as CA modes.

    Args:
        ca: CA coordinates ``(N, 3)``.
        n_modes: Number of soft modes after discarding rigid zeros.
        cutoff: ANM cutoff in Å.
        gamma: Spring constant.

    Returns:
        Modes ``(K, N, 3)``, softest first.

    Raises:
        ValueError: Fewer than four CAs or no positive eigenvalues.
    """
    n_ca = ca.shape[0]
    if n_ca < 4:
        msg = f"need ≥4 CA for ANM, got {n_ca}"
        raise ValueError(msg)
    hess = anm_hessian(ca, cutoff=cutoff, gamma=gamma)
    evals, evecs = np.linalg.eigh(hess)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    keep = np.where(evals > 1e-8)[0]
    if len(keep) == 0:
        msg = "ANM has no positive eigenvalues"
        raise ValueError(msg)
    take = keep[:n_modes]
    return evecs[:, take].T.reshape(len(take), n_ca, 3)


def remove_rigid(delta: FloatArr) -> FloatArr:
    """Remove net translation from a CA displacement.

    Args:
        delta: Displacement ``(N, 3)``.

    Returns:
        COM-centered displacement of the same shape.
    """
    return delta - delta.mean(axis=0, keepdims=True)


def null_c_k(
    modes: FloatArr,
    n_samples: int,
    k: int,
    rng: np.random.Generator,
) -> FloatArr:
    """Null ``C_k`` for random CA directions on the mode support.

    Args:
        modes: Soft modes ``(K, N, 3)``.
        n_samples: Number of random draws.
        k: Subspace dimension.
        rng: NumPy RNG.

    Returns:
        Null ``C_k`` samples.
    """
    n_ca = modes.shape[1]
    values = np.empty(n_samples, dtype=np.float64)
    n_keep = min(k, len(modes))
    for i in range(n_samples):
        noise = rng.normal(size=(n_ca, 3))
        noise = remove_rigid(noise)
        values[i] = cumulative_subspace_overlap(modes, noise, k=n_keep)
    return values


def experimental_delta_on_pred(
    system: str,
    pred_ca: FloatArr,
    pred_seq: str,
) -> tuple[FloatArr, FloatArr, float]:
    """Map apo→holo ``Δx`` onto predicted CA sites in the pose frame.

    Args:
        system: Apo table id.
        pred_ca: Predicted CA ``(N, 3)``.
        pred_seq: Predicted sequence.

    Returns:
        Matched pred CA, ``Δx`` in pose frame, recomputed RMSD.
    """
    spec = pair_by_apo()[system]
    _pdb, apo_model, _ = parse_label(spec["apo_id"])
    _pdb, holo_model, _ = parse_label(spec["holo_id"])
    apo_ca, apo_seq = load_ca_sequence(pdb_path(spec["apo_id"]), apo_model)
    holo_ca, holo_seq = load_ca_sequence(pdb_path(spec["holo_id"]), holo_model)
    apo_m, delta_aligned, rmsd, apo_match_idx, _ = apo_holo_delta_aligned(
        apo_ca, apo_seq, holo_ca, holo_seq
    )
    pred_idx, apo_idx = matched_residue_indices(pred_seq, apo_seq)
    apo_to_delta = {int(i): k for k, i in enumerate(apo_match_idx)}
    keep_pred = [
        int(p_i)
        for p_i, a_i in zip(pred_idx, apo_idx, strict=True)
        if int(a_i) in apo_to_delta
    ]
    keep_delta = [apo_to_delta[int(a_i)] for a_i in apo_idx if int(a_i) in apo_to_delta]
    pred_m = pred_ca[keep_pred]
    delta_apo = delta_aligned[keep_delta]
    # Rotate apo-frame vectors into the predicted pose frame.
    rot, _cm, _ct = kabsch_rotation(pred_m, apo_m[keep_delta])
    delta_pose = delta_apo @ rot.T
    return pred_m, delta_pose, float(rmsd)


def evaluate_system(
    system: str,
    predict_root: Path,
    *,
    n_null: int = 500,
    n_soft_modes: int = 15,
    k: int = 10,
    cutoff: float = 15.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute ANM ``C_k`` enrichment for one apo↔holo pair.

    Args:
        system: Apo table id.
        predict_root: Directory containing ``{system}/boltz_results_*/``.
        n_null: Random null samples.
        n_soft_modes: Soft ANM modes to keep.
        k: Subspace size for ``C_k``.
        cutoff: ANM cutoff in Å.
        seed: RNG seed.

    Returns:
        JSON-serializable result row.
    """
    spec = pair_by_apo()[system]
    system_root = predict_root / system
    cif = find_predicted_cif(system_root, system)
    pred_ca, pred_seq = load_pred_ca_sequence(cif)
    pred_m, delta, rmsd = experimental_delta_on_pred(system, pred_ca, pred_seq)
    modes = anm_soft_modes(pred_m, n_modes=n_soft_modes, cutoff=cutoff)
    c10_exp = cumulative_subspace_overlap(modes, delta, k=min(k, len(modes)))
    c5_exp = cumulative_subspace_overlap(modes, delta, k=min(5, len(modes)))
    rng = np.random.default_rng(seed)
    c10_null = null_c_k(modes, n_null, k=k, rng=rng)
    c5_null = null_c_k(modes, n_null, k=5, rng=rng)
    return {
        "system": system,
        "holo_id": spec["holo_id"],
        "motion_type": spec["motion_type"],
        "rmsd_table": float(spec["rmsd_apo_holo"]),
        "rmsd_recomputed": rmsd,
        "n_ca": float(pred_m.shape[0]),
        "C5_exp": float(c5_exp),
        "C10_exp": float(c10_exp),
        "C5_null_mean": float(np.mean(c5_null)),
        "C10_null_mean": float(np.mean(c10_null)),
        "C5_enriched_fraction": float(np.mean(c5_exp > c5_null)),
        "C10_enriched_fraction": float(np.mean(c10_exp > c10_null)),
        "n_null": float(n_null),
        "n_soft_modes": float(len(modes)),
        "anm_cutoff": float(cutoff),
        "cif": str(cif),
        "method": "ANM",
    }


def summarize(rows: list[dict[str, Any]]) -> None:
    """Print aggregate enrichment stats to stdout.

    Args:
        rows: Per-system result dictionaries.
    """
    for label, subset in [
        ("all", rows),
        ("domain", [r for r in rows if "domain" in str(r["motion_type"])]),
        ("loop", [r for r in rows if "loop" in str(r["motion_type"])]),
    ]:
        if not subset:
            continue
        c10 = np.asarray([float(r["C10_exp"]) for r in subset], dtype=np.float64)
        c10n = np.asarray([float(r["C10_null_mean"]) for r in subset], dtype=np.float64)
        n_enr = int(np.sum(c10 > c10n))
        fold = float(np.median(c10 / np.maximum(c10n, 1e-12)))
        print(
            f"ANM {label:6s} n={len(subset):2d}  "
            f"C10_enr={n_enr}/{len(subset)}  "
            f"med C10={float(np.median(c10)):.3f}/{float(np.median(c10n)):.3f}  "
            f"fold={fold:.1f}x",
            flush=True,
        )


def main() -> None:
    """CLI: run ANM soft enrichment over the soft91 panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predict_root",
        type=Path,
        default=EXP_ROOT / "outputs" / "predict_soft91",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=EXP_ROOT / "outputs" / "analysis_soft91",
    )
    parser.add_argument("--n_null", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in load_pairs():
        system = spec["apo_id"]
        json_path = args.out_dir / f"{system}_anm_soft_enrichment.json"
        if json_path.exists() and not args.force:
            rows.append(json.loads(json_path.read_text()))
            print(f"skip {system}: exists", flush=True)
            continue
        print(f"ANM enrich {system}", flush=True)
        row = evaluate_system(system, args.predict_root, n_null=args.n_null)
        json_path.write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(
            f"  C10={row['C10_exp']:.3f}  null={row['C10_null_mean']:.3f}  "
            f"enriched={row['C10_enriched_fraction']:.3f}",
            flush=True,
        )
    csv_path = args.out_dir / "soft_anm_panel.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}  n={len(rows)}", flush=True)
    summarize(rows)


if __name__ == "__main__":
    main()
    sys.exit(0)
