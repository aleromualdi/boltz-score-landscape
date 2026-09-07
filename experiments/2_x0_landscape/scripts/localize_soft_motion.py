"""Localize experimental flexibility from Boltz soft modes (Exp 2).

This script tests whether Boltz soft modes can identify *where* proteins move
from a single predicted structure. For each system it computes:

- Boltz soft participation, ``S_i = sum_{k=1..K} ||v_{k,i}||^2``.
- Experimental per-residue displacement, ``D_i = ||Delta x_i||``.
- Optional ANM reference participation on the same CA coordinates.

It writes per-residue CSVs, a per-system summary CSV, and a representative
figure with one row per system.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from numpy.typing import NDArray


EXP_ROOT = Path(__file__).resolve().parents[1]
EXP100_ROOT = Path(__file__).resolve().parents[2] / "1_codnas_apo_holo"
EXP100_SCRIPTS = EXP100_ROOT / "scripts"
RUN_HESSIAN = EXP100_SCRIPTS / "run_hessian.py"

sys.path.insert(0, str(EXP100_SCRIPTS))

from geom import apo_holo_delta_aligned, kabsch_rotation, load_ca_sequence, matched_residue_indices, parse_label
from load_pose import load_cif_atoms
from pairs import pair_by_apo, pdb_path


type FloatArr = NDArray[np.float64]


N_TERM_MASK = 2
N_MORPH_FRAMES = 11
MORPH_MAX_DISP_A = 7.0
MORPH_CMAP = plt.get_cmap("coolwarm")
MORPH_CAPTIONS = {
    "1AW2_A": "Loop",
    "1GUD_A": "Hinge",
    "1EX6_B": "Domain closure",
}


@dataclass(frozen=True)
class SystemPaths:
    """Resolved paths for one system.

    Attributes:
        system: Apo identifier.
        soft_json: Soft-enrichment JSON path.
        mode_npz: CA soft-mode NPZ path.
        predict_root: Directory containing `boltz_results_*/`.
        processed_dir: Boltz `processed/` folder.
        cif_path: Predicted model path.
    """

    system: str
    soft_json: Path
    mode_npz: Path
    predict_root: Path
    processed_dir: Path
    cif_path: Path


@dataclass(frozen=True)
class SystemResult:
    """Summary metrics for one system.

    Attributes:
        system: Apo identifier.
        motion_type: Motion class from CoDNaS table.
        n_overlap: Number of matched CA residues in analysis.
        spearman_boltz: Spearman correlation between normalized S and D.
        topk_overlap_boltz: Overlap of top-q residues for S and D.
        spearman_anm: Optional Spearman correlation for ANM participation.
        topk_overlap_anm: Optional top-q overlap for ANM participation.
    """

    system: str
    motion_type: str
    n_overlap: int
    spearman_boltz: float
    topk_overlap_boltz: float
    spearman_anm: float | None
    topk_overlap_anm: float | None


def _safe_normalize(values: FloatArr) -> FloatArr:
    """Normalize a vector to [0, 1].

    Args:
        values: Input values.

    Returns:
        Normalized values or zeros when constant.
    """
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax <= vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _mask_terminal_artifact(values: FloatArr) -> FloatArr:
    """Zero unconstrained terminal spikes before display normalization.

    Args:
        values: Per-residue scores.

    Returns:
        Copy with terminal residues set to zero when the chain is long enough.
    """
    if values.size <= 2 * N_TERM_MASK + 8:
        return values
    out = values.copy()
    out[:N_TERM_MASK] = 0.0
    out[-N_TERM_MASK:] = 0.0
    return out


def _rankdata(values: FloatArr) -> FloatArr:
    """Return average-tie ranks for a 1D array.

    Args:
        values: Input vector.

    Returns:
        Rank vector starting at 1.
    """
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    idx = 0
    while idx < len(values):
        end = idx + 1
        while end < len(values) and sorted_vals[end] == sorted_vals[idx]:
            end += 1
        avg_rank = 0.5 * (idx + 1 + end)
        ranks[order[idx:end]] = avg_rank
        idx = end
    return ranks


def _pearson(x: FloatArr, y: FloatArr) -> float:
    """Compute Pearson correlation.

    Args:
        x: First vector.
        y: Second vector.

    Returns:
        Pearson correlation, or 0.0 if degenerate.
    """
    x_center = x - np.mean(x)
    y_center = y - np.mean(y)
    denom = float(np.linalg.norm(x_center) * np.linalg.norm(y_center))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x_center, y_center) / denom)


def spearman(x: FloatArr, y: FloatArr) -> float:
    """Compute Spearman rank correlation.

    Args:
        x: First vector.
        y: Second vector.

    Returns:
        Spearman rank correlation.
    """
    return _pearson(_rankdata(x), _rankdata(y))


def top_fraction_overlap(a: FloatArr, b: FloatArr, fraction: float) -> float:
    """Compute overlap of top-ranked residues for two score vectors.

    Args:
        a: First score vector.
        b: Second score vector.
        fraction: Fraction in (0, 1] for selecting top residues.

    Returns:
        Overlap ratio in [0, 1].
    """
    n_sites = len(a)
    k_sites = max(1, int(round(fraction * n_sites)))
    top_a = set(np.argsort(a)[-k_sites:].tolist())
    top_b = set(np.argsort(b)[-k_sites:].tolist())
    return float(len(top_a & top_b) / k_sites)


def _infer_paths(system: str, json_dir: Path, mode_dir: Path) -> SystemPaths:
    """Infer all system-specific paths from Exp 100 outputs.

    Args:
        system: Apo id.
        json_dir: Directory with `*_soft_enrichment.json`.
        mode_dir: Directory with `*_score_ascent_ca_modes.npz`.

    Returns:
        Resolved path bundle.
    """
    soft_json = json_dir / f"{system}_soft_enrichment.json"
    if not soft_json.exists():
        msg = f"missing soft-enrichment json: {soft_json}"
        raise FileNotFoundError(msg)
    payload = json.loads(soft_json.read_text())
    cif_path = Path(str(payload["cif"]).replace(
        "/experiments/100_codnas_apo_holo/",
        "/experiments/1_codnas_apo_holo/",
    ))
    if not cif_path.exists():
        # Fall back to the current Exp-1 predict tree when JSON still has old paths.
        alt = (
            EXP100_ROOT
            / "outputs"
            / "predict_soft91"
            / system
            / f"boltz_results_{system}"
            / "predictions"
            / system
            / f"{system}_model_0.cif"
        )
        if alt.exists():
            cif_path = alt
    results_dir = cif_path.parents[2]
    predict_root = results_dir.parent
    processed_dir = results_dir / "processed"
    mode_npz = mode_dir / f"{system}_score_ascent_ca_modes.npz"
    return SystemPaths(
        system=system,
        soft_json=soft_json,
        mode_npz=mode_npz,
        predict_root=predict_root,
        processed_dir=processed_dir,
        cif_path=cif_path,
    )


def _ensure_modes(paths: SystemPaths, n_soft_modes: int, lanczos_steps: int, force: bool) -> None:
    """Build missing CA mode NPZ by calling Exp 100 `run_hessian.py`.

    Args:
        paths: Resolved system paths.
        n_soft_modes: Number of soft modes requested.
        lanczos_steps: Lanczos iteration count.
        force: Whether to overwrite existing files.
    """
    if paths.mode_npz.exists() and not force:
        return
    command = [
        sys.executable,
        str(RUN_HESSIAN),
        "--system",
        paths.system,
        "--processed_dir",
        str(paths.processed_dir),
        "--predict_root",
        str(paths.predict_root),
        "--out_dir",
        str(paths.mode_npz.parent),
        "--pose",
        "predicted",
        "--n_soft_modes",
        str(n_soft_modes),
        "--lanczos_steps",
        str(lanczos_steps),
    ]
    if force:
        command.append("--force")
    subprocess.run(command, check=True)


def _load_mode_npz(mode_npz: Path) -> tuple[FloatArr, FloatArr, NDArray[np.int64], FloatArr]:
    """Load mode arrays from run_hessian output.

    Args:
        mode_npz: Path to `*_score_ascent_ca_modes.npz`.

    Returns:
        Tuple `(modes, evals, ca_idx, pred_ca)`.
    """
    data = np.load(mode_npz, allow_pickle=True)
    modes = np.asarray(data["modes"], dtype=np.float64)
    evals = np.asarray(data["evals"], dtype=np.float64)
    ca_idx = np.asarray(data["ca_idx"], dtype=np.int64)
    pred_ca = np.asarray(data["pred_ca"], dtype=np.float64)
    return modes, evals, ca_idx, pred_ca


def _aligned_experimental_delta(paths: SystemPaths) -> tuple[FloatArr, NDArray[np.int64]]:
    """Compute experimental apo->holo displacement in predicted-pose frame.

    Args:
        paths: Resolved system paths.

    Returns:
        Tuple `(delta_pose, ca_idx_overlap)` for sequence-matched residues.
    """
    spec = pair_by_apo()[paths.system]
    _p, apo_model, _ = parse_label(spec["apo_id"])
    _p, holo_model, _ = parse_label(spec["holo_id"])
    apo_ca, apo_seq = load_ca_sequence(pdb_path(spec["apo_id"]), apo_model)
    holo_ca, holo_seq = load_ca_sequence(pdb_path(spec["holo_id"]), holo_model)
    apo_matched, delta, _rmsd, apo_match_idx, _ = apo_holo_delta_aligned(
        apo_ca, apo_seq, holo_ca, holo_seq
    )
    names, xyz = load_cif_atoms(paths.cif_path)
    pred_ca_all = xyz[[idx for idx, name in enumerate(names) if name == "CA"]]
    npz = np.load(paths.processed_dir / "structures" / f"{paths.system}.npz")
    boltz_res = npz["residues"]
    boltz_seq = "".join(str(name).strip()[0] for name in boltz_res["name"])
    pred_idx, apo_idx = matched_residue_indices(boltz_seq, apo_seq)
    apo_to_delta = {int(i): k for k, i in enumerate(apo_match_idx)}
    keep_pred: list[int] = []
    keep_delta: list[int] = []
    for pred_site, apo_site in zip(pred_idx, apo_idx, strict=True):
        if int(apo_site) in apo_to_delta:
            keep_pred.append(int(pred_site))
            keep_delta.append(apo_to_delta[int(apo_site)])
    pred_ca = pred_ca_all[keep_pred]
    delta_match = delta[keep_delta]
    apo_match = apo_matched[keep_delta]
    rot, _cm, _ct = kabsch_rotation(pred_ca, apo_match)
    delta_pose = delta_match @ rot.T
    ca_idx_full = np.where(np.asarray(npz["atoms"]["name"]) == "CA")[0].astype(np.int64)
    ca_idx_overlap = ca_idx_full[np.asarray(keep_pred, dtype=np.int64)]
    return delta_pose.astype(np.float64), ca_idx_overlap


def soft_participation(modes: FloatArr, k_soft: int) -> FloatArr:
    """Compute per-residue soft participation score.

    Args:
        modes: Soft modes with shape `(K, N, 3)`.
        k_soft: Number of leading soft modes to include.

    Returns:
        Participation vector `S` with shape `(N,)`.
    """
    n_keep = min(k_soft, modes.shape[0])
    return np.sum(np.linalg.norm(modes[:n_keep], axis=2) ** 2, axis=0)


def _anm_hessian(ca: FloatArr, cutoff: float, gamma: float) -> FloatArr:
    """Build a simple C-alpha ANM Hessian.

    Args:
        ca: CA coordinates `(N, 3)`.
        cutoff: Contact cutoff in Angstrom.
        gamma: Uniform spring constant.

    Returns:
        ANM Hessian matrix `(3N, 3N)`.
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


def anm_participation(ca: FloatArr, k_soft: int, cutoff: float, gamma: float) -> FloatArr:
    """Compute ANM participation score analogous to Boltz `S_i`.

    Args:
        ca: CA coordinates `(N, 3)`.
        k_soft: Number of non-rigid low modes to include.
        cutoff: ANM contact cutoff in Angstrom.
        gamma: Uniform spring constant.

    Returns:
        Per-residue ANM participation vector.
    """
    n_ca = ca.shape[0]
    if n_ca < 4:
        return np.zeros(n_ca, dtype=np.float64)
    hess = _anm_hessian(ca, cutoff=cutoff, gamma=gamma)
    evals, evecs = np.linalg.eigh(hess)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    keep = np.where(evals > 1e-8)[0]
    if len(keep) == 0:
        return np.zeros(n_ca, dtype=np.float64)
    take = keep[:k_soft]
    modes = evecs[:, take].T.reshape(len(take), n_ca, 3)
    return np.sum(np.linalg.norm(modes, axis=2) ** 2, axis=0)


def _write_per_residue_csv(
    out_csv: Path,
    s_boltz: FloatArr,
    d_exp: FloatArr,
    s_anm: FloatArr | None,
) -> None:
    """Write per-residue analysis table.

    Args:
        out_csv: Output CSV path.
        s_boltz: Boltz participation scores.
        d_exp: Experimental displacement magnitudes.
        s_anm: Optional ANM participation scores.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        fieldnames = [
            "residue_index_1based",
            "S_boltz",
            "S_boltz_norm",
            "D_exp",
            "D_exp_norm",
        ]
        if s_anm is not None:
            fieldnames.extend(["S_anm", "S_anm_norm"])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        s_norm = _safe_normalize(s_boltz)
        d_norm = _safe_normalize(d_exp)
        anm_norm = _safe_normalize(s_anm) if s_anm is not None else None
        for idx in range(len(s_boltz)):
            row: dict[str, float | int] = {
                "residue_index_1based": idx + 1,
                "S_boltz": float(s_boltz[idx]),
                "S_boltz_norm": float(s_norm[idx]),
                "D_exp": float(d_exp[idx]),
                "D_exp_norm": float(d_norm[idx]),
            }
            if s_anm is not None and anm_norm is not None:
                row["S_anm"] = float(s_anm[idx])
                row["S_anm_norm"] = float(anm_norm[idx])
            writer.writerow(row)

@dataclass(frozen=True)
class PlotRow:
    """Per-system arrays used for the representative figure.

    Attributes:
        system: Apo identifier.
        motion_type: Motion class label.
        d_norm: Normalized experimental mobility.
        s_norm: Normalized Boltz soft participation.
        s_anm_norm: Optional normalized ANM participation.
        morph_frames: Soft-mode morph frames ``(T, N, 3)``.
        mode_index: 1-based index of the displayed soft mode.
        mode_cosine: Absolute cosine of that mode with experimental ``Delta x``.
        draw_domain_arrows: Whether to annotate collective domain motion.
    """

    system: str
    motion_type: str
    d_norm: FloatArr
    s_norm: FloatArr
    s_anm_norm: FloatArr | None
    morph_frames: FloatArr
    mode_index: int
    mode_cosine: float
    draw_domain_arrows: bool


def _morph_caption(system: str, motion_type: str) -> str:
    """Return a short motion caption for one system.

    Args:
        system: Apo identifier.
        motion_type: CoDNaS motion class.

    Returns:
        Figure caption.
    """
    if system in MORPH_CAPTIONS:
        return MORPH_CAPTIONS[system]
    lowered = motion_type.lower()
    if "loop" in lowered:
        return "Loop"
    if "domain" in lowered:
        return "Domain"
    return motion_type


def _select_soft_mode(
    modes: FloatArr, delta: FloatArr, k_soft: int
) -> tuple[int, FloatArr, float]:
    """Pick the soft mode with largest overlap onto experimental displacement.

    Args:
        modes: Soft modes ``(K, N, 3)``.
        delta: Experimental displacement ``(N, 3)``.
        k_soft: Number of leading modes to consider.

    Returns:
        Tuple ``(index, signed_mode, abs_cosine)``.
    """
    n_keep = min(k_soft, modes.shape[0])
    delta_flat = delta.reshape(-1)
    denom = float(np.linalg.norm(delta_flat))
    if denom <= 1e-12:
        return 0, modes[0], 0.0
    unit_delta = delta_flat / denom
    scores = np.zeros(n_keep, dtype=np.float64)
    for idx in range(n_keep):
        vec = modes[idx].reshape(-1)
        nrm = float(np.linalg.norm(vec))
        if nrm <= 1e-12:
            continue
        scores[idx] = float(vec @ unit_delta / nrm)
    best = int(np.argmax(np.abs(scores)))
    signed = modes[best].copy()
    if scores[best] < 0.0:
        signed *= -1.0
    return best, signed, float(abs(scores[best]))


def _mode_amplitude(mode: FloatArr, target_max_disp: float) -> float:
    """Scale a mode so the largest interior CA move equals a target.

    Args:
        mode: Mode displacements ``(N, 3)``.
        target_max_disp: Target max CA displacement in Angstrom.

    Returns:
        Amplitude ``alpha``.
    """
    mag = np.linalg.norm(mode, axis=1)
    if mag.size > 2 * N_TERM_MASK + 4:
        mag = mag[N_TERM_MASK:-N_TERM_MASK]
    peak = float(np.max(mag))
    if peak <= 1e-12:
        return 0.0
    return target_max_disp / peak


def _morph_frames(x0: FloatArr, mode: FloatArr, alpha: float, n_frames: int) -> FloatArr:
    """Build endpoint-interpolated coordinates ``x0 + t alpha v``.

    Args:
        x0: Reference CA coordinates.
        mode: Signed mode displacements.
        alpha: Amplitude.
        n_frames: Number of samples from ``t=-1`` to ``t=+1``.

    Returns:
        Frame stack ``(T, N, 3)``.
    """
    scales = np.linspace(-1.0, 1.0, n_frames)
    return x0[None, :, :] + scales[:, None, None] * alpha * mode[None, :, :]


def _motion_plane(frames: FloatArr) -> FloatArr:
    """Project morph frames onto the deformation plane.

    Args:
        frames: Frame stack ``(T, N, 3)``.

    Returns:
        Projected coordinates ``(T, N, 2)``.
    """
    cloud = frames.reshape(-1, 3)
    centroid = np.mean(cloud, axis=0)
    centered = cloud - centroid
    _u_mat, _s_vals, vt_mat = np.linalg.svd(centered, full_matrices=False)
    axes = vt_mat[:2]
    xy_coords = (frames - centroid).reshape(-1, 3) @ axes.T
    return xy_coords.reshape(frames.shape[0], frames.shape[1], 2)


def _draw_morph_trace(
    ax: Axes, xy_coords: FloatArr, color: str, lw: float, alpha: float
) -> None:
    """Draw one CA backbone morph trace.

    Args:
        ax: Target axes.
        xy_coords: Projected coordinates ``(N, 2)``.
        color: Line color.
        lw: Line width.
        alpha: Opacity.
    """
    points = xy_coords.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    ax.add_collection(
        LineCollection(
            segments,
            colors=color,
            linewidths=lw,
            capstyle="round",
            joinstyle="round",
            alpha=alpha,
            zorder=2,
        )
    )


def _domain_clusters(
    xy0: FloatArr, xy_m: FloatArr, xy_p: FloatArr
) -> list[NDArray[np.int64]]:
    """Split high-amplitude residues into two spatial domain clusters.

    Args:
        xy0: Predicted backbone in the plot plane.
        xy_m: ``x^-`` coordinates.
        xy_p: ``x^+`` coordinates.

    Returns:
        Up to two residue-index arrays for the moving domains.
    """
    mag = np.linalg.norm(xy_p - xy_m, axis=1)
    if mag.size > 2 * N_TERM_MASK + 4:
        mag = mag.copy()
        mag[:N_TERM_MASK] = 0.0
        mag[-N_TERM_MASK:] = 0.0
    peak = float(np.percentile(mag, 90))
    if peak <= 1e-12:
        return []
    mobile = np.where(mag >= 0.40 * peak)[0]
    if len(mobile) < 8:
        return [mobile] if len(mobile) else []
    pts = xy0[mobile] - np.mean(xy0[mobile], axis=0)
    _u_mat, _s_vals, vt_mat = np.linalg.svd(pts, full_matrices=False)
    side = pts @ vt_mat[0]
    left = mobile[side >= 0.0]
    right = mobile[side < 0.0]
    return [arr for arr in (left, right) if len(arr) >= 4]


def _draw_domain_arrows(ax: Axes, xy_m: FloatArr, xy0: FloatArr, xy_p: FloatArr) -> None:
    """Draw large curved arrows for collective domain motion.

    Args:
        ax: Target axes.
        xy_m: ``x^-`` coordinates.
        xy0: ``x0`` coordinates.
        xy_p: ``x^+`` coordinates.
    """
    clusters = _domain_clusters(xy0, xy_m, xy_p)
    center = np.mean(xy0, axis=0)
    span = max(float(np.ptp(xy0[:, 0])), float(np.ptp(xy0[:, 1])), 1.0)
    offset = 0.28 * span
    arrow_len = 0.22 * span
    for idx in clusters:
        start = np.mean(xy_m[idx], axis=0)
        end = np.mean(xy_p[idx], axis=0)
        mid = np.mean(xy0[idx], axis=0)
        chord = end - start
        chord_n = float(np.linalg.norm(chord))
        if chord_n <= 1e-8:
            continue
        direction = chord / chord_n * arrow_len
        outward = mid - center
        nrm = float(np.linalg.norm(outward))
        if nrm <= 1e-8:
            continue
        shift = outward / nrm * offset
        anchor = mid + shift
        p0 = anchor - 0.5 * direction
        p1 = anchor + 0.5 * direction
        cross = float(direction[0] * shift[1] - direction[1] * shift[0])
        rad = 0.45 * (1.0 if cross >= 0.0 else -1.0)
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-|>",
                mutation_scale=18,
                connectionstyle=f"arc3,rad={rad:.2f}",
                color="#111111",
                lw=1.8,
                zorder=8,
            )
        )


def _draw_morph_panel(ax: Axes, row: PlotRow, norm: Normalize) -> None:
    """Draw one soft-mode morph panel.

    Args:
        ax: Target axes.
        row: Per-system plot payload.
        norm: Shared color normalization for morph frames.
    """
    xy_all = _motion_plane(row.morph_frames)
    n_frames = xy_all.shape[0]
    t_vals = np.linspace(-1.0, 1.0, n_frames)
    for frame_idx, t_val in enumerate(t_vals):
        is_center = abs(t_val) < 1e-12
        color = "#2c3e50" if is_center else MORPH_CMAP(norm(t_val))
        lw = 2.6 if is_center else 1.5
        alpha = 1.0 if is_center else 0.70
        _draw_morph_trace(ax, xy_all[frame_idx], color=color, lw=lw, alpha=alpha)
    mid = n_frames // 2
    if row.draw_domain_arrows:
        _draw_domain_arrows(ax, xy_all[0], xy_all[mid], xy_all[-1])
    pts = xy_all.reshape(-1, 2)
    pad_frac = 0.20 if row.draw_domain_arrows else 0.10
    pad = pad_frac * max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
    ax.set_xlim(float(np.min(pts[:, 0])) - pad, float(np.max(pts[:, 0])) + pad)
    ax.set_ylim(float(np.min(pts[:, 1])) - pad, float(np.max(pts[:, 1])) + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    caption = _morph_caption(row.system, row.motion_type)
    ax.set_title(
        f"{row.system} · {caption}\n"
        rf"$v_{{{row.mode_index}}}$, $|\cos\theta|={row.mode_cosine:.2f}$",
        fontsize=10,
        pad=4,
    )


def plot_representative_rows(
    series: list[PlotRow],
    out_png: Path,
    *,
    title: str = "Exp 2: Soft-mode morphs and residue-level localization",
) -> None:
    """Plot morph overlays plus per-residue localization profiles.

    Args:
        series: Per-system plot payloads.
        out_png: Output figure path.
        title: Figure title.
    """
    n_rows = len(series)
    fig = plt.figure(figsize=(14.8, 3.15 * n_rows + 0.7))
    outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.0, 0.06],
        hspace=0.08,
        left=0.04,
        right=0.98,
        top=0.90,
        bottom=0.06,
    )
    grid = outer[0].subgridspec(
        n_rows,
        2,
        width_ratios=[1.15, 2.55],
        wspace=0.18,
        hspace=0.38,
    )
    fig.suptitle(title, fontsize=14, y=0.98)
    norm = Normalize(vmin=-1.0, vmax=1.0)
    profile_axes: list[Axes] = []

    for row_idx, row in enumerate(series):
        ax_morph = fig.add_subplot(grid[row_idx, 0])
        ax_prof = fig.add_subplot(grid[row_idx, 1])
        profile_axes.append(ax_prof)
        _draw_morph_panel(ax_morph, row, norm)

        x_vals = np.arange(1, len(row.d_norm) + 1)
        ax_prof.plot(x_vals, row.d_norm, color="#111111", lw=1.8, label="Experimental D_i")
        ax_prof.plot(x_vals, row.s_norm, color="#d35400", lw=1.6, label="Boltz soft S_i")
        if row.s_anm_norm is not None:
            ax_prof.plot(
                x_vals,
                row.s_anm_norm,
                color="#1f77b4",
                lw=1.3,
                ls="--",
                label="ANM S_i^NMA",
            )
        ax_prof.set_ylim(-0.02, 1.05)
        ax_prof.set_ylabel("Normalized score")
        ax_prof.set_title(f"{row.system} — {row.motion_type}", loc="left", fontsize=11)
        ax_prof.grid(alpha=0.2, axis="y")
        ax_prof.legend(frameon=False, loc="upper right", fontsize=8)

    profile_axes[-1].set_xlabel("Residue index (sequence-matched CA sites)")
    cbar_grid = outer[1].subgridspec(1, 2, width_ratios=[1.15, 2.55], wspace=0.18)
    cax = fig.add_subplot(cbar_grid[0, 0])
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=MORPH_CMAP),
        cax=cax,
        orientation="horizontal",
    )
    cbar.set_ticks([-1.0, 0.0, 1.0])
    cbar.set_ticklabels([r"$x^-$", r"$x_0$", r"$x^+$"])
    cbar.set_label(r"$x_0 + t\,\alpha v_k$", fontsize=9)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)



def main() -> None:
    """Run soft-mode localization analysis for representative systems."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        default="1EX6_B,1GUD_A,1AW2_A",
        help="Comma-separated apo ids for representative analysis.",
    )
    parser.add_argument(
        "--soft_json_dir",
        type=Path,
        default=EXP100_ROOT / "outputs" / "analysis_soft91",
    )
    parser.add_argument(
        "--mode_dir",
        type=Path,
        default=EXP100_ROOT / "outputs" / "analysis_soft91",
    )
    parser.add_argument("--out_dir", type=Path, default=EXP_ROOT / "outputs" / "localization")
    parser.add_argument("--fig_name", default="soft_localization_representative.png")
    parser.add_argument(
        "--title",
        default="Exp 2: Soft-mode morphs and residue-level localization",
    )
    parser.add_argument("--k_soft", type=int, default=10)
    parser.add_argument("--top_fraction", type=float, default=0.15)
    parser.add_argument("--with_anm", action="store_true", help="Compute ANM reference scores.")
    parser.add_argument(
        "--auto_build_modes",
        action="store_true",
        help="Run Exp 1 run_hessian.py when mode NPZ files are missing.",
    )
    parser.add_argument("--lanczos_steps", type=int, default=40)
    parser.add_argument("--n_soft_modes", type=int, default=15)
    parser.add_argument("--anm_cutoff", type=float, default=15.0)
    parser.add_argument("--anm_gamma", type=float, default=1.0)
    parser.add_argument("--force_modes", action="store_true")
    args = parser.parse_args()

    systems = [token.strip() for token in args.systems.split(",") if token.strip()]
    if not systems:
        raise ValueError("no systems provided")

    out_per_res = args.out_dir / "per_residue"
    out_per_res.mkdir(parents=True, exist_ok=True)

    summary_rows: list[SystemResult] = []
    plot_rows: list[PlotRow] = []
    pair_table = pair_by_apo()

    for system in systems:
        if system not in pair_table:
            raise KeyError(f"{system} not in pair table")
        paths = _infer_paths(system, args.soft_json_dir, args.mode_dir)
        if not paths.mode_npz.exists():
            if not args.auto_build_modes:
                raise FileNotFoundError(
                    f"missing mode NPZ {paths.mode_npz}; rerun with --auto_build_modes"
                )
            _ensure_modes(
                paths,
                n_soft_modes=max(args.n_soft_modes, args.k_soft),
                lanczos_steps=args.lanczos_steps,
                force=args.force_modes,
            )

        modes, _evals, ca_idx_full, pred_ca_full = _load_mode_npz(paths.mode_npz)
        delta_pose, ca_idx_overlap = _aligned_experimental_delta(paths)

        atom_to_pos = {int(atom_idx): pos for pos, atom_idx in enumerate(ca_idx_full.tolist())}
        keep_pos = [atom_to_pos[int(atom_idx)] for atom_idx in ca_idx_overlap]
        modes_overlap = modes[:, keep_pos, :]
        pred_ca_overlap = pred_ca_full[keep_pos]

        s_boltz = soft_participation(modes_overlap, k_soft=args.k_soft)
        d_exp = np.linalg.norm(delta_pose, axis=1)
        s_boltz_plot = _safe_normalize(_mask_terminal_artifact(s_boltz))
        d_exp_norm = _safe_normalize(d_exp)

        s_anm: FloatArr | None = None
        s_anm_norm: FloatArr | None = None
        if args.with_anm:
            s_anm = anm_participation(
                pred_ca_overlap,
                k_soft=args.k_soft,
                cutoff=args.anm_cutoff,
                gamma=args.anm_gamma,
            )
            s_anm_norm = _safe_normalize(_mask_terminal_artifact(s_anm))

        per_res_csv = out_per_res / f"{system}_localization.csv"
        _write_per_residue_csv(per_res_csv, s_boltz=s_boltz, d_exp=d_exp, s_anm=s_anm)

        spearman_b = spearman(s_boltz_plot, d_exp_norm)
        overlap_b = top_fraction_overlap(s_boltz_plot, d_exp_norm, fraction=args.top_fraction)
        spearman_nma = spearman(s_anm_norm, d_exp_norm) if s_anm_norm is not None else None
        overlap_nma = (
            top_fraction_overlap(s_anm_norm, d_exp_norm, fraction=args.top_fraction)
            if s_anm_norm is not None
            else None
        )
        motion_type = str(pair_table[system]["motion_type"])
        summary_rows.append(
            SystemResult(
                system=system,
                motion_type=motion_type,
                n_overlap=len(d_exp_norm),
                spearman_boltz=spearman_b,
                topk_overlap_boltz=overlap_b,
                spearman_anm=spearman_nma,
                topk_overlap_anm=overlap_nma,
            )
        )

        mode_idx, signed_mode, mode_cos = _select_soft_mode(
            modes_overlap, delta_pose, k_soft=args.k_soft
        )
        alpha = _mode_amplitude(signed_mode, target_max_disp=MORPH_MAX_DISP_A)
        morph = _morph_frames(
            pred_ca_overlap, signed_mode, alpha=alpha, n_frames=N_MORPH_FRAMES
        )
        caption = _morph_caption(system, motion_type)
        plot_rows.append(
            PlotRow(
                system=system,
                motion_type=motion_type,
                d_norm=d_exp_norm,
                s_norm=s_boltz_plot,
                s_anm_norm=s_anm_norm,
                morph_frames=morph,
                mode_index=mode_idx + 1,
                mode_cosine=mode_cos,
                draw_domain_arrows=caption != "Loop" and "loop" not in motion_type.lower(),
            )
        )
        print(
            f"{system}: n={len(d_exp_norm)} mode=v_{mode_idx + 1} |cos|={mode_cos:.3f} "
            f"rho_boltz={spearman_b:.3f} class={motion_type}",
            flush=True,
        )

    summary_csv = args.out_dir / "localization_summary.csv"
    with summary_csv.open("w", newline="") as handle:
        fieldnames = [
            "system",
            "motion_type",
            "n_overlap",
            "spearman_boltz",
            "topk_overlap_boltz",
            "spearman_anm",
            "topk_overlap_anm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "system": row.system,
                    "motion_type": row.motion_type,
                    "n_overlap": row.n_overlap,
                    "spearman_boltz": row.spearman_boltz,
                    "topk_overlap_boltz": row.topk_overlap_boltz,
                    "spearman_anm": row.spearman_anm,
                    "topk_overlap_anm": row.topk_overlap_anm,
                }
            )
    fig_path = args.out_dir / args.fig_name
    plot_representative_rows(plot_rows, fig_path, title=args.title)
    print(f"wrote {summary_csv}", flush=True)
    print(f"wrote {fig_path}", flush=True)


if __name__ == "__main__":
    main()
