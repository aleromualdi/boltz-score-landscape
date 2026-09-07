"""Blind retrieval metrics for PSCDB soft-mode localization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type FloatArr = NDArray[np.float64]
type BoolArr = NDArray[np.bool_]


def select_mode_indices(
    evals: FloatArr,
    pr: FloatArr,
    method: str,
    *,
    pr_cut: float,
    n_cand: int,
) -> NDArray[np.int64]:
    """Select mode indices with a Δx-blind recipe.

    Args:
        evals: Eigenvalues softest-first.
        pr: Participation ratios (unused for ``top10_lambda``).
        method: Selection recipe name.
        pr_cut: Localization cutoff for ``soft_localized*``.
        n_cand: Candidate soft window size.

    Returns:
        Selected mode indices.

    Raises:
        ValueError: Unknown method name.
    """
    n_avail = len(evals)
    if n_avail == 0:
        return np.zeros(0, dtype=np.int64)
    window = min(n_cand, n_avail)
    if method == "top10_lambda":
        return np.arange(min(10, n_avail), dtype=np.int64)
    if method == "soft50":
        return np.arange(window, dtype=np.int64)
    if method in {"soft_localized", "soft_localized_invlam"}:
        cand = np.arange(window, dtype=np.int64)
        keep = cand[pr[cand] <= pr_cut]
        if len(keep) == 0:
            order = np.argsort(pr[cand])
            keep = cand[order[: min(10, window)]]
        return keep.astype(np.int64)
    msg = f"unknown method {method}"
    raise ValueError(msg)


def softness_score(
    modes: FloatArr,
    evals: FloatArr,
    indices: NDArray[np.int64],
    *,
    inv_lambda: bool,
) -> FloatArr:
    """Per-residue soft score from a selected mode set.

    Args:
        modes: Mode stack ``(K, N, 3)``.
        evals: Eigenvalues.
        indices: Selected mode indices.
        inv_lambda: Weight by ``1/λ`` if True.

    Returns:
        Score vector ``(N,)``.
    """
    n_sites = modes.shape[1] if len(modes) else 0
    scores = np.zeros(n_sites, dtype=np.float64)
    for idx in indices:
        weight = 1.0
        if inv_lambda:
            weight = 1.0 / max(float(evals[idx]), 1e-8)
        scores += weight * np.sum(modes[idx] * modes[idx], axis=1)
    return scores


def auroc(scores: FloatArr, labels: BoolArr) -> float:
    """AUROC via Mann–Whitney (no sklearn dependency).

    Args:
        scores: Continuous scores.
        labels: Binary labels.

    Returns:
        AUROC in ``[0, 1]``, or NaN if undefined.
    """
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j > i + 1:
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
        i = j
    sum_pos = float(np.sum(ranks[labels]))
    n_pos = float(len(pos))
    n_neg = float(len(neg))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auprc(scores: FloatArr, labels: BoolArr) -> float:
    """Average precision (AUPRC).

    Args:
        scores: Continuous scores.
        labels: Binary labels.

    Returns:
        Average precision, or NaN if undefined.
    """
    if not np.any(labels) or np.all(labels):
        return float("nan")
    order = np.argsort(scores)[::-1]
    y = labels[order].astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / float(np.sum(labels))
    ap = 0.0
    prev_r = 0.0
    for prec, rec, yi in zip(precision, recall, y, strict=True):
        if yi > 0.5:
            ap += (rec - prev_r) * prec
            prev_r = rec
    return float(ap)


def top_m_precision(scores: FloatArr, labels: BoolArr, m: int) -> float:
    """Precision among top-``m`` scored residues.

    Args:
        scores: Continuous scores.
        labels: Binary labels.
        m: Top count.

    Returns:
        Precision, or NaN if undefined.
    """
    m_use = min(m, len(scores))
    if m_use < 1:
        return float("nan")
    order = np.argsort(scores)[::-1][:m_use]
    return float(np.mean(labels[order]))
