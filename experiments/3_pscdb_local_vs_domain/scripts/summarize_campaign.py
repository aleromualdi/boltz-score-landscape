"""Summarize full campaign: local vs domain contrast."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

EXP_ROOT = Path(__file__).resolve().parents[1]
PER_SYSTEM = EXP_ROOT / "outputs" / "per_system"
OUT_DIR = EXP_ROOT / "outputs" / "analysis"

type FloatArr = NDArray[np.float64]


def bootstrap_median_ci(
    values: FloatArr, *, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Bootstrap median CI.

    Args:
        values: Samples.
        n_boot: Replicates.
        seed: RNG seed.

    Returns:
        ``(median, lo, hi)``.
    """
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=np.float64)
    n = values.size
    for i in range(n_boot):
        estimates[i] = float(np.median(values[rng.integers(0, n, size=n)]))
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(np.median(values)), float(lo), float(hi)


def agg(rows: list[dict[str, Any]], key: str, *, seed: int, n_boot: int) -> dict[str, Any]:
    """Aggregate a paired delta metric.

    Args:
        rows: OK result rows.
        key: Delta field name.
        seed: Bootstrap seed.
        n_boot: Replicates.

    Returns:
        Summary dict.
    """
    values = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    med, lo, hi = bootstrap_median_ci(values, n_boot=n_boot, seed=seed)
    return {
        "n": len(rows),
        f"med_{key}": med,
        f"{key}_ci95": [lo, hi],
        "boltz_wins": int(np.sum(values > 0)),
        "anm_wins": int(np.sum(values < 0)),
        "pass_ci_above_zero": bool(rows) and bool(lo > 0),
        "med_auroc_boltz": float(np.median([float(r["auroc_boltz"]) for r in rows]))
        if rows
        else float("nan"),
        "med_auroc_anm": float(np.median([float(r["auroc_anm"]) for r in rows]))
        if rows
        else float("nan"),
        "med_auprc_boltz": float(np.median([float(r["auprc_boltz"]) for r in rows]))
        if rows
        else float("nan"),
        "med_auprc_anm": float(np.median([float(r["auprc_anm"]) for r in rows]))
        if rows
        else float("nan"),
        "med_topm_boltz": float(np.median([float(r["topm_boltz"]) for r in rows]))
        if rows
        else float("nan"),
        "med_topm_anm": float(np.median([float(r["topm_anm"]) for r in rows]))
        if rows
        else float("nan"),
        "med_delta_auprc": float(np.median([float(r["delta_auprc"]) for r in rows]))
        if rows
        else float("nan"),
        "med_delta_topm": float(np.median([float(r["delta_topm"]) for r in rows]))
        if rows
        else float("nan"),
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=204)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for path in sorted(PER_SYSTEM.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("ok"):
            rows.append(payload)

    if not rows:
        msg = f"no OK results in {PER_SYSTEM}"
        raise FileNotFoundError(msg)

    panel_csv = args.out_dir / "retrieval_panel.csv"
    with panel_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_class: dict[str, dict[str, Any]] = {}
    for cls in ("IL", "CL", "CD", "ID"):
        subset = [r for r in rows if r["class"] == cls]
        by_class[cls] = agg(subset, "delta_auroc", seed=args.seed, n_boot=args.n_boot)

    local = [r for r in rows if r["class"] in {"IL", "CL"}]
    domain = [r for r in rows if r["class"] in {"CD", "ID"}]
    local_stats = agg(local, "delta_auroc", seed=args.seed + 1, n_boot=args.n_boot)
    domain_stats = agg(domain, "delta_auroc", seed=args.seed + 2, n_boot=args.n_boot)

    # Contrast of medians with bootstrap on both pools (difference of medians).
    rng = np.random.default_rng(args.seed + 3)
    local_d = np.asarray([float(r["delta_auroc"]) for r in local], dtype=np.float64)
    domain_d = np.asarray([float(r["delta_auroc"]) for r in domain], dtype=np.float64)
    contrast_boot = np.empty(args.n_boot, dtype=np.float64)
    for i in range(args.n_boot):
        if local_d.size and domain_d.size:
            contrast_boot[i] = float(
                np.median(local_d[rng.integers(0, local_d.size, size=local_d.size)])
                - np.median(domain_d[rng.integers(0, domain_d.size, size=domain_d.size)])
            )
        else:
            contrast_boot[i] = float("nan")
    contrast_point = float(np.median(local_d) - np.median(domain_d)) if local_d.size and domain_d.size else float("nan")
    clo, chi = (
        (float(np.percentile(contrast_boot, 2.5)), float(np.percentile(contrast_boot, 97.5)))
        if local_d.size and domain_d.size
        else (float("nan"), float("nan"))
    )

    summary = {
        "n_ok": len(rows),
        "by_class": by_class,
        "local_all": local_stats,
        "domain_all": domain_stats,
        "contrast_local_minus_domain": {
            "med_delta_auroc": contrast_point,
            "ci95": [clo, chi],
            "pass_ci_above_zero": bool(math.isfinite(clo) and clo > 0),
        },
        "hypothesis": {
            "local_delta_auroc_gt_0": local_stats["pass_ci_above_zero"],
            "domain_delta_auroc_approx_0": not domain_stats["pass_ci_above_zero"],
            "contrast_gt_0": bool(math.isfinite(clo) and clo > 0),
        },
    }
    (args.out_dir / "summary_stats.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Exp 3 — Full PSCDB local vs domain",
        "",
        f"OK systems: **{len(rows)}**",
        "",
        "## By class (ΔAUROC = Boltz − ANM)",
        "",
        "| class | n | med AUROC B/A | med ΔAUROC [95% CI] | med ΔAUPRC | med Δtop-m | wins B/A | lock |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for cls in ("IL", "CL", "CD", "ID"):
        s = by_class[cls]
        ci = s["delta_auroc_ci95"]
        lines.append(
            f"| {cls} | {s['n']} | {s['med_auroc_boltz']:.3f}/{s['med_auroc_anm']:.3f} | "
            f"{s['med_delta_auroc']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{s['med_delta_auprc']:+.3f} | {s['med_delta_topm']:+.3f} | "
            f"{s['boltz_wins']}/{s['anm_wins']} | "
            f"{'yes' if s['pass_ci_above_zero'] else 'no'} |"
        )
    for label, s in (("IL+CL local", local_stats), ("CD+ID domain", domain_stats)):
        ci = s["delta_auroc_ci95"]
        lines.append(
            f"| **{label}** | {s['n']} | {s['med_auroc_boltz']:.3f}/{s['med_auroc_anm']:.3f} | "
            f"**{s['med_delta_auroc']:+.3f}** [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{s['med_delta_auprc']:+.3f} | {s['med_delta_topm']:+.3f} | "
            f"{s['boltz_wins']}/{s['anm_wins']} | "
            f"{'yes' if s['pass_ci_above_zero'] else 'no'} |"
        )

    hyp = summary["hypothesis"]
    c = summary["contrast_local_minus_domain"]
    lines.extend(
        [
            "",
            "## Primary contrast (local − domain) med ΔAUROC",
            "",
            f"**{c['med_delta_auroc']:+.3f}** [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  "
            f"lock CI>0: **{'PASS' if c['pass_ci_above_zero'] else 'no'}**",
            "",
            "## Hypothesis checklist",
            "",
            f"- ΔAUROC_local > 0: **{'PASS' if hyp['local_delta_auroc_gt_0'] else 'FAIL'}**",
            f"- ΔAUROC_domain ≉ locked >0: **{'PASS' if hyp['domain_delta_auroc_approx_0'] else 'FAIL'}**",
            f"- (CL+IL)−(CD+ID) > 0: **{'PASS' if hyp['contrast_gt_0'] else 'FAIL'}**",
            "",
        ]
    )
    md = args.out_dir / "SUMMARY.md"
    md.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {panel_csv}", flush=True)
    print(f"wrote {md}", flush=True)


if __name__ == "__main__":
    main()
