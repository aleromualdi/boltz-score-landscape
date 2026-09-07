"""Compose ``outputs/score_landscape/trio_3d_2d.png``.

Merges landscape cols 1–2 with soft-mode morph cartoons and per-residue
``D_i`` / Boltz ``S_i`` / ANM profiles. Draws a clean morph colorbar under
column 3. This script writes only ``trio_3d_2d.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXP_ROOT = Path(__file__).resolve().parents[1]
LAND_PATH = EXP_ROOT / "outputs" / "score_landscape" / "trio_3d_2d_landscape_only.png"
LOC_PATH = (
    EXP_ROOT / "outputs" / "localization" / "soft_localization_representative.png"
)
OUT_PATH = EXP_ROOT / "outputs" / "score_landscape" / "trio_3d_2d.png"


def _gutter_cuts(dark: np.ndarray, axis: int, thr_pct: float) -> list[tuple[int, int]]:
    """Find contiguous low-ink bands along one image axis.

    Args:
        dark: Per-pixel darkness ``H x W``.
        axis: ``0`` for horizontal gutters (row cuts), ``1`` for vertical.
        thr_pct: Percentile threshold for low-ink columns/rows.

    Returns:
        List of ``(start, end)`` inclusive-exclusive low-ink segments.
    """
    profile = dark.mean(axis=axis)
    thr = float(np.percentile(profile, thr_pct))
    low = profile < thr
    segs: list[tuple[int, int]] = []
    idx = 0
    n_vals = len(low)
    while idx < n_vals:
        if not low[idx]:
            idx += 1
            continue
        start = idx
        while idx < n_vals and low[idx]:
            idx += 1
        segs.append((start, idx))
    return segs


def _content_bounds(im: Image.Image) -> tuple[int, int, int, int]:
    """Return tight content box ``(x0, y0, x1, y1)`` excluding near-white margins.

    Args:
        im: RGB image.

    Returns:
        Inclusive-exclusive content bounds.
    """
    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    dark = 255.0 - arr.mean(axis=2)
    rows = np.where(dark.mean(axis=1) > 2.0)[0]
    cols = np.where(dark.mean(axis=0) > 2.0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0, im.width, im.height
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _split_landscape_rows(im: Image.Image) -> tuple[list[tuple[int, int]], int, int]:
    """Split a landscape image into equal row bands and the col-1–2 x-range.

    Args:
        im: Landscape-only PNG (cols 1–2) or a full 4-col landscape trio.

    Returns:
        ``(row_bands, x_left, x_right)`` for the landscape half.
    """
    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    dark = 255.0 - arr.mean(axis=2)
    h_img, w_img = dark.shape
    x_left = 8
    if w_img < 2200:
        x_right = w_img - 4
    else:
        v_segs = [
            seg for seg in _gutter_cuts(dark, axis=0, thr_pct=20) if seg[1] - seg[0] > 8
        ]
        interior = [seg for seg in v_segs if seg[0] > 40 and seg[1] < w_img - 40]
        target = 0.45 * w_img
        if interior:
            best = min(interior, key=lambda seg: abs(0.5 * (seg[0] + seg[1]) - target))
            x_right = (best[0] + best[1]) // 2
        else:
            x_right = w_img // 2
    y0 = max(40, int(0.02 * h_img))
    y1 = h_img - 8
    row_h = (y1 - y0) // 3
    rows = [(y0 + i * row_h, y0 + (i + 1) * row_h) for i in range(3)]
    return rows, x_left, x_right


def _split_localization(
    im: Image.Image,
) -> tuple[list[tuple[int, int]], int, int, int, int]:
    """Split localization PNG into morph/profile columns (panels only).

    Args:
        im: Soft-localization representative figure.

    Returns:
        ``(row_bands, morph_x0, morph_x1, prof_x0, prof_x1)``.
    """
    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    dark = 255.0 - arr.mean(axis=2)
    h_img, _w_img = dark.shape
    x0, y0, x1, _y1 = _content_bounds(im)

    # Morph:profile width ratio in localize_soft_motion is 1.15:2.55 ≈ 31%:69%.
    # Prefer a wide gutter near that boundary instead of thin gaps inside panels.
    target = x0 + 0.31 * (x1 - x0)
    v_segs = [
        seg for seg in _gutter_cuts(dark, axis=0, thr_pct=22) if seg[1] - seg[0] > 25
    ]
    interior = [seg for seg in v_segs if seg[0] > x0 + 120 and seg[1] < x1 - 200]
    if interior:
        best = min(interior, key=lambda seg: abs(0.5 * (seg[0] + seg[1]) - target))
        split = (best[0] + best[1]) // 2
    else:
        split = int(round(target))

    # Drop the bottom strip that holds the source colorbar / labels.
    loc_top = max(y0, int(0.06 * h_img))
    loc_bot = int(0.88 * h_img)
    row_h = (loc_bot - loc_top) // 3
    rows = [(loc_top + i * row_h, loc_top + (i + 1) * row_h) for i in range(3)]
    return rows, x0, split, split, x1


def _fit_box(im: Image.Image, width: int, height: int) -> Image.Image:
    """Letterbox ``im`` into a fixed width×height canvas (centered).

    Args:
        im: Source crop.
        width: Target width.
        height: Target height.

    Returns:
        RGB canvas of size ``(width, height)``.
    """
    scale = min(width / im.width, height / im.height)
    new_w = max(1, int(im.width * scale))
    new_h = max(1, int(im.height * scale))
    resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return canvas


def _width_at_height(im: Image.Image, height: int) -> int:
    """Return width that preserves aspect ratio at a target height.

    Args:
        im: Source crop.
        height: Target height in pixels.

    Returns:
        Scaled width.
    """
    return max(1, int(im.width * height / im.height))


def _render_morph_colorbar(width: int, height: int = 90) -> Image.Image:
    """Draw a full-width horizontal morph colorbar with tick labels.

    Args:
        width: Target width in pixels (morph column width).
        height: Target height in pixels including labels.

    Returns:
        RGB image with the colorbar and tick labels.
    """
    # Sample matplotlib coolwarm directly (no figure export) for a smooth bar.
    from matplotlib import colormaps

    cmap = colormaps["coolwarm"]
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
        )
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    pad_x = max(8, width // 40)
    bar_top = 8
    bar_h = max(22, height // 3)
    bar_left = pad_x
    bar_right = width - pad_x
    bar_w = max(1, bar_right - bar_left)

    for col in range(bar_w):
        rgba = cmap(col / max(bar_w - 1, 1))
        color = (int(255 * rgba[0]), int(255 * rgba[1]), int(255 * rgba[2]))
        draw.line(
            [(bar_left + col, bar_top), (bar_left + col, bar_top + bar_h)],
            fill=color,
        )
    draw.rectangle(
        [bar_left, bar_top, bar_right - 1, bar_top + bar_h],
        outline=(40, 40, 40),
        width=1,
    )

    labels = [(0.0, "x⁻"), (0.5, "x₀"), (1.0, "x⁺")]
    tick_y0 = bar_top + bar_h
    for frac, text in labels:
        x_pos = bar_left + int(frac * (bar_w - 1))
        draw.line([(x_pos, tick_y0), (x_pos, tick_y0 + 5)], fill=(40, 40, 40), width=1)
        bbox = draw.textbbox((0, 0), text, font=font_small)
        tw = bbox[2] - bbox[0]
        draw.text(
            (x_pos - tw // 2, tick_y0 + 6), text, fill=(20, 20, 20), font=font_small
        )

    label = "x₀ + t α vₖ"
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, tick_y0 + 28), label, fill=(20, 20, 20), font=font)
    return canvas



def compose_trio(land_path: Path, loc_path: Path, out_path: Path, *, title: str) -> None:
    """Compose the merged Exp-2 figure into ``trio_3d_2d.png``.

    Args:
        land_path: Landscape-only PNG (cols 1–2).
        loc_path: Soft-localization representative PNG.
        out_path: Destination path (``trio_3d_2d.png``).
        title: Figure title drawn at the top.
    """
    im_land = Image.open(land_path).convert("RGB")
    im_loc = Image.open(loc_path).convert("RGB")

    rows_land, x_land_l, x_land_r = _split_landscape_rows(im_land)
    rows_loc, x_morph_l, x_morph_r, x_prof_l, x_prof_r = _split_localization(im_loc)

    row_h = 620
    gap_land_loc = 20
    gap_morph_prof = 16
    title_h = 58
    cbar_pad = 10
    pad_bottom = 14

    land_w = max(
        _width_at_height(im_land.crop((x_land_l, y0, x_land_r, y1)), row_h)
        for y0, y1 in rows_land
    )
    morph_w = max(
        _width_at_height(im_loc.crop((x_morph_l, y0, x_morph_r, y1)), row_h)
        for y0, y1 in rows_loc
    )
    prof_w = max(
        _width_at_height(im_loc.crop((x_prof_l, y0, x_prof_r, y1)), row_h)
        for y0, y1 in rows_loc
    )

    row_imgs: list[Image.Image] = []
    for (yl0, yl1), (yr0, yr1) in zip(rows_land, rows_loc, strict=True):
        left = _fit_box(im_land.crop((x_land_l, yl0, x_land_r, yl1)), land_w, row_h)
        morph = _fit_box(im_loc.crop((x_morph_l, yr0, x_morph_r, yr1)), morph_w, row_h)
        prof = _fit_box(im_loc.crop((x_prof_l, yr0, x_prof_r, yr1)), prof_w, row_h)
        width = land_w + gap_land_loc + morph_w + gap_morph_prof + prof_w
        row = Image.new("RGB", (width, row_h), (255, 255, 255))
        x_pos = 0
        row.paste(left, (x_pos, 0))
        x_pos += land_w + gap_land_loc
        row.paste(morph, (x_pos, 0))
        x_pos += morph_w + gap_morph_prof
        row.paste(prof, (x_pos, 0))
        row_imgs.append(row)

    row_w = row_imgs[0].width
    cbar = _render_morph_colorbar(morph_w, height=78)
    morph_x0 = land_w + gap_land_loc

    canvas_h = title_h + row_h * 3 + cbar_pad + cbar.height + pad_bottom
    canvas = Image.new("RGB", (row_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26
        )
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((row_w - text_w) // 2, 14), title, fill=(15, 23, 42), font=font)

    y_pos = title_h
    for row in row_imgs:
        canvas.paste(row, (0, y_pos))
        y_pos += row_h
    canvas.paste(cbar, (morph_x0, y_pos + cbar_pad))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"wrote {out_path} size={canvas.size}", flush=True)


def main() -> None:
    """Compose ``trio_3d_2d.png`` only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--title",
        default="Measured U_model landscape + soft-mode localization",
    )
    args = parser.parse_args()

    if not LAND_PATH.exists():
        raise FileNotFoundError(f"missing landscape figure: {LAND_PATH}")
    if not LOC_PATH.exists():
        raise FileNotFoundError(f"missing localization figure: {LOC_PATH}")

    compose_trio(LAND_PATH, LOC_PATH, OUT_PATH, title=args.title)


if __name__ == "__main__":
    main()
