"""Top-view PNG of the radial gem pattern, drawn from :mod:`gemopt.pattern`.

The picture is a diagram, not a ray-traced render: the same kites the
MaxScript rebuilds.  Title text sits outside the circular mask so the
rotation-symmetry test can ignore it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pattern import (
    PatternLayout,
    halo_radii,
    layout_from_n,
    n_gon_vertices,
    radial_array,
)

__all__ = [
    "draw_pattern",
    "save_pattern_png",
    "pattern_rgb",
    "rotation_symmetry_error",
    "assert_n_fold_symmetric",
]


NAVY = (6, 16, 46)
WHITE = (255, 255, 255)
HALO = (160, 185, 220, 70)
LIGHT = (110, 195, 255)
DARK = (36, 88, 185)
NOTCH = (52, 96, 168)
TITLE_BAND = 0.10
"""Fraction of the image height reserved for the caption, above the pattern."""


def _facet_colours(n: int, i: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Same light/dark split on every kite so the picture is N-fold symmetric."""
    return LIGHT, DARK


def _to_px(xy: tuple[float, float], cx: float, cy: float, scale: float) -> tuple[float, float]:
    return (cx + xy[0] * scale, cy - xy[1] * scale)


def draw_pattern(
    n: int | PatternLayout,
    size: int = 900,
    title: str | None = None,
) -> Image.Image:
    layout = n if isinstance(n, PatternLayout) else layout_from_n(int(n))
    over = max(size * 3, 600)
    img = Image.new("RGBA", (over, over), NAVY + (255,))
    draw = ImageDraw.Draw(img, "RGBA")

    cx = over / 2.0
    cy = over / 2.0
    # Tips sit well inside the frame; halos are clipped if this goes above ~0.44.
    scale = over * 0.42

    px = lambda p: _to_px(p, cx, cy, scale)
    stroke = max(1.0, layout.stroke_width * scale)
    halo_w = max(1.0, layout.halo_stroke * scale)
    gon_w = max(1.0, layout.n_gon_stroke * scale)

    for radius in halo_radii(layout):
        x0, y0 = px((-radius, radius))
        x1, y1 = px((radius, -radius))
        box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        draw.ellipse(box, outline=HALO, width=max(1, int(round(halo_w))))

    wedges = radial_array(layout)
    for i, w in enumerate(wedges):
        nxt = wedges[(i + 1) % layout.n]
        # The V between two tips is part of the N-gon.  Leaving it navy made
        # N=16 a skeleton of needles; filling it keeps the star and a solid
        # crown at every fold count.
        draw.polygon([px(w.tip), px(w.valley_right), px(nxt.tip)], fill=NOTCH)
    for w in wedges:
        light, dark = _facet_colours(layout.n, w.index)
        draw.polygon([px(p) for p in w.left_poly], fill=light)
        draw.polygon([px(p) for p in w.right_poly], fill=dark)
        outline = [px(w.origin), px(w.valley_left), px(w.tip),
                   px(w.valley_right)]
        draw.line(outline + [outline[0]], fill=WHITE, width=max(1, int(round(stroke))))
        if layout.n <= 10:
            draw.line([px(w.origin), px(w.tip)], fill=WHITE,
                      width=max(1, int(round(stroke))))
        tx, ty = px(w.tip)
        cap = max(1.5, stroke * 0.55)
        draw.ellipse([tx - cap, ty - cap, tx + cap, ty + cap], fill=WHITE)

    gon = [px(v) for v in n_gon_vertices(layout)]
    draw.line(gon + [gon[0]], fill=WHITE, width=max(1, int(round(gon_w))))

    # PIL does not reshape Arabic, so a Persian caption comes out backwards.
    # Latin is the caption the rotation test and the PNG can both read.
    caption = title if title is not None else f"N = {layout.n}"
    font = ImageFont.load_default()
    for candidate in (
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ):
        if candidate.exists():
            try:
                font = ImageFont.truetype(str(candidate), max(22, over // 28))
                break
            except OSError:
                continue
    bbox = draw.textbbox((0, 0), caption, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((over - tw) / 2.0, int(over * 0.028)), caption, fill=WHITE, font=font)

    rgb = img.convert("RGB")
    if over != size:
        rgb = rgb.resize((size, size), Image.LANCZOS)
    return rgb


def save_pattern_png(path, n: int | PatternLayout, size: int = 900) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    draw_pattern(n, size=size).save(path)
    return path


def pattern_rgb(n: int | PatternLayout, size: int = 420) -> np.ndarray:
    """0..1 float RGB for the same picture the GUI writes as ``out_crown``."""
    arr = np.asarray(draw_pattern(n, size=size), dtype=np.float64)
    return arr / 255.0


def _pattern_mask(size: int) -> np.ndarray:
    """Pixels that belong to the diagram, not the caption strip."""
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = size / 2.0
    r = np.hypot(xx - cx, yy - cy)
    return (r <= size * 0.42) & (yy > size * TITLE_BAND * 0.55)


def rotation_symmetry_error(
    image: Image.Image | np.ndarray,
    n: int,
) -> float:
    """Mean absolute difference after a ``360/N`` rotation, on the pattern mask.

    ``0`` is perfect N-fold symmetry.  Caption pixels are excluded.
    """
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    else:
        arr = np.asarray(image, dtype=np.float64)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
    size = arr.shape[0]
    angle = 360.0 / int(n)
    pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    rotated = np.asarray(
        pil.rotate(angle, resample=Image.BICUBIC, center=(size / 2.0, size / 2.0)),
        dtype=np.float64,
    )
    mask = _pattern_mask(size)
    # shrink the mask so resampling at the rim does not inflate the score
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = size / 2.0
    r = np.hypot(xx - cx, yy - cy)
    mask = mask & (r < size * 0.38)
    diff = np.abs(arr - rotated)[mask]
    if diff.size == 0:
        return 1.0
    return float(diff.mean() / 255.0)


def assert_n_fold_symmetric(image: Image.Image | np.ndarray, n: int,
                            limit: float = 0.04) -> float:
    err = rotation_symmetry_error(image, n)
    if err > limit:
        raise AssertionError(
            f"pattern is not {n}-fold symmetric: mean abs error {err:.4f} > {limit}"
        )
    return err
