"""Top-view PNG of the radial gem pattern, drawn from :mod:`gemopt.pattern`.

The picture is the same crown map the MaxScript rebuilds: table N-gon, N
spokes from the table vertices, N outer kites.  Palette matches the 8-fold
optical still the client treats as the good picture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pattern import (
    PatternLayout,
    layout_from_n,
    n_gon_vertices,
    radial_array,
    table_vertices,
)

__all__ = [
    "draw_pattern",
    "save_pattern_png",
    "pattern_rgb",
    "rotation_symmetry_error",
    "assert_n_fold_symmetric",
    "homologous_optics",
]


# Earth tones taken from the accepted N=8 optical still.
BG = (0, 0, 0)
TABLE = (42, 24, 16)
BEIGE = (214, 176, 138)
BEIGE_D = (168, 118, 82)
FILL = (118, 52, 36)
FILL_D = (78, 32, 24)
STROKE = (28, 16, 12)
TITLE = (230, 210, 190)
TITLE_BAND = 0.10


def _to_px(xy: tuple[float, float], cx: float, cy: float, scale: float) -> tuple[float, float]:
    return (cx + xy[0] * scale, cy - xy[1] * scale)


def draw_pattern(
    n: int | PatternLayout,
    size: int = 900,
    title: str | None = None,
) -> Image.Image:
    layout = n if isinstance(n, PatternLayout) else layout_from_n(int(n))
    over = max(size * 3, 600)
    img = Image.new("RGBA", (over, over), BG + (255,))
    draw = ImageDraw.Draw(img, "RGBA")

    cx = over / 2.0
    cy = over / 2.0
    scale = over * 0.42

    px = lambda p: _to_px(p, cx, cy, scale)
    stroke = max(1.0, layout.stroke_width * scale)
    gon_w = max(1.0, layout.n_gon_stroke * scale)

    wedges = radial_array(layout)
    for i, w in enumerate(wedges):
        nxt = wedges[(i + 1) % layout.n]
        draw.polygon([px(w.tip), px(w.valley_right), px(nxt.tip)], fill=FILL)
        draw.polygon(
            [px(w.valley_right), px(w.table), px(nxt.table)],
            fill=FILL_D,
        )
    for w in wedges:
        draw.polygon([px(p) for p in w.left_poly], fill=BEIGE)
        draw.polygon([px(p) for p in w.right_poly], fill=BEIGE_D)
        outline = [px(w.table), px(w.valley_left), px(w.tip), px(w.valley_right)]
        draw.line(outline + [outline[0]], fill=STROKE, width=max(1, int(round(stroke))))

    table = [px(v) for v in table_vertices(layout)]
    draw.polygon(table, fill=TABLE)
    draw.line(table + [table[0]], fill=STROKE, width=max(1, int(round(stroke))))

    gon = [px(v) for v in n_gon_vertices(layout)]
    draw.line(gon + [gon[0]], fill=STROKE, width=max(1, int(round(gon_w))))

    caption = title if title is not None else f"N = {layout.n}"
    if caption:
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
        draw.text(((over - tw) / 2.0, int(over * 0.028)), caption, fill=TITLE, font=font)

    rgb = img.convert("RGB")
    if over != size:
        rgb = rgb.resize((size, size), Image.LANCZOS)
    return rgb


def save_pattern_png(path, n: int | PatternLayout, size: int = 900) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    draw_pattern(n, size=size).save(path)
    return path


def pattern_rgb(n: int | PatternLayout, size: int = 420,
                title: str | None = None) -> np.ndarray:
    """0..1 float RGB for the same picture the GUI writes as ``out_crown``."""
    arr = np.asarray(draw_pattern(n, size=size, title=title), dtype=np.float64)
    return arr / 255.0


def homologous_optics(image8: np.ndarray, dest_n: int, source_n: int = 8) -> np.ndarray:
    """Design preview from a fitted source render, with only N changed.

    This is an angular remapping, not a ray trace of the destination mesh.
    Colours and motif geometry must come from the supplied image.
    """
    from .fold import remap_image_fold

    return remap_image_fold(image8, source_n, dest_n)


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
    """Mean absolute difference after a ``360/N`` rotation, on the pattern mask."""
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
