"""Second section: the employer's drawing, on a round stone, same proportions.

This is not the column search. The picture keeps its own direction and size.
A round stone is only the body the drawing sits on.
"""

from __future__ import annotations

import numpy as np

from .geometry import CutParams, build_stone

__all__ = [
    "design_circle",
    "girdle_radius",
    "pavilion_with_design",
    "round_stone",
    "table_half_width",
    "table_with_design",
    "seated_pattern",
    "stone_with_design",
]

_SHADE: dict[tuple[str, int], np.ndarray] = {}


def round_stone():
    return build_stone(CutParams(flat_girdle=True))


def girdle_radius(mesh) -> float:
    """Largest horizontal distance from the axis, in the mesh's own units."""
    xy = np.asarray(mesh.vertices[:, :2], dtype=np.float64)
    if len(xy) == 0:
        raise ValueError("the mesh has no vertices")
    return float(np.linalg.norm(xy, axis=1).max())


def _as_photo(rgb: np.ndarray) -> np.ndarray:
    photo = np.asarray(rgb, dtype=np.float64)
    if photo.ndim == 2:
        photo = np.repeat(photo[:, :, None], 3, axis=2)
    photo = np.ascontiguousarray(photo[..., :3])
    if photo.size and float(photo.max()) > 1.5:
        photo = photo / 255.0
    return photo


def _content_disc(photo: np.ndarray) -> tuple[float, float, float]:
    """Centre and radius of whatever is not the corner colour."""
    height, width = photo.shape[:2]
    corners = np.stack([photo[0, 0], photo[0, -1], photo[-1, 0], photo[-1, -1]])
    background = np.median(corners, axis=0)
    mask = np.linalg.norm(photo - background, axis=2) > 0.08
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return (width - 1) / 2.0, (height - 1) / 2.0, 0.5 * min(height, width)
    cx = (float(xs.min()) + float(xs.max())) / 2.0
    cy = (float(ys.min()) + float(ys.max())) / 2.0
    radius = max(float(ys.max() - ys.min()), float(xs.max() - xs.min())) / 2.0
    return cx, cy, max(radius, 1.0)


def design_circle(rgb: np.ndarray) -> tuple[float, float, float]:
    """Centre ``x``, centre ``y`` and radius of the drawing's circle, in pixels.

    ``y`` grows downward, the same way image rows do. The circle is the round
    outline of the design. A picture with no consistent circle falls back to
    the disc that bounds the ink.
    """
    photo = _as_photo(rgb)
    height, width = photo.shape[:2]
    if min(height, width) < 8:
        return _content_disc(photo)
    step = max(1, int(round(max(height, width) / 240.0)))
    small = photo[::step, ::step]
    lum = small @ np.array([0.2126, 0.7152, 0.0722])
    mag = np.hypot(*np.gradient(lum))
    hh, ww = mag.shape
    r_lo = max(5, int(0.10 * min(hh, ww)))
    r_hi = max(r_lo + 2, int(0.50 * min(hh, ww)) - 1)
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    ct, st = np.cos(theta), np.sin(theta)
    radii = np.unique(np.linspace(r_lo, r_hi, 14).astype(int))
    best_score = -1.0
    cx = ww / 2.0
    cy = hh / 2.0
    radius = float(radii[len(radii) // 2])
    for cy_i in range(r_lo, max(r_lo + 1, hh - r_lo), 7):
        for cx_i in range(r_lo, max(r_lo + 1, ww - r_lo), 7):
            limit = min(cx_i, cy_i, ww - 1 - cx_i, hh - 1 - cy_i, r_hi)
            if limit < r_lo:
                continue
            usable = radii[radii <= limit]
            if len(usable) == 0:
                continue
            rr = usable[:, None]
            xs = np.clip((cx_i + rr * ct).astype(int), 0, ww - 1)
            ys = np.clip((cy_i + rr * st).astype(int), 0, hh - 1)
            scores = mag[ys, xs].mean(axis=1)
            pick = int(np.argmax(scores))
            if float(scores[pick]) > best_score:
                best_score = float(scores[pick])
                cx, cy, radius = float(cx_i), float(cy_i), float(usable[pick])
    refined = (best_score, cx, cy, radius)
    for cy2 in np.arange(cy - 5.0, cy + 6.0):
        for cx2 in np.arange(cx - 5.0, cx + 6.0):
            for r2 in np.arange(max(r_lo, radius - 4.0), radius + 5.0):
                xs = cx2 + r2 * ct
                ys = cy2 + r2 * st
                if xs.min() < 0 or ys.min() < 0 or xs.max() > ww - 1 or ys.max() > hh - 1:
                    continue
                score = float(mag[ys.astype(int), xs.astype(int)].mean())
                if score > refined[0]:
                    refined = (score, float(cx2), float(cy2), float(r2))
    score, cx, cy, radius = refined
    peaks = []
    for angle in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
        band = np.arange(max(2, int(radius * 0.82)), int(radius * 1.18) + 1)
        if len(band) == 0:
            continue
        xs = np.clip((cx + band * np.cos(angle)).astype(int), 0, ww - 1)
        ys = np.clip((cy + band * np.sin(angle)).astype(int), 0, hh - 1)
        peaks.append(float(band[int(np.argmax(mag[ys, xs]))]))
    if score < 0.02 or not peaks:
        return _content_disc(photo)
    peaks_a = np.asarray(peaks, dtype=np.float64)
    if float(peaks_a.std() / max(float(peaks_a.mean()), 1.0)) > 0.08:
        return _content_disc(photo)
    cx *= step
    cy *= step
    radius *= step
    # The downsampled estimate is a couple of pixels wide. Lock it on the
    # full picture so the outline, not the margin beside it, is the radius.
    full = photo @ np.array([0.2126, 0.7152, 0.0722])
    full_mag = np.hypot(*np.gradient(full))
    locked = None
    span = float(step)
    for cy2 in np.linspace(cy - span, cy + span, 5):
        for cx2 in np.linspace(cx - span, cx + span, 5):
            for r2 in np.linspace(max(1.0, radius - span), radius + span, 5):
                xs = cx2 + r2 * ct
                ys = cy2 + r2 * st
                if xs.min() < -0.5 or ys.min() < -0.5 or xs.max() > width - 0.5 or ys.max() > height - 0.5:
                    continue
                xi = np.clip(np.rint(xs).astype(int), 0, width - 1)
                yi = np.clip(np.rint(ys).astype(int), 0, height - 1)
                got = float(full_mag[yi, xi].mean())
                if locked is None or got > locked[0]:
                    locked = (got, float(cx2), float(cy2), float(r2))
    if locked is None:
        return cx, cy, max(radius, 1.0)
    return locked[1], locked[2], max(locked[3], 1.0)


def seated_pattern(rgb: np.ndarray, size: int | None = None) -> np.ndarray:
    """Square picture with the drawing's circle centred on the outer pixel.

    A planar map whose half-width is the girdle radius then puts that circle
    on the girdle: texture ``u = 1`` is the last pixel, and the last pixel is
    the circle. The motif stays where it was relative to the circle, and the
    top of the picture stays the top.
    """
    from PIL import Image

    photo = _as_photo(rgb)
    cx, cy, radius = design_circle(photo)
    radius = max(float(radius), 1.0)
    if size is None:
        size = max(32, int(round(2.0 * radius)))
    else:
        size = max(32, int(size))
    centre = (size - 1) / 2.0
    scale = (2.0 * radius) / (size - 1)
    corners = np.stack([photo[0, 0], photo[0, -1], photo[-1, 0], photo[-1, -1]])
    fill = tuple(int(round(float(v) * 255.0)) for v in np.median(corners, axis=0))
    image = Image.fromarray(np.clip(photo * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    warped = image.transform(
        (size, size),
        Image.AFFINE,
        (scale, 0.0, cx - scale * centre, 0.0, scale, cy - scale * centre),
        resample=Image.Resampling.BILINEAR,
        fillcolor=fill,
    )
    return np.asarray(warped, dtype=np.float64) / 255.0


def _shaded_round(size: int) -> np.ndarray:
    cached = _SHADE.get(("crown", size))
    if cached is not None:
        return cached
    mesh = round_stone()
    tv = mesh.triangle_vertices()
    normals = mesh.normals()
    image = np.zeros((size, size, 3), dtype=np.float64)
    light = np.array([0.15, -0.35, 0.92])
    light /= np.linalg.norm(light)
    half = light + np.array([0.0, 0.0, 1.0])
    half /= np.linalg.norm(half)
    order = np.argsort(tv[:, :, 2].mean(axis=1))
    xs = (tv[:, :, 0] * 0.48 + 0.5) * (size - 1)
    ys = (0.5 - tv[:, :, 1] * 0.48) * (size - 1)
    for tri in order:
        normal = normals[tri]
        if normal[2] <= 0.05:
            continue
        lit = float(np.clip(normal @ light, 0.0, 1.0))
        spec = float(np.clip(normal @ half, 0.0, 1.0)) ** 50
        colour = np.clip(0.22 + 0.62 * lit + 0.90 * spec, 0.0, 1.0)
        colour = colour * np.array([0.90, 0.94, 0.98])
        tri_x, tri_y = xs[tri], ys[tri]
        minx = max(int(np.floor(tri_x.min())), 0)
        maxx = min(int(np.ceil(tri_x.max())), size - 1)
        miny = max(int(np.floor(tri_y.min())), 0)
        maxy = min(int(np.ceil(tri_y.max())), size - 1)
        if maxx < minx or maxy < miny:
            continue
        px, py = np.meshgrid(np.arange(minx, maxx + 1), np.arange(miny, maxy + 1))
        x0, y0 = tri_x[0], tri_y[0]
        x1, y1 = tri_x[1], tri_y[1]
        x2, y2 = tri_x[2], tri_y[2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-8:
            continue
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
        w2 = 1.0 - w0 - w1
        hit = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        image[miny:maxy + 1, minx:maxx + 1][hit] = colour
    _SHADE[("crown", size)] = image
    return image


def _shaded_pavilion(size: int) -> np.ndarray:
    """The lower cone, seen from below.  The disc is the girdle."""
    cached = _SHADE.get(("pavilion", size))
    if cached is not None:
        return cached
    mesh = round_stone()
    tv = mesh.triangle_vertices()
    normals = mesh.normals()
    image = np.zeros((size, size, 3), dtype=np.float64)
    light = np.array([0.15, -0.35, -0.92])
    light /= np.linalg.norm(light)
    half = light + np.array([0.0, 0.0, -1.0])
    half /= np.linalg.norm(half)
    # The culet is closest to a camera under the stone, so it paints last.
    order = np.argsort(-tv[:, :, 2].mean(axis=1))
    xs = (tv[:, :, 0] * 0.48 + 0.5) * (size - 1)
    ys = (0.5 - tv[:, :, 1] * 0.48) * (size - 1)
    for tri in order:
        normal = normals[tri]
        if normal[2] >= -0.05:
            continue
        lit = float(np.clip(normal @ light, 0.0, 1.0))
        spec = float(np.clip(normal @ half, 0.0, 1.0)) ** 50
        colour = np.clip(0.22 + 0.62 * lit + 0.90 * spec, 0.0, 1.0)
        colour = colour * np.array([0.90, 0.94, 0.98])
        tri_x, tri_y = xs[tri], ys[tri]
        minx = max(int(np.floor(tri_x.min())), 0)
        maxx = min(int(np.ceil(tri_x.max())), size - 1)
        miny = max(int(np.floor(tri_y.min())), 0)
        maxy = min(int(np.ceil(tri_y.max())), size - 1)
        if maxx < minx or maxy < miny:
            continue
        px, py = np.meshgrid(np.arange(minx, maxx + 1), np.arange(miny, maxy + 1))
        x0, y0 = tri_x[0], tri_y[0]
        x1, y1 = tri_x[1], tri_y[1]
        x2, y2 = tri_x[2], tri_y[2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-8:
            continue
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
        w2 = 1.0 - w0 - w1
        hit = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        image[miny:maxy + 1, minx:maxx + 1][hit] = colour
    _SHADE[("pavilion", size)] = image
    return image


def stone_with_design(rgb: np.ndarray) -> np.ndarray:
    """Put ``rgb`` on a round stone without tiling it around the circle.

    The drawing's own disc lands on the stone's disc, so direction and
    proportion stay those of the input.
    """
    photo = np.asarray(rgb, dtype=np.float64)
    if photo.ndim == 2:
        photo = np.repeat(photo[:, :, None], 3, axis=2)
    photo = photo[..., :3]
    if photo.max() > 1.5:
        photo = photo / 255.0
    height, width = photo.shape[:2]
    gem = _shaded_round(max(height, width))
    if gem.shape[0] != height or gem.shape[1] != width:
        from PIL import Image
        gem = np.asarray(
            Image.fromarray(np.clip(gem * 255.0, 0, 255).astype(np.uint8)).resize(
                (width, height), Image.Resampling.BILINEAR),
            dtype=np.float64,
        ) / 255.0

    cx, cy, photo_r = design_circle(photo)
    gem_r = min(height, width) * 0.48
    scale = gem_r / max(photo_r, 1.0)

    yy, xx = np.mgrid[:height, :width]
    gx = (width - 1) / 2.0
    gy = (height - 1) / 2.0
    src_x = (xx - gx) / scale + cx
    src_y = (yy - gy) / scale + cy
    ix = np.clip(np.rint(src_x).astype(int), 0, width - 1)
    iy = np.clip(np.rint(src_y).astype(int), 0, height - 1)
    src = photo[iy, ix]
    inside = np.hypot(src_x - cx, src_y - cy) < photo_r
    lum = src @ np.array([0.2126, 0.7152, 0.0722])
    # Blank paper stays the stone. The drawing itself — wings, body, lines — is the ink.
    paper = lum > 0.90
    ink = inside & ~paper & (lum > 0.03)
    out = gem.copy()
    out[ink] = np.clip(gem[ink] * 0.35 + src[ink] * 0.85, 0.0, 1.0)
    return out


def table_half_width(mesh, scale: float) -> float:
    """Picture half-width that puts the drawing's circle on the table edge.

    The circle's area is then the table's area, and not more.  ``scale`` is
    the girdle radius of the mesh written into the ``.gem`` file.
    """
    return girdle_radius(mesh) * float(scale) * float(CutParams().table)


def pavilion_with_design(rgb: np.ndarray) -> np.ndarray:
    """The drawing on the pavilion, inside the table, angles left alone.

    The lower cone keeps the standard cut.  The drawing's circle is scaled
    to the table, so the lines show where they fall and the cap they form
    is no larger than the table.
    """
    from PIL import Image

    photo = _as_photo(rgb)
    height, width = photo.shape[:2]
    gem = _shaded_pavilion(max(height, width))
    if gem.shape[0] != height or gem.shape[1] != width:
        gem = np.asarray(
            Image.fromarray(np.clip(gem * 255.0, 0, 255).astype(np.uint8)).resize(
                (width, height), Image.Resampling.BILINEAR),
            dtype=np.float64,
        ) / 255.0
    cx, cy, photo_r = design_circle(photo)
    gem_r = min(height, width) * 0.48
    cap_r = gem_r * float(CutParams().table)
    scale = cap_r / max(photo_r, 1.0)
    yy, xx = np.mgrid[:height, :width]
    gx = (width - 1) / 2.0
    gy = (height - 1) / 2.0
    src_x = (xx - gx) / scale + cx
    src_y = (yy - gy) / scale + cy
    ix = np.clip(np.rint(src_x).astype(int), 0, width - 1)
    iy = np.clip(np.rint(src_y).astype(int), 0, height - 1)
    src = photo[iy, ix]
    on_cap = np.hypot(xx - gx, yy - gy) <= cap_r + 0.5
    inside = np.hypot(src_x - cx, src_y - cy) < photo_r
    lum = src @ np.array([0.2126, 0.7152, 0.0722])
    paper = lum > 0.90
    ink = on_cap & inside & ~paper & (lum > 0.03)
    out = gem.copy()
    out[ink] = np.clip(gem[ink] * 0.35 + src[ink] * 0.85, 0.0, 1.0)
    return out


def _table_polygon(width: int, height: int) -> np.ndarray:
    """Table corners in image pixels. World +Y is up, row 0 is the top."""
    table = float(CutParams().table)
    n = int(CutParams().n_base)
    reach = 0.48 * (min(width, height) - 1) * table
    angles = 2.0 * np.pi * np.arange(n) / n
    return np.column_stack([
        (width - 1) / 2.0 + reach * np.cos(angles),
        (height - 1) / 2.0 - reach * np.sin(angles),
    ])


def _inside_polygon(xx, yy, polygon: np.ndarray) -> np.ndarray:
    inside = np.zeros(xx.shape, dtype=bool)
    count = len(polygon)
    j = count - 1
    for i in range(count):
        yi, yj = polygon[i, 1], polygon[j, 1]
        xi, xj = polygon[i, 0], polygon[j, 0]
        cross = ((yi > yy) != (yj > yy)) & (
            xx < (xj - xi) * (yy - yi) / (yj - yi + 1e-12) + xi
        )
        inside ^= cross
        j = i
    return inside


def table_with_design(rgb: np.ndarray) -> np.ndarray:
    """The drawing on the table, seen from above, and not past that face.

    Star and bezel pixels stay the plain crown. The picture stops on the
    table polygon, so a round bitmap cannot bulge out through the edges.
    """
    from PIL import Image

    photo = _as_photo(rgb)
    height, width = photo.shape[:2]
    gem = _shaded_round(max(height, width))
    if gem.shape[0] != height or gem.shape[1] != width:
        gem = np.asarray(
            Image.fromarray(np.clip(gem * 255.0, 0, 255).astype(np.uint8)).resize(
                (width, height), Image.Resampling.BILINEAR),
            dtype=np.float64,
        ) / 255.0
    cx, cy, photo_r = design_circle(photo)
    centre_x = (width - 1) / 2.0
    centre_y = (height - 1) / 2.0
    table_r = 0.48 * (min(height, width) - 1) * float(CutParams().table)
    scale = table_r / max(photo_r, 1.0)
    yy, xx = np.mgrid[:height, :width]
    src_x = (xx - centre_x) / scale + cx
    src_y = (yy - centre_y) / scale + cy
    ix = np.clip(np.rint(src_x).astype(int), 0, width - 1)
    iy = np.clip(np.rint(src_y).astype(int), 0, height - 1)
    src = photo[iy, ix]
    on_table = _inside_polygon(xx, yy, _table_polygon(width, height))
    inside = np.hypot(src_x - cx, src_y - cy) < photo_r
    lum = src @ np.array([0.2126, 0.7152, 0.0722])
    paper = lum > 0.90
    ink = on_table & inside & ~paper & (lum > 0.03)
    out = gem.copy()
    out[ink] = np.clip(gem[ink] * 0.35 + src[ink] * 0.85, 0.0, 1.0)
    return out
