"""Turn a smooth motif into an origami crease drawing.

A polished outline is a curve. Origami is straight folds: a coarse polygon
and the triangles inside it. A drawing that is already broken into corners
is left as it was drawn.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["origami_pattern"]


def origami_pattern(rgb: np.ndarray) -> np.ndarray:
    """Replace a smooth filled motif with flat origami facets.

    The circle around the motif stays, so the drawing still seats on the
    table. Thin line-work that already has sharp corners is unchanged.
    """
    photo = np.asarray(rgb, dtype=np.float64)
    if photo.ndim == 2:
        photo = np.repeat(photo[:, :, None], 3, axis=2)
    photo = np.ascontiguousarray(photo[..., :3])
    if photo.size and float(photo.max()) > 1.5:
        photo = photo / 255.0
    from .design_output import design_circle

    cx, cy, radius = design_circle(photo)
    if radius < 8:
        return photo
    height, width = photo.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    lum = photo @ np.array([0.2126, 0.7152, 0.0722])
    inside = np.hypot(xx - cx, yy - cy) < radius * 0.90
    ink = inside & (lum < 0.88)
    ink = ndimage.binary_fill_holes(ink)
    ink = ndimage.binary_opening(ink, iterations=1)
    labels, count = ndimage.label(ink)
    if count == 0:
        return photo
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    lab = int(np.argmax(areas))
    mask = labels == lab
    disc = max(int(inside.sum()), 1)
    if int(mask.sum()) > 0.82 * disc:
        return photo
    contour = _walk(mask)
    if len(contour) < 12:
        return photo
    # A boundary that already turns in sharp corners is origami. A polished
    # curve turns only a little between samples, and that is what we replace.
    light = _simplify(contour, 1.0)
    if _median_turn(light) >= np.deg2rad(25.0):
        return photo
    poly = _simplify(contour, max(2.0, 0.045 * radius))
    if len(poly) > 22:
        poly = _simplify(contour, max(3.0, 0.08 * radius))
    if len(poly) < 4:
        return photo
    wipe = ndimage.binary_dilation(mask, iterations=2)
    drawn = _paint(photo, poly, wipe)
    return drawn


def _walk(mask: np.ndarray) -> np.ndarray:
    """Boundary of ``mask`` as ``(x, y)`` rows, one loop."""
    edge = mask & ~ndimage.binary_erosion(mask)
    ys, xs = np.nonzero(edge)
    if len(xs) < 8:
        return np.zeros((0, 2))
    start = int(np.argmin(ys * mask.shape[1] + xs))
    y, x = int(ys[start]), int(xs[start])
    dirs = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
    height, width = mask.shape
    points = [(x, y)]
    previous = 4
    for _ in range(int(edge.sum()) * 2 + 8):
        stepped = False
        for turn in range(8):
            nd = (previous + turn) % 8
            ny, nx = y + dirs[nd][0], x + dirs[nd][1]
            if 0 <= ny < height and 0 <= nx < width and edge[ny, nx]:
                y, x = ny, nx
                points.append((x, y))
                previous = (nd + 5) % 8
                stepped = True
                break
        if not stepped or (len(points) > 8 and (x, y) == points[0]):
            break
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def _simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return pts

    def chain(poly: np.ndarray) -> list[int]:
        def rec(i, j):
            if j <= i + 1:
                return [i]
            chord = poly[j] - poly[i]
            length = float(np.linalg.norm(chord)) or 1.0
            mid = poly[i + 1:j] - poly[i]
            dist = np.abs(mid[:, 0] * chord[1] - mid[:, 1] * chord[0]) / length
            k = int(np.argmax(dist)) + i + 1
            if float(dist[k - i - 1]) > epsilon:
                return rec(i, k) + rec(k, j)
            return [i]
        return rec(0, len(poly) - 1) + [len(poly) - 1]

    far = int(np.argmax(np.linalg.norm(pts - pts[0], axis=1)))
    if far in (0, len(pts) - 1):
        far = len(pts) // 2
    left = [pts[i] for i in chain(pts[:far + 1])[:-1]]
    right = np.vstack([pts[far:], pts[:1]])
    chosen = left + [right[i] for i in chain(right)[:-1]]
    return np.asarray(chosen, dtype=np.float64)


def _median_turn(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    turns = []
    count = len(poly)
    for i in range(count):
        a = poly[(i - 1) % count]
        b = poly[i]
        c = poly[(i + 1) % count]
        u = a - b
        v = c - b
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu < 1e-8 or nv < 1e-8:
            continue
        cos = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
        turns.append(abs(np.pi - np.arccos(cos)))
    if not turns:
        return 0.0
    return float(np.median(turns))


def _paint(photo: np.ndarray, poly: np.ndarray, wipe: np.ndarray) -> np.ndarray:
    from PIL import Image, ImageDraw

    cleared = photo.copy()
    cleared[wipe] = (0.97, 0.97, 0.97)
    base = Image.fromarray(np.clip(cleared * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(base)
    xy = [(float(x), float(y)) for x, y in poly]
    math_poly = np.column_stack([poly[:, 0], -poly[:, 1]])
    area = float(np.sum(
        math_poly[:, 0] * np.roll(math_poly[:, 1], -1)
        - math_poly[:, 1] * np.roll(math_poly[:, 0], -1)
    ))
    if area < 0.0:
        math_poly = math_poly[::-1]
        order = np.arange(len(poly))[::-1]
    else:
        order = np.arange(len(poly))
    triangles = _ears(math_poly)
    for n, tri in enumerate(triangles):
        shade = 0.62 + 0.28 * ((n * 3) % 5) / 4.0
        colour = (int(shade * 170), int(shade * 205), int(shade * 230))
        corners = [tuple(poly[int(order[i])]) for i in tri]
        draw.polygon(corners, fill=colour)
    crease = (45, 60, 78)
    loop = xy + xy[:1]
    draw.line(loop, fill=crease, width=2)
    for tri in triangles:
        corners = [tuple(poly[int(order[i])]) for i in tri]
        draw.line(corners + corners[:1], fill=crease, width=1)
    return np.asarray(base, dtype=np.float64) / 255.0


def _ears(poly: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear-clip ``poly`` (CCW, y up). Indices refer to the input rows."""
    count = len(poly)
    if count < 3:
        return []
    if count == 3:
        return [(0, 1, 2)]
    left = list(range(count))
    tris: list[tuple[int, int, int]] = []
    guard = 0
    while len(left) > 3 and guard < count * count:
        guard += 1
        clipped = False
        n = len(left)
        for i in range(n):
            ia, ib, ic = left[(i - 1) % n], left[i], left[(i + 1) % n]
            a, b, c = poly[ia], poly[ib], poly[ic]
            if _cross(a, b, c) <= 1e-8:
                continue
            if any(_inside(poly[p], a, b, c) for p in left if p not in (ia, ib, ic)):
                continue
            tris.append((ia, ib, ic))
            del left[i]
            clipped = True
            break
        if not clipped:
            break
    if len(left) == 3:
        tris.append((left[0], left[1], left[2]))
    return tris


def _cross(a, b, c) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _inside(p, a, b, c) -> bool:
    c1 = _cross(a, b, p)
    c2 = _cross(b, c, p)
    c3 = _cross(c, a, p)
    return c1 > 1e-6 and c2 > 1e-6 and c3 > 1e-6
