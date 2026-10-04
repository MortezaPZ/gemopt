"""A single drawing cut into a round stone.

A repeating stone keeps the existing columnar search. A picture that does not
repeat is not a column count: its own regions become the facets, one motif,
in the orientation it was drawn.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geometry import Mesh

__all__ = ["facet_preview", "mesh_from_drawing"]


def _disc(shape: tuple[int, int]):
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    centre = (height - 1) / 2.0
    radius = min(height, width) * 0.46
    inside = np.hypot(xx - centre, yy - centre) < radius
    return inside, centre, radius


def _labels(rgb: np.ndarray):
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    inside, centre, radius = _disc(lum.shape)
    background = ndimage.uniform_filter(lum, size=5)
    lines = inside & ((background - lum) > 0.045)
    lines = ndimage.binary_closing(lines, iterations=1)
    lines = ndimage.binary_dilation(lines, iterations=1)
    labels, count = ndimage.label(inside & ~lines)
    return labels, count, centre, radius, inside


def _trace(mask: np.ndarray) -> list[tuple[int, int]]:
    ys, xs = np.nonzero(mask)
    if len(ys) < 8:
        return []
    start = (int(ys[0]), int(xs[0]))
    dirs = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
    height, width = mask.shape
    cy, cx = start
    direction = 6
    points = [(cy, cx)]
    limit = int(mask.sum()) * 4 + 8
    for _ in range(limit):
        stepped = False
        for turn in range(8):
            nd = (direction + turn) % 8
            ny, nx = cy + dirs[nd][0], cx + dirs[nd][1]
            if 0 <= ny < height and 0 <= nx < width and mask[ny, nx]:
                cy, cx = ny, nx
                points.append((cy, cx))
                direction = (nd + 6) % 8
                stepped = True
                break
        if not stepped or ((cy, cx) == start and len(points) > 4):
            break
    return points


def _simplify(points: list[tuple[int, int]], epsilon: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
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
    if far == 0:
        far = len(pts) // 2
    left = chain(pts[:far + 1])
    right = np.vstack([pts[far:], pts[:1]])
    chosen = [pts[i] for i in left[:-1]] + [right[i] for i in chain(right)[:-1]]
    return np.asarray(chosen, dtype=np.float64)


def _polygons(rgb: np.ndarray, min_area: int = 220):
    labels, count, centre, radius, _inside = _labels(rgb)
    found = []
    for lab in range(1, count + 1):
        mask = labels == lab
        area = int(mask.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        simple = _simplify(_trace(mask[y0:y1, x0:x1]), 2.2)
        if len(simple) < 3:
            continue
        xy = np.column_stack([
            (simple[:, 1] + x0 - centre) / radius,
            (centre - (simple[:, 0] + y0)) / radius,
        ])
        span = 0.5 * np.abs(np.sum(
            xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1)
        ))
        if span < 0.002:
            continue
        colour = rgb[mask].mean(axis=0)
        found.append((xy, colour))
    return found


def _brightness(poly: np.ndarray) -> float:
    centre = poly.mean(axis=0)
    rc = float(np.linalg.norm(centre)) + 1e-6
    direction = centre / rc
    normal = np.array([0.45 * direction[0], 0.45 * direction[1], 1.0])
    normal /= np.linalg.norm(normal)
    light = np.array([0.35, -0.25, 0.90])
    light /= np.linalg.norm(light)
    return float(np.clip(0.28 + 0.72 * (normal @ light), 0.15, 1.0))


def facet_preview(rgb: np.ndarray) -> np.ndarray:
    """The drawing as one faceted disc: same place, same facing, no tiled sector."""
    from PIL import Image, ImageDraw

    image = np.asarray(rgb, dtype=np.float64)
    height, width = image.shape[:2]
    _inside, centre, radius = _disc((height, width))
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    box = (centre - radius, centre - radius, centre + radius, centre + radius)
    draw.ellipse(box, fill=(214, 220, 226))
    for poly, colour in _polygons(image):
        level = _brightness(poly)
        fill = tuple(int(np.clip((0.72 * level + 0.28 * channel) * 255.0, 0, 255))
                     for channel in colour)
        points = [(float(x * radius + centre), float(centre - y * radius)) for x, y in poly]
        draw.polygon(points, fill=fill, outline=(24, 28, 36))
    out = np.asarray(canvas, dtype=np.float64) / 255.0
    yy, xx = np.mgrid[:height, :width]
    out[np.hypot(xx - centre, yy - centre) >= radius * 1.01] = 0.0
    return out


def mesh_from_drawing(rgb: np.ndarray, n_r: int = 36, n_theta: int = 144) -> Mesh:
    """One closed stone. Crown height follows the drawing's regions, not a column count."""
    image = np.asarray(rgb, dtype=np.float64)
    labels, _count, centre, radius, _inside = _labels(image)
    height, width = labels.shape
    angles = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    radii = np.linspace(1.0 / n_r, 1.0, n_r)
    verts = []
    index = np.zeros((n_r, n_theta), dtype=np.int32)
    for i, rho in enumerate(radii):
        for k, ang in enumerate(angles):
            x = float(rho * np.cos(ang))
            y = float(rho * np.sin(ang))
            px = int(np.clip(round(centre + x * radius), 0, width - 1))
            py = int(np.clip(round(centre - y * radius), 0, height - 1))
            lab = int(labels[py, px])
            # Neighbouring regions sit at different heights, so the motif is a cut.
            step = 0.0 if lab == 0 else 0.045 * ((lab * 17) % 5) / 4.0
            z = (0.14 + step) * (1.0 - rho * rho)
            if i == n_r - 1:
                z = 0.0
            index[i, k] = len(verts)
            verts.append((x, y, z))
    apex = len(verts)
    verts.append((0.0, 0.0, 0.14))
    culet = len(verts)
    verts.append((0.0, 0.0, -0.43))
    tris = []
    fids = []
    groups = []
    polygons = []

    def face(poly, group):
        fid = len(groups)
        groups.append(group)
        polygons.append(list(poly))
        for a in range(1, len(poly) - 1):
            tris.append((poly[0], poly[a], poly[a + 1]))
            fids.append(fid)

    for k in range(n_theta):
        nxt = (k + 1) % n_theta
        face([apex, index[0, k], index[0, nxt]], "motif")
    for i in range(n_r - 1):
        for k in range(n_theta):
            nxt = (k + 1) % n_theta
            face([index[i, k], index[i, nxt], index[i + 1, nxt], index[i + 1, k]], "motif")
    for k in range(n_theta):
        nxt = (k + 1) % n_theta
        face([index[n_r - 1, k], culet, index[n_r - 1, nxt]], "pavilion")
    mesh = Mesh(
        vertices=np.asarray(verts, dtype=np.float64),
        triangles=np.asarray(tris, dtype=np.int32),
        facet_id=np.asarray(fids, dtype=np.int32),
        facet_group=groups,
        polygons=polygons,
    )
    if mesh.volume() < 0.0:
        mesh.triangles = mesh.triangles[:, ::-1]
        mesh.polygons = [list(reversed(poly)) for poly in mesh.polygons]
    return mesh
