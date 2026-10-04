"""Map a pattern picture onto a stone in two deterministic ways.

Both modes share one frame.  The picture is centred on the stone's origin
and covers the square ``[-S, S]`` in the horizontal plane.  ``S`` is the
half-width of that square.  The picture the program itself draws is framed
with :attr:`gemopt.scope.Camera.extent` (1.02 girdle radii); pass that, in
the mesh's own units, when those files are the picture.  Left unset, ``S``
is the mesh's maximum horizontal radius.

Row 0 of an image array is the top of the picture and world ``+Y``.  Texture
``v`` follows 3ds Max: 0 at the bottom of the bitmap, 1 at the top.  A render
is already stored with its top row at the top of the file, so it is used as
it was written.

planar
    The picture's horizontal coordinates are the surface point's.  Slope
    does not move the design in plan, so an orthographic view from above
    reproduces the picture.  On a steep face the same picture is packed into
    a short plan distance and gathers along the surface.

surface
    A vertical plane through the axis, one per azimuth, cuts the profile.
    Arc length from the axis out to the crown's plan silhouette replaces the
    plan radius, so relief stretches the picture when seen from above.  Past
    that silhouette the arc length keeps growing and the coordinates run off
    the edge of the picture, which puts the pattern on the pavilion too.

The single vertex that sits on the axis (a pointed culet) has no azimuth, so
it stays at the centre of the picture.  Every other vertex carries its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Mesh
from .optics import Geometry

__all__ = [
    "TopView",
    "frame_half_width",
    "cap_face_uv",
    "pavilion_uv",
    "planar_uv",
    "surface_uv",
    "top_view",
]

# A girdle is a few hundredths of a radius thick.  This is far smaller than
# that and far larger than coordinate noise, so cuts through a vertex stay
# on the vertex without welding two real ones together.
_TOL = 1e-8
_KEY = 8


@dataclass(frozen=True)
class TopView:
    """Orthographic picture from above.

    ``hit`` marks pixels whose ray met the stone and whose texture coordinate
    falls inside the picture.
    """

    image: np.ndarray
    hit: np.ndarray


def frame_half_width(mesh: Mesh, half_width: float | None = None) -> float:
    """``S`` for this mesh: the given half-width, or the plan radius."""
    if half_width is None:
        xy = np.asarray(mesh.vertices[:, :2], dtype=np.float64)
        if len(xy) == 0:
            raise ValueError("the mesh has no vertices")
        half_width = float(np.linalg.norm(xy, axis=1).max())
    half_width = float(half_width)
    if not np.isfinite(half_width) or half_width <= 0.0:
        raise ValueError(f"half-width must be positive, got {half_width}")
    return half_width


def planar_uv(mesh: Mesh, half_width: float | None = None) -> np.ndarray:
    """``(V, 2)`` texture coordinates.  Horizontal position is the picture."""
    s = frame_half_width(mesh, half_width)
    return _xy_to_uv(np.asarray(mesh.vertices[:, :2], dtype=np.float64), s)


def cap_face_uv(mesh: Mesh, half_width: float | None = None) -> list[np.ndarray]:
    """Per-facet coordinates so the picture stays on one face.

    The table, seen from above, and the pavilion, seen from below, carry the
    picture.  Every other facet gets coordinates outside the bitmap, and those
    coordinates are not shared with the table, so the picture cannot bleed
    past the table's edge.
    """
    s = frame_half_width(mesh, half_width)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    z = verts[:, 2]
    zmax = float(z.max()) if len(z) else 0.0
    faces: list[np.ndarray] = []
    for poly in mesh.polygons:
        idx = np.asarray(poly, dtype=np.int64)
        zz = z[idx]
        on_table = bool(np.all(np.abs(zz - zmax) <= 1e-4))
        on_pavilion = bool(np.all(zz < -1e-6))
        if on_table or on_pavilion:
            faces.append(_xy_to_uv(verts[idx, :2], s))
        else:
            faces.append(np.full((len(idx), 2), -1.0))
    return faces


def pavilion_uv(mesh: Mesh, half_width: float | None = None) -> np.ndarray:
    """Planar coordinates on the lower side only.

    Vertices at or above the girdle stay outside the picture, so the drawing
    cannot land on the crown.  ``half_width`` is the table radius when the
    drawing's circle must not exceed the table.
    """
    s = frame_half_width(mesh, half_width)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    uv = _xy_to_uv(verts[:, :2], s)
    uv[verts[:, 2] >= -1e-6] = (-1.0, -1.0)
    return uv


def surface_uv(mesh: Mesh, half_width: float | None = None) -> np.ndarray:
    """``(V, 2)`` texture coordinates measured along the surface.

    In each azimuth the picture radius is ``(σ / σ_edge) * ρ_edge``, where
    ``σ`` is arc length from the axis and the edge is the first point of the
    profile at maximum plan radius -- the top of the girdle.
    """
    s = frame_half_width(mesh, half_width)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    uv = np.empty((len(verts), 2), dtype=np.float64)
    cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    rho = np.linalg.norm(verts[:, :2], axis=1)
    for i, vertex in enumerate(verts):
        if rho[i] <= _TOL:
            uv[i] = (0.5, 0.5)
            continue
        phi = round(math.atan2(float(vertex[1]), float(vertex[0])), _KEY)
        path = cache.get(phi)
        if path is None:
            path = _meridian(verts, tris, phi)
            cache[phi] = path
        sigma, _ = _sigma_at(path[0], path[1], vertex)
        image_r = (sigma / path[2]) * path[3]
        image_xy = image_r * vertex[:2] / rho[i]
        uv[i] = _xy_to_uv(image_xy, s)
    return uv


def top_view(
    mesh: Mesh,
    uv: np.ndarray,
    image,
    half_width: float | None = None,
) -> TopView:
    """Rasterise ``uv`` from above into the same frame as the picture.

    One ray per pixel, straight down.  The closest hit is the surface seen
    from above.  Its texture coordinate is the barycentric mix of the three
    vertices, which is what a mesh map channel draws, and the picture is
    read there with nearest-neighbour sampling.
    """
    picture = _as_picture(image)
    height, width = picture.shape[:2]
    s = frame_half_width(mesh, half_width)
    uv = np.asarray(uv, dtype=np.float64)
    if uv.shape != (len(mesh.vertices), 2):
        raise ValueError(
            f"uv must be {(len(mesh.vertices), 2)}, got {uv.shape}")

    col = (np.arange(width) + 0.5) / width
    row = (np.arange(height) + 0.5) / height
    xs = (col * 2.0 - 1.0) * s
    ys = (1.0 - row * 2.0) * s
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    n = height * width
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()
    z_above = float(mesh.vertices[:, 2].max()) + 1.0
    origin = np.column_stack((flat_x, flat_y, np.full(n, z_above)))
    direction = np.zeros((n, 3), dtype=np.float64)
    direction[:, 2] = -1.0

    geom = Geometry(mesh)
    _, tri = geom.intersect(origin, direction)
    hit = tri >= 0
    out = np.zeros((height, width, picture.shape[2]), dtype=np.float64)
    if not np.any(hit):
        return TopView(out, hit.reshape(height, width))

    chosen = tri[hit]
    tv = mesh.triangle_vertices()
    normals = mesh.normals()
    nrm = normals[chosen]
    denom = nrm[:, 2] * -1.0
    usable = np.abs(denom) > _TOL
    plane_d = np.einsum("ij,ij->i", normals, tv[:, 0])
    t = np.zeros(int(hit.sum()), dtype=np.float64)
    origins = origin[hit][usable]
    t[usable] = (
        plane_d[chosen[usable]] - np.einsum("ij,ij->i", nrm[usable], origins)
    ) / denom[usable]
    point = origin[hit].copy()
    point[usable, 2] = z_above - t[usable]

    corners = tv[chosen]
    bary = _barycentric(point, corners[:, 0], corners[:, 1], corners[:, 2])
    mapped = np.einsum("ij,ijk->ik", bary, uv[mesh.triangles[chosen]])
    keep = usable & _inside_picture(mapped, width, height)
    if np.any(keep):
        rows, cols = _nearest_pixel(mapped[keep], width, height)
        linear = np.flatnonzero(hit)
        chosen_linear = linear[keep]
        out.reshape(n, -1)[chosen_linear] = picture[rows, cols]
        hit = np.zeros(n, dtype=bool)
        hit[chosen_linear] = True
    else:
        hit = np.zeros(n, dtype=bool)
    return TopView(out, hit.reshape(height, width))


def _xy_to_uv(xy: np.ndarray, half_width: float) -> np.ndarray:
    """World horizontal position to a bitmap coordinate, ``v`` upward."""
    xy = np.asarray(xy, dtype=np.float64)
    single = xy.ndim == 1
    if single:
        xy = xy.reshape(1, 2)
    uv = np.empty_like(xy)
    uv[:, 0] = 0.5 + xy[:, 0] / (2.0 * half_width)
    uv[:, 1] = 0.5 + xy[:, 1] / (2.0 * half_width)
    return uv[0] if single else uv


def _as_picture(image) -> np.ndarray:
    picture = np.asarray(image)
    if picture.ndim == 2:
        picture = np.repeat(picture[:, :, None], 3, axis=2)
    if picture.ndim != 3 or picture.shape[2] < 3:
        raise ValueError(f"image must be HxW or HxWx3, got {picture.shape}")
    if np.issubdtype(picture.dtype, np.integer):
        picture = picture.astype(np.float64) / 255.0
    else:
        picture = picture.astype(np.float64)
    return picture[:, :, :3]


def _inside_picture(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    col = uv[:, 0] * width - 0.5
    row = (1.0 - uv[:, 1]) * height - 0.5
    return (
        (col >= -0.5) & (col < width - 0.5)
        & (row >= -0.5) & (row < height - 0.5)
    )


def _nearest_pixel(uv: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    col = np.rint(uv[:, 0] * width - 0.5).astype(np.int64)
    row = np.rint((1.0 - uv[:, 1]) * height - 0.5).astype(np.int64)
    return (
        np.clip(row, 0, height - 1),
        np.clip(col, 0, width - 1),
    )


def _barycentric(point, v0, v1, v2) -> np.ndarray:
    e1 = v1 - v0
    e2 = v2 - v0
    ep = point - v0
    d00 = np.einsum("ij,ij->i", e1, e1)
    d01 = np.einsum("ij,ij->i", e1, e2)
    d11 = np.einsum("ij,ij->i", e2, e2)
    d20 = np.einsum("ij,ij->i", ep, e1)
    d21 = np.einsum("ij,ij->i", ep, e2)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-18, 1.0, den)
    b1 = (d11 * d20 - d01 * d21) / den
    b2 = (d00 * d21 - d01 * d20) / den
    b0 = 1.0 - b1 - b2
    return np.column_stack((b0, b1, b2))


def _meridian(
    verts: np.ndarray, tris: np.ndarray, phi: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Profile in one azimuth, from the crown axis down to the culet.

    Returns ``(points, arc length at each point, σ at the crown edge,
    plan radius of that edge)``.
    """
    direction = np.array([math.cos(phi), math.sin(phi), 0.0])
    normal = np.array([-direction[1], direction[0], 0.0])
    # A corner cut twice, or read back from a .gem rounded to 1e-5, can land
    # a hair apart and split one profile into two.  Welding those copies
    # keeps the chain.  The tolerance stays far below a thin girdle.
    weld = max(1e-6, 1e-5 * _extent(verts))
    points: dict[int, np.ndarray] = {}
    adj: dict[int, set] = {}
    nxt_id = 0

    def add_point(p: np.ndarray) -> int:
        nonlocal nxt_id
        p = np.asarray(p, dtype=np.float64)
        for key, got in points.items():
            if float(np.linalg.norm(got - p)) <= weld:
                return key
        key = nxt_id
        nxt_id += 1
        points[key] = p.copy()
        return key

    def add_segment(p: np.ndarray, q: np.ndarray) -> None:
        clipped = _clip_forward(p, q, direction)
        if clipped is None:
            return
        a = add_point(clipped[0])
        b = add_point(clipped[1])
        if a == b:
            return
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    for tri in tris:
        _cut_triangle(verts[tri], normal, add_segment, weld)

    if not points:
        raise ValueError("radial cut missed the mesh")

    axis = [key for key, p in points.items() if float(np.hypot(p[0], p[1])) <= _TOL]
    if not axis:
        raise ValueError("radial cut never reached the axis")
    start = max(axis, key=lambda key: float(points[key][2]))
    path_keys = _profile(start, adj, points)
    path = np.vstack([points[key] for key in path_keys])
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    sigma = np.concatenate(([0.0], np.cumsum(step)))
    radius = np.linalg.norm(path[:, :2], axis=1)
    # Top and bottom of a vertical girdle share a radius.  Rounded
    # coordinates can leave the lower one a hair larger, and an absolute
    # 1e-8 then treats only the lower one as the crown edge.
    edge = int(np.argmax(radius >= radius.max() - weld))
    if sigma[edge] <= _TOL:
        raise ValueError("crown edge has no arc length")
    return path, sigma, float(sigma[edge]), float(radius[edge])


def _extent(verts: np.ndarray) -> float:
    """Largest absolute coordinate, at least 1, so a unit stone stays put."""
    if len(verts) == 0:
        return 1.0
    return max(1.0, float(np.max(np.abs(verts))))


def _cut_triangle(corners: np.ndarray, normal: np.ndarray, add_segment, weld: float) -> None:
    """Add the segment where one triangle meets the radial plane."""
    signed = corners @ normal
    side = np.zeros(3, dtype=np.int8)
    side[signed > _TOL] = 1
    side[signed < -_TOL] = -1
    if np.all(side == 0):
        return

    hits: list[np.ndarray] = []
    for i in range(3):
        j = (i + 1) % 3
        if side[i] == 0 and side[j] == 0:
            hits.append(corners[i])
            hits.append(corners[j])
        elif side[i] == 0 and side[j] != 0:
            hits.append(corners[i])
        elif side[i] != 0 and side[j] != 0 and side[i] != side[j]:
            t = signed[i] / (signed[i] - signed[j])
            hits.append(corners[i] + t * (corners[j] - corners[i]))

    unique: list[np.ndarray] = []
    for hit in hits:
        if not any(np.linalg.norm(hit - got) <= weld for got in unique):
            unique.append(hit)
    if len(unique) >= 2:
        add_segment(unique[0], unique[1])


def _clip_forward(p: np.ndarray, q: np.ndarray, direction: np.ndarray):
    """Keep the part of a segment with plan coordinate ``>= 0``."""
    dp = float(np.dot(p, direction))
    dq = float(np.dot(q, direction))
    if dp < -_TOL and dq < -_TOL:
        return None
    if dp >= -_TOL and dq >= -_TOL:
        return p, q
    t = dp / (dp - dq)
    axis = p + t * (q - p)
    axis = np.array([0.0, 0.0, float(axis[2])])
    if dp >= -_TOL:
        return p, axis
    return q, axis


def _walk(start: tuple, adj: dict, points: dict) -> list:
    """Follow the profile from the crown axis.  One unused edge at a time."""
    path = [start]
    used: set[tuple] = set()
    prev = None
    current = start
    while True:
        nxt = [
            other for other in adj.get(current, ())
            if (current, other) not in used and (other, current) not in used
        ]
        if not nxt:
            break
        if len(nxt) == 1 or prev is None:
            chosen = nxt[0] if len(nxt) == 1 else max(
                nxt, key=lambda key: float(np.hypot(points[key][0], points[key][1])))
        else:
            travel = points[current] - points[prev]
            travel_n = float(np.linalg.norm(travel))

            def score(key, travel=travel, travel_n=travel_n):
                step = points[key] - points[current]
                return float(np.dot(step, travel) / (np.linalg.norm(step) * travel_n + 1e-15))

            chosen = max(nxt, key=score)
        used.add((current, chosen))
        path.append(chosen)
        prev, current = current, chosen
    return path


def _touches(start, adj, visited) -> bool:
    """Whether ``start``'s chain already meets the profile that was walked."""
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for nxt in adj.get(current, ()):
            if nxt in visited:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _profile(start, adj, points) -> list:
    """The radial chain from the crown, including pieces a gap had split off.

    A watertight crown is one chain.  Rounding can open a hairline break in
    it, and a crown built without star facets has a real opening under the
    table edge.  Either way the next piece is the detached chain whose end
    sits closest to where this one stopped, and the straight jump across
    that opening counts in the arc length.
    """
    path = _walk(start, adj, points)
    visited = set(path)
    for _ in range(1000):
        ends = []
        for key, nbrs in adj.items():
            if key in visited or not nbrs:
                continue
            if sum(1 for nbr in nbrs if nbr not in visited) != 1:
                continue
            if _touches(key, adj, visited):
                continue
            ends.append(key)
        if not ends:
            break
        tail = points[path[-1]]
        nxt = min(ends, key=lambda key: float(np.linalg.norm(points[key] - tail)))
        extra = _walk(nxt, adj, points)
        path.extend(extra)
        visited.update(extra)
    return path


def _sigma_at(path: np.ndarray, sigma: np.ndarray, vertex: np.ndarray) -> tuple[float, float]:
    """Arc length of the profile point that lands on this vertex."""
    gap = np.linalg.norm(path - vertex, axis=1)
    nearest = int(np.argmin(gap))
    best_s = float(sigma[nearest])
    best_d = float(gap[nearest])
    extent = max(1.0, float(np.max(np.abs(vertex))), float(np.max(np.abs(path))))
    limit = max(1e-4, 1e-4 * extent)
    if best_d <= limit * 0.1:
        return best_s, best_d

    for i in range(len(path) - 1):
        edge = path[i + 1] - path[i]
        length2 = float(np.dot(edge, edge))
        if length2 <= 1e-24:
            continue
        t = float(np.clip(np.dot(vertex - path[i], edge) / length2, 0.0, 1.0))
        dist = float(np.linalg.norm(path[i] + t * edge - vertex))
        if dist < best_d:
            best_d = dist
            best_s = float(sigma[i] + t * (sigma[i + 1] - sigma[i]))
    if best_d > limit:
        raise ValueError(
            "vertex is not on its radial profile "
            f"(distance {best_d:.3e}): {vertex.tolist()}")
    return best_s, best_d
