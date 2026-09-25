"""MaxScript for the same radial pattern :mod:`gemopt.pattern` describes.

N-gon cylinder sides, kite splines and rotation offset all come from
``PatternLayout`` — there is no second copy of the numbers.
"""

from __future__ import annotations

from pathlib import Path

from .pattern import PatternLayout, halo_radii, layout_from_n, n_gon_vertices, radial_array, table_vertices

__all__ = ["write_pattern_ms"]


def _pt(xy, z: float = 0.0) -> str:
    return f"[{xy[0]:.5f}, {xy[1]:.5f}, {z:.5f}]"


def write_pattern_ms(path, n: int | PatternLayout, scale: float = 50.0) -> Path:
    """Write a ``.ms`` that rebuilds the diagram as splines + an N-sided cylinder."""
    layout = n if isinstance(n, PatternLayout) else layout_from_n(int(n))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    s = float(scale)
    wedges = radial_array(layout)
    gon = n_gon_vertices(layout)

    lines = [
        "/*  gemopt radial pattern — generated from PatternLayout",
        f"    N = {layout.n}   sector = {layout.sector_deg:.5f} deg",
        f"    stroke = {layout.stroke_width:.5f}   offset = {layout.rotation_offset_deg:.5f}",
        "    Scripting menu -> Run Script.",
        "*/",
        f"n = {layout.n}",
        f"sectorDeg = {layout.sector_deg:.8f}",
        f"offsetDeg = {layout.rotation_offset_deg:.8f}",
        f"radius = {layout.radius * s:.8f}",
        f"stroke = {layout.stroke_width * s:.8f}",
        f"tableRadius = {layout.table_radius * s:.8f}",
        "",
        "-- Outer N-gon and the table N-gon: same Sides, same offset as the PNG.",
        "cyl = cylinder radius:radius height:(radius * 0.08) sides:n capstart:on capend:on",
        "cyl.name = \"gemopt_ngon\"",
        "rotate cyl (angleaxis offsetDeg [0,0,1])",
        "tbl = cylinder radius:tableRadius height:(radius * 0.04) sides:n capstart:on capend:on",
        "tbl.name = \"gemopt_table\"",
        "rotate tbl (angleaxis offsetDeg [0,0,1])",
        "",
        "fn gemSpline pts name = (",
        "    local sp = splineShape name:name",
        "    addNewSpline sp",
        "    for p in pts do addKnot sp 1 #corner #line p",
        "    close sp 1",
        "    updateShape sp",
        "    sp",
        ")",
        "",
    ]

    for w in wedges:
        pts = [
            (w.table[0] * s, w.table[1] * s),
            (w.valley_left[0] * s, w.valley_left[1] * s),
            (w.tip[0] * s, w.tip[1] * s),
            (w.valley_right[0] * s, w.valley_right[1] * s),
        ]
        arr = ", ".join(_pt(p) for p in pts)
        lines.append(f"gemSpline #({arr}) \"wedge_{w.index:02d}\"")

    table_pts = ", ".join(_pt((x * s, y * s)) for x, y in table_vertices(layout))
    lines.append(f"gemSpline #({table_pts}) \"table_outline\"")
    gon_pts = ", ".join(_pt((x * s, y * s)) for x, y in gon)
    lines.append(f"gemSpline #({gon_pts}) \"ngon_outline\"")
    lines.append("")
    for i, r in enumerate(halo_radii(layout)):
        lines.append(
            f"c{i} = circle radius:{r * s:.8f} name:\"halo_{i:02d}\""
        )
    lines.append("")
    lines.append("print (\"gemopt pattern N=\" + n as string + \" sector=\" + sectorDeg as string)")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
