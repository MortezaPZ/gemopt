"""Getting stones in and out of 3ds Max.

The client's existing workflow lives in MAXScript and talks to Excel.  This
module keeps that door open in both directions:

* :func:`write_gem` saves an optimised stone in a small text format that the
  bundled ``import_stone.ms`` rebuilds as a real Editable Poly -- facets as
  facets, not triangles, so it can still be edited by hand afterwards;
* :func:`read_gem` and :func:`read_obj` load geometry that came *out* of Max,
  so an existing model can be measured and scored without being rebuilt;
* :func:`write_vertex_csv` writes the same vertex table the current tool
  pushes to Excel, keeping the client's spreadsheet habits intact;
* :func:`run_maxscript` drives ``3dsmaxbatch.exe`` for the final V-Ray render
  of the winner, which is the one place a real renderer is still wanted.

The optimisation itself never needs 3ds Max to be running.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .geometry import CutParams, Mesh
from .precision import DECIMALS

__all__ = [
    "write_gem",
    "read_gem",
    "read_obj",
    "write_vertex_csv",
    "maxscript_dir",
    "find_3dsmaxbatch",
    "find_3dsmax_exe",
    "run_maxscript",
]

GEM_VERSION = 1


def maxscript_dir() -> Path:
    """Where the bundled ``.ms`` files live."""
    return Path(__file__).resolve().parent.parent / "maxscript"


# --------------------------------------------------------------------------
# the interchange format
# --------------------------------------------------------------------------


def write_gem(
    path,
    mesh: Mesh,
    params: CutParams | None = None,
    scale: float = 1.0,
    metrics: dict | None = None,
    ior: float | None = None,
) -> Path:
    """Write a stone as plain text.

    Deliberately not OBJ: MAXScript can parse this in a dozen lines with no
    importer settings, no dialog boxes and no unit guessing, which matters
    when the file is being read inside a batch run with nobody watching.
    ``scale`` converts the unit-radius model into scene units (millimetres
    for a stone of that diameter).
    """
    path = Path(path)
    verts = mesh.vertices * float(scale)
    lines = [
        f"# gemopt stone v{GEM_VERSION}",
        f"# coordinates in scene units, {DECIMALS} decimals",
        f"scale {scale:.{DECIMALS}f}",
        f"facets {len(mesh.polygons)}",
        f"verts {len(verts)}",
    ]
    # The refractive index travels with the geometry so 3ds Max builds the
    # material for the material the stone was actually optimised for, instead
    # of the glass the importer used to assume.
    if ior is not None:
        lines.append(f"ior {float(ior):.{DECIMALS}f}")
    if params is not None:
        lines.append(f"rows {3 + int(params.extra_crown_rows)} "
                     f"{2 + int(params.extra_pavilion_rows)}")
        lines.append(f"columns {int(params.n_base)}")
    for v in verts:
        lines.append(f"v {v[0]:.{DECIMALS}f} {v[1]:.{DECIMALS}f} {v[2]:.{DECIMALS}f}")
    for fid, poly in enumerate(mesh.polygons):
        group = mesh.facet_group[fid] if fid < len(mesh.facet_group) else "facet"
        lines.append("f " + group + " " + " ".join(str(i + 1) for i in poly))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if params is not None:
        side = path.with_suffix(".json")
        payload = {
            "parameters": asdict(params),
            "derived": params.derived(),
            "scale": scale,
        }
        if metrics:
            payload["metrics"] = metrics
        side.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_gem(path) -> Mesh:
    """Read back a ``.gem`` file, including ones written by MAXScript."""
    verts: list[list[float]] = []
    polys: list[list[int]] = []
    groups: list[str] = []

    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            rest = parts[1:]
            if rest and not _is_int(rest[0]):
                groups.append(rest[0])
                rest = rest[1:]
            else:
                groups.append("facet")
            polys.append([int(i) - 1 for i in rest])

    if not verts or not polys:
        raise ValueError(f"{path} contains no geometry")
    return _mesh_from_polygons(np.asarray(verts, dtype=np.float64), polys, groups)


def read_obj(path) -> Mesh:
    """Minimal OBJ reader for meshes exported from 3ds Max.

    Only ``v``, ``f`` and ``g`` are honoured; that is all a faceted stone
    needs, and it avoids dragging in a mesh library for one file format.
    """
    verts: list[list[float]] = []
    polys: list[list[int]] = []
    groups: list[str] = []
    current = "facet"

    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif parts[0] == "g":
            current = parts[1] if len(parts) > 1 else "facet"
        elif parts[0] == "f":
            # face entries may be "1", "1/2", "1//3" or "1/2/3"
            idx = [int(tok.split("/")[0]) for tok in parts[1:]]
            n = len(verts)
            polys.append([(i - 1) if i > 0 else (n + i) for i in idx])
            groups.append(current)

    if not verts or not polys:
        raise ValueError(f"{path} contains no geometry")
    return _mesh_from_polygons(np.asarray(verts, dtype=np.float64), polys, groups)


def _is_int(token: str) -> bool:
    try:
        int(token)
    except ValueError:
        return False
    return True


def _mesh_from_polygons(verts: np.ndarray, polys: list[list[int]], groups: list[str]) -> Mesh:
    tris: list[tuple[int, int, int]] = []
    fids: list[int] = []
    for fid, poly in enumerate(polys):
        for i in range(1, len(poly) - 1):
            tris.append((poly[0], poly[i], poly[i + 1]))
            fids.append(fid)
    return Mesh(
        vertices=verts,
        triangles=np.asarray(tris, dtype=np.int32),
        facet_id=np.asarray(fids, dtype=np.int32),
        facet_group=groups,
        polygons=polys,
    )


def write_vertex_csv(path, mesh: Mesh, scale: float = 1.0) -> Path:
    """The vertex table, in the shape the client's Excel sheet expects.

    One row per vertex with its index and coordinates, plus which facet rows
    it belongs to -- the "which row is this vertex in" question the existing
    MAXScript tool answers with its numbered buttons.
    """
    path = Path(path)
    member: dict[int, set[str]] = {}
    for fid, poly in enumerate(mesh.polygons):
        group = mesh.facet_group[fid] if fid < len(mesh.facet_group) else "facet"
        for v in poly:
            member.setdefault(v, set()).add(group)

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["vertex", "x", "y", "z", "rows"])
        for i, v in enumerate(mesh.vertices * float(scale)):
            writer.writerow([
                i + 1,
                f"{v[0]:.{DECIMALS}f}", f"{v[1]:.{DECIMALS}f}", f"{v[2]:.{DECIMALS}f}",
                "+".join(sorted(member.get(i, {"?"}))),
            ])
    return path


# --------------------------------------------------------------------------
# driving 3ds Max
# --------------------------------------------------------------------------


MAX_EXE_ENV = "GEMOPT_MAX_EXE"
"""Set this to the full path of 3dsmaxbatch.exe (or 3dsmax.exe) to skip the search.

The last resort for an install the search cannot see -- a network path, a
portable copy, a version Autodesk registers somewhere new.  One environment
variable beats editing code on the client's machine.
"""


def _from_environment() -> Path | None:
    value = os.environ.get(MAX_EXE_ENV, "").strip().strip('"')
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _install_dirs() -> list[Path]:
    """Every folder on this machine that might be a 3ds Max installation.

    Newest first.  The registry is asked before the disk is guessed at,
    because an install moved off the default path -- which is common on a
    workstation whose C: drive is a small SSD -- is invisible to guessing.
    """
    seen: list[Path] = []

    def add(path) -> None:
        try:
            p = Path(path)
        except (TypeError, ValueError):
            return
        if p.is_dir() and p not in seen:
            seen.append(p)

    try:                                             # pragma: no cover - Windows
        import winreg

        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for key_path in (r"SOFTWARE\Autodesk\3dsMax",
                             r"SOFTWARE\WOW6432Node\Autodesk\3dsMax"):
                try:
                    with winreg.OpenKey(hive, key_path) as root:
                        count = winreg.QueryInfoKey(root)[0]
                        subs = [winreg.EnumKey(root, i) for i in range(count)]
                        for sub in sorted(subs, reverse=True):
                            try:
                                with winreg.OpenKey(root, sub) as node:
                                    add(winreg.QueryValueEx(node, "Installdir")[0])
                            except OSError:
                                continue
                except OSError:
                    continue
    except ImportError:
        pass

    roots = [Path(r"C:\Program Files\Autodesk"), Path(r"C:\Program Files (x86)\Autodesk")]
    for var in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        value = os.environ.get(var)
        if value:
            roots.append(Path(value) / "Autodesk")
    for letter in "DEFG":                            # installs moved off C:
        roots.append(Path(f"{letter}:\\Program Files\\Autodesk"))
        roots.append(Path(f"{letter}:\\Autodesk"))

    for root in roots:
        if not root.is_dir():
            continue
        try:
            children = sorted(root.iterdir(), reverse=True)
        except OSError:
            continue
        for child in children:
            if child.is_dir() and "3ds max" in child.name.lower():
                add(child)

    # A folder called "3ds Max 2027" sitting anywhere on a drive.
    #
    # The registry is not the last word: an install whose files were moved
    # afterwards leaves an Installdir pointing at a folder that no longer
    # exists, and then nothing above finds it.  Two levels down from a drive
    # root is where those end up -- D:\anything\3ds Max 2027 -- and stopping
    # at two keeps this to a few dozen directory reads rather than a disk
    # crawl.
    for letter in "CDEFG":
        drive = Path(f"{letter}:\\")
        if not drive.is_dir():
            continue
        try:
            top = sorted(drive.iterdir(), reverse=True)
        except OSError:
            continue
        for folder in top:
            try:
                if not folder.is_dir():
                    continue
                if "3ds max" in folder.name.lower():
                    add(folder)
                    continue
                for child in folder.iterdir():
                    if child.is_dir() and "3ds max" in child.name.lower():
                        add(child)
            except OSError:                      # unreadable or offline
                continue
    return seen


def find_3dsmaxbatch(explicit: str | None = None) -> Path | None:
    """Locate ``3dsmaxbatch.exe``.

    Checked in order: what the caller passed, ``PATH``, the registry, then the
    usual install folders newest-first.
    """
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    override = _from_environment()
    if override is not None:
        return override

    found = shutil.which("3dsmaxbatch")
    if found:
        return Path(found)

    for folder in _install_dirs():
        candidate = folder / "3dsmaxbatch.exe"
        if candidate.exists():
            return candidate
    return None


def find_3dsmax_exe(explicit: str | None = None) -> Path | None:
    """Locate ``3dsmax.exe`` -- the fallback when the batch tool is missing.

    ``3dsmaxbatch.exe`` ships with 3ds Max but not with every licence tier, and
    it is the first thing a stripped-down install drops.  The main executable
    can run a script too (``-U MAXScript file.ms``); it opens a window on the
    way, which is worse, but a stone the client can actually open beats a
    tidier failure.
    """
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    override = _from_environment()
    if override is not None:
        return override

    found = shutil.which("3dsmax")
    if found:
        return Path(found)

    for folder in _install_dirs():
        candidate = folder / "3dsmax.exe"
        if candidate.exists():
            return candidate
    return None


def run_maxscript(
    script: Path | str,
    variables: dict | None = None,
    max_batch: str | None = None,
    timeout: int = 1800,
) -> subprocess.CompletedProcess:
    """Run a MAXScript file head-less.

    Values in ``variables`` are handed over with ``-mxsString``, which is how
    3dsmaxbatch passes arguments; the script reads them back out of the
    global scope.  Raises if 3ds Max cannot be found, because failing loudly
    beats silently skipping the render of the final stone.
    """
    exe = find_3dsmaxbatch(max_batch)
    if exe is None:
        raise FileNotFoundError(
            "3dsmaxbatch.exe not found. Pass its full path explicitly, or run "
            "the generated .ms file from inside 3ds Max instead -- the "
            "optimisation itself does not need 3ds Max."
        )

    cmd = [str(exe), str(Path(script).resolve())]
    for key, value in (variables or {}).items():
        cmd += ["-mxsString", f"{key}:{value}"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
