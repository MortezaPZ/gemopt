"""The handover file, checked against the rules the MAXScript parses by.

3ds Max is not installed here, so the importer itself cannot be executed --
that has to happen on the client's machine.  What *can* be pinned down is the
contract between the two sides: this re-implements the parser exactly as the
MAXScript reads it, token for token, and asserts that what Python writes comes
back as the same stone.  A format change that would break the import shows up
here rather than in front of the client.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from gemopt.geometry import CutParams, build_stone
from gemopt.maxbridge import write_gem
from gemopt.precision import DECIMALS

SCRIPT = Path("maxscript/import_stone.ms")

# What rounding to DECIMALS places can cost a coordinate: half a unit in the
# last place, with room to spare.  Tied to the constant rather than typed as a
# number, so changing the project's precision moves the test with it instead of
# silently making it wrong.
ROUNDING = 10.0 ** -DECIMALS


def parse_like_maxscript(path):
    """The importer's reading rules, followed literally.

    ``f`` lines carry an optional group name before the indices, and the
    script tells them apart by asking whether the second token parses as an
    integer -- so that is what is done here too.
    """
    verts, faces, names, header = [], [], [], {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "v":
            verts.append([float(t) for t in tokens[1:4]])
        elif tokens[0] == "f":
            start, group = 1, "facet"
            try:
                int(tokens[1])
            except ValueError:
                group, start = tokens[1], 2
            idx = [int(t) for t in tokens[start:]]
            if len(idx) >= 3:
                faces.append(idx)
                names.append(group)
        elif tokens[0] in ("ior", "scale"):
            header[tokens[0]] = float(tokens[1])
        elif tokens[0] in ("facets", "verts", "columns"):
            header[tokens[0]] = int(tokens[1])
        elif tokens[0] == "rows":
            header["rows"] = tuple(int(t) for t in tokens[1:3])
    return verts, faces, names, header


@pytest.mark.parametrize("params", [
    CutParams(),
    CutParams(n_base=16, girdle_multiple=3),
    CutParams(extra_crown_rows=2, extra_pavilion_rows=2),
    CutParams(culet=0.03, extra_crown_rows=1),
])
def test_every_stone_survives_the_handover(tmp_path, params):
    mesh = build_stone(params)
    path = write_gem(tmp_path / "s.gem", mesh, params=params, ior=2.42)
    verts, faces, names, header = parse_like_maxscript(path)

    assert len(verts) == len(mesh.vertices)
    assert len(faces) == len(mesh.polygons) == header["facets"]
    assert header["verts"] == len(verts)
    assert np.allclose(verts, mesh.vertices, atol=ROUNDING)

    for got, poly in zip(faces, mesh.polygons):
        # MAXScript indices are 1-based
        assert [i - 1 for i in got] == list(poly)
    assert names == [mesh.facet_group[f] for f in range(mesh.n_facets)]


def test_the_refractive_index_travels_with_the_stone(tmp_path):
    """Otherwise Max builds a glass material for a diamond."""
    mesh = build_stone(CutParams())
    path = write_gem(tmp_path / "s.gem", mesh, params=CutParams(), ior=2.4175)
    _, _, _, header = parse_like_maxscript(path)
    assert header["ior"] == pytest.approx(2.4175, abs=1e-3)


def test_row_and_column_counts_are_declared(tmp_path):
    """The two numbers the client's video was about."""
    params = CutParams(n_base=12, extra_crown_rows=2, extra_pavilion_rows=1)
    path = write_gem(tmp_path / "s.gem", build_stone(params), params=params)
    _, _, _, header = parse_like_maxscript(path)
    assert header["columns"] == 12
    assert header["rows"] == (5, 3)          # 3+2 crown, 2+1 pavilion


def test_group_names_never_look_like_numbers(tmp_path):
    """The parser tells a group from an index by trying to parse an integer.

    A row called "12" would be swallowed as a vertex index and the facet would
    come out with the wrong corners.
    """
    mesh = build_stone(CutParams(extra_crown_rows=2, extra_pavilion_rows=2))
    for name in set(mesh.facet_group):
        with pytest.raises(ValueError):
            int(name)
        assert " " not in name, "a space would split the name into two tokens"


def test_no_facet_is_degenerate(tmp_path):
    """Fewer than three corners and the importer silently drops the facet."""
    mesh = build_stone(CutParams(extra_crown_rows=1))
    path = write_gem(tmp_path / "s.gem", mesh, params=CutParams())
    _, faces, _, _ = parse_like_maxscript(path)
    assert all(len(f) >= 3 for f in faces)
    assert all(len(set(f)) == len(f) for f in faces), "a repeated corner"


def test_the_script_declares_the_functions_the_main_block_calls():
    """A missing definition only shows up when Max runs it, which is too late."""
    source = SCRIPT.read_text(encoding="utf-8")
    defined = set(re.findall(r"^fn\s+(\w+)", source, re.M))
    for name in ("gemReadFile", "gemBuild", "gemRowsToMatIDs", "gemGlassMaterial",
                 "gemAssignMaterial", "gemAddLight", "gemDefault"):
        assert name in defined, f"{name} is called but never defined"


def test_if_do_is_never_paired_with_else():
    """``if ... do`` takes no else branch; the pair is a syntax error.

    3ds Max rejects the whole file at parse time, so one of these stops the
    import before a single vertex is built -- and it cannot be caught by
    reading the script, only by running Max.  Hence this.
    """
    for path in (SCRIPT, Path("maxscript/export_stone.ms")):
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\bif\b[^\n]*?\bdo\b[ \t]*\(", source):
            # walk to the paren that closes this block, then see what follows
            depth = 0
            for j in range(match.end() - 1, len(source)):
                if source[j] == "(":
                    depth += 1
                elif source[j] == ")":
                    depth -= 1
                    if depth == 0:
                        line = source[:match.start()].count("\n") + 1
                        assert not re.match(r"\s*else\b", source[j + 1:j + 40]), (
                            f"{path}:{line} pairs `do` with `else` -- "
                            f"3ds Max will not parse the file")
                        break


def test_no_call_to_a_maxscript_function_that_does_not_exist():
    """These were called, silently failed, and cost a whole feature.

    ``setNamedSelSetBitArray`` is not a MAXScript global at all -- 3ds Max 2027
    reports it as undefined -- and ``Editable_Poly`` cannot be constructed.
    Both were only found by running the script inside Max, so they are pinned
    here rather than left to be rediscovered.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    banned = {
        "setNamedSelSetBitArray": "not a MAXScript global; use material IDs",
        "getNamedSelSetBitArray": "not a MAXScript global",
    }
    for name, why in banned.items():
        assert name not in source, f"{name} does not exist -- {why}"
    assert not re.search(r"=\s*Editable_Poly\b", source), (
        "Editable_Poly is not creatable; convert an empty mesh instead")


def test_the_script_passes_an_index_to_the_material_function():
    """The material function grew a parameter; every call has to have grown too."""
    source = SCRIPT.read_text(encoding="utf-8")
    signature = re.search(r"fn gemAssignMaterial ([^=]+)=", source).group(1).split()
    for call in re.findall(r"gemAssignMaterial ([^\n]+)", source):
        if call.strip().startswith("obj kind"):        # the definition itself
            continue
        assert len(call.split()) >= len(signature), f"too few arguments: {call}"
