"""Producing a real ``.max`` file.

Autodesk's format is closed and undocumented, so nothing here writes one --
3ds Max does, driven head-lessly.  That means the interesting behaviour to
pin down is everything *around* the call: that a machine without Max gets a
clear answer instead of a crash, that the command is assembled correctly, and
that the importer it drives actually accepts the arguments being passed.
"""

import re
from pathlib import Path

import pytest

from gemopt.geometry import CutParams, build_stone
from gemopt.maxbridge import write_gem
from gemopt.tomax import MaxResult, gem_to_max, max_is_available, write_build_script

SCRIPT = Path("maxscript/import_stone.ms")


@pytest.fixture
def gem(tmp_path):
    path = tmp_path / "stone.gem"
    write_gem(path, build_stone(CutParams()), params=CutParams(), ior=1.52)
    return path


def test_a_missing_file_is_reported_not_raised(tmp_path):
    result = gem_to_max(tmp_path / "nope.gem")
    assert isinstance(result, MaxResult)
    assert not result.ok
    assert "پیدا نشد" in result.message


def test_without_3ds_max_it_says_so_plainly(gem, monkeypatch):
    """The common case on a machine that only runs the optimiser."""
    monkeypatch.setattr("gemopt.tomax.max_is_available", lambda *_a, **_k: None)
    result = gem_to_max(gem)
    assert not result.ok
    assert "3ds Max" in result.message
    # and it should point at what the client *can* use instead
    assert ".gem" in result.message and ".obj" in result.message


def _drive(monkeypatch, out_path):
    """Stand in for 3dsmaxbatch and capture the script it was handed."""
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["script"] = Path(command[1]).read_text(encoding="utf-8")
        Path(out_path).touch()

        class Done:
            stdout = stderr = ""
        return Done()

    monkeypatch.setattr("gemopt.tomax.max_is_available",
                        lambda *_a, **_k: Path("3dsmaxbatch.exe"))
    monkeypatch.setattr("subprocess.run", fake_run)
    return seen


def as_maxscript(path) -> str:
    """A Windows path as it must appear inside a MAXScript string literal.

    MAXScript reads backslash escapes, so ``results\\app\\stone.gem`` arrives
    with a BEL where ``\\a`` was and a path Max cannot open.  Doubling them is
    the fix, and this spells out what the script is expected to contain rather
    than checking for the raw path, which would only pass on a format that is
    quietly broken.
    """
    return str(path).replace("\\", "\\\\")


def test_every_path_reaches_the_script(gem, tmp_path, monkeypatch):
    out = (tmp_path / "made.max").resolve()
    seen = _drive(monkeypatch, out)

    pattern = tmp_path / "out_crown.png"
    pattern.write_bytes(b"png")
    result = gem_to_max(gem, out, material="physical", name="mystone",
                        pattern=pattern, half_width=1.02)
    assert result.ok and result.path == out

    script = result.script.read_text(encoding="utf-8")
    assert f'global gemFile = "{as_maxscript(gem.resolve())}"' in script
    assert f'global gemSaveMax = "{as_maxscript(out)}"' in script
    assert 'global gemMaterial = "physical"' in script
    assert 'global gemName = "mystone"' in script
    assert f'global gemPattern = "{as_maxscript(pattern.resolve())}"' in script
    assert "global gemHalfWidth = 1.02" in script
    assert "import_stone.ms" in script
    # The file the client runs must still open the rollout.
    assert "gemoptHeadless" not in script

    runner = seen["script"]
    assert "global gemoptHeadless = true" in runner
    assert f'fileIn "{as_maxscript(result.script)}"' in runner
    assert "quitMax" not in runner


def test_backslash_escapes_cannot_eat_a_path(tmp_path, monkeypatch):
    """``results\\app`` is the program's own output folder.

    Written raw, MAXScript reads the ``\\a`` as a bell character and the path
    stops pointing at anything.  This is the exact folder the client's runs
    land in, so it gets its own test.
    """
    folder = tmp_path / "results" / "app"
    folder.mkdir(parents=True)
    gem = folder / "stone.gem"
    write_gem(gem, build_stone(CutParams()), params=CutParams(), ior=1.52)
    out = (folder / "stone.max").resolve()
    seen = _drive(monkeypatch, out)

    result = gem_to_max(gem, out)
    assert result.ok
    script = result.script.read_text(encoding="utf-8")
    assert "\\\\app\\\\stone.gem" in script
    assert "\\app\\stone.gem" not in script.replace("\\\\", "")
    # The runner points at that same folder, so its path is escaped too.
    assert "\\\\app\\\\" in seen["script"]


def test_paths_with_spaces_and_non_ascii_survive(tmp_path, monkeypatch):
    """The client's own folders are Persian and contain spaces.

    Passing these as -mxsString would carry the shell's quotes into the string
    and hand Max a path it cannot open, which is why they go through a
    generated script instead.
    """
    folder = tmp_path / "خروجی کارفرما (1)"
    folder.mkdir()
    gem = folder / "stone.gem"
    write_gem(gem, build_stone(CutParams()), params=CutParams(), ior=1.52)
    out = (folder / "stone.max").resolve()
    seen = _drive(monkeypatch, out)

    pattern = folder / "out_crown.png"
    pattern.write_bytes(b"png")
    result = gem_to_max(gem, out, pattern=pattern, half_width=1.02)
    assert result.ok
    script = result.script.read_text(encoding="utf-8")
    assert as_maxscript(gem.resolve()) in script
    assert as_maxscript(pattern.resolve()) in script
    assert "خروجی کارفرما (1)" in script
    # string globals stay one quoted path; the half-width is a bare number
    for line in script.splitlines():
        if not line.startswith("global "):
            continue
        if "undefined" in line or "gemHalfWidth" in line:
            assert '"' not in line
        else:
            assert line.count('"') == 2, f"unbalanced quotes: {line}"


def test_the_full_executable_quits_without_marking_the_kept_script(gem, tmp_path, monkeypatch):
    """3dsmax.exe does not exit on its own, and must not open the rollout."""
    out = (tmp_path / "made.max").resolve()
    seen = {}

    def fake_run(command, **kwargs):
        seen["script"] = Path(command[3]).read_text(encoding="utf-8")
        Path(out).touch()

        class Done:
            stdout = stderr = ""
        return Done()

    monkeypatch.setattr("gemopt.tomax.max_is_available",
                        lambda *_a, **_k: Path("3dsmax.exe"))
    monkeypatch.setattr("subprocess.run", fake_run)
    result = gem_to_max(gem, out)

    assert result.ok
    assert "global gemoptHeadless = true" in seen["script"]
    assert "quitMax #noPrompt" in seen["script"]
    assert "gemoptHeadless" not in result.script.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*_run.ms"))


def test_crown_is_the_default_picture_when_both_views_exist(tmp_path):
    from gemopt.app import _pattern_image

    (tmp_path / "out_crown.png").write_bytes(b"a")
    (tmp_path / "out_pavilion.png").write_bytes(b"b")
    assert _pattern_image(tmp_path, ["pavilion", "crown"]).name == "out_crown.png"
    assert _pattern_image(tmp_path, ["pavilion"]).name == "out_pavilion.png"


def test_each_build_drops_the_previous_stones_picture(gem, tmp_path):
    """Max keeps globals for the whole session.

    A second stone that has no picture of its own must not inherit the first
    stone's file, and a picture that does travel with this stone must replace
    the cleared value rather than sit underneath it.
    """
    bare = write_build_script(gem).read_text(encoding="utf-8").splitlines()
    assert bare.count("global gemPattern = undefined") == 1
    assert not any(line.startswith('global gemPattern = "') for line in bare)

    pattern = tmp_path / "out_crown.png"
    pattern.write_bytes(b"png")
    lines = write_build_script(gem, pattern=pattern, half_width=1.02).read_text(
        encoding="utf-8").splitlines()
    cleared = lines.index("global gemPattern = undefined")
    assigned = next(i for i, line in enumerate(lines)
                    if line.startswith('global gemPattern = "'))
    assert cleared < assigned
    width_cleared = lines.index("global gemHalfWidth = undefined")
    width_set = lines.index("global gemHalfWidth = 1.02")
    assert width_cleared < width_set


def test_a_half_width_that_is_not_a_length_is_left_unset(gem):
    """``inf`` is not a MAXScript number; writing it stops the stone building."""
    for bad in (float("inf"), float("-inf"), float("nan"), 0.0, -2.0):
        lines = write_build_script(gem, half_width=bad).read_text(encoding="utf-8").splitlines()
        assert [line for line in lines if "gemHalfWidth" in line] == [
            "global gemHalfWidth = undefined"
        ]


def test_the_temporary_script_is_cleaned_up(gem, tmp_path, monkeypatch):
    out = (tmp_path / "made.max").resolve()
    _drive(monkeypatch, out)
    gem_to_max(gem, out)
    leftovers = list(tmp_path.glob("*_auto.ms")) + list(tmp_path.glob("*_run.ms"))
    assert not leftovers, f"left behind {leftovers}"


def test_the_build_script_is_kept_even_without_max(gem, tmp_path, monkeypatch):
    """It is the only route to a .max on a machine that has no 3ds Max.

    The optimiser runs here; 3ds Max is on the client's machine.  Writing this
    file unconditionally is what lets the stone cross that gap -- one Run
    Script, no paths to type.
    """
    monkeypatch.setattr("gemopt.tomax.max_is_available", lambda *_a, **_k: None)
    pattern = tmp_path / "out_pavilion.png"
    pattern.write_bytes(b"png")
    result = gem_to_max(gem, tmp_path / "made.max", pattern=pattern, half_width=2.04)

    assert not result.ok
    assert result.script is not None and result.script.exists()
    body = result.script.read_text(encoding="utf-8")
    assert "import_stone.ms" in body and "gemSaveMax" in body
    assert as_maxscript(pattern.resolve()) in body
    assert "global gemHalfWidth = 2.04" in body
    assert "gemoptHeadless" not in body


def test_it_reports_when_max_ran_but_wrote_nothing(gem, monkeypatch):
    """Silence from Max must not be mistaken for success."""
    class Done:
        stdout = "MAXScript error somewhere"
        stderr = ""

    monkeypatch.setattr("gemopt.tomax.max_is_available",
                        lambda *_a, **_k: Path("3dsmaxbatch.exe"))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: Done())
    result = gem_to_max(gem)
    assert not result.ok
    assert "MAXScript error" in result.log


def test_a_real_subprocess_run_produces_the_max(gem, tmp_path):
    """The whole chain, with nothing mocked except 3ds Max itself.

    Every other test here patches ``subprocess.run`` and therefore proves
    nothing about the command actually being launchable.  This one puts a
    stand-in executable on disk and runs it for real, so the argument order,
    the path escaping and the "did the file appear" check are all exercised
    the way they will be on the client's machine.
    """
    import subprocess
    import sys

    out = (tmp_path / "made.max").resolve()
    helper = tmp_path / "stand_in.py"
    helper.write_text(
        "import re, sys\n"
        "from pathlib import Path\n"
        "def unesc(text):\n"
        "    return text.replace('\\\\\\\\', '\\\\')\n"
        "launched = Path(sys.argv[1]).read_text(encoding='utf-8')\n"
        "assert 'global gemoptHeadless = true' in launched\n"
        "inc = re.search(r'fileIn \"(.+?)\"', launched).group(1)\n"
        "body = Path(unesc(inc)).read_text(encoding='utf-8')\n"
        "save = re.search(r'global gemSaveMax = \"(.+?)\"', body).group(1)\n"
        "gem  = re.search(r'global gemFile = \"(.+?)\"', body).group(1)\n"
        "assert Path(unesc(gem)).exists(), 'gemFile unreadable'\n"
        "Path(unesc(save)).write_bytes(b'FAKE.MAX')\n"
        "print('gemopt: saved', save)\n",
        encoding="utf-8")

    # named so gem_to_max takes the 3dsmaxbatch branch
    stand_in = tmp_path / "3dsmaxbatch.bat"
    stand_in.write_text(f'@echo off\n"{sys.executable}" "{helper}" %*\n',
                        encoding="utf-8")

    if not sys.platform.startswith("win"):
        pytest.skip("the .bat stand-in only runs on Windows")

    result = gem_to_max(gem, out, batch_exe=str(stand_in))

    assert result.ok, f"{result.message}\n{result.log}"
    assert out.exists() and out.read_bytes() == b"FAKE.MAX"
    assert result.path == out
    assert "saved" in result.log
    assert not list(tmp_path.glob("*_auto.ms"))
    assert not list(tmp_path.glob("*_run.ms"))


def test_the_environment_can_point_at_an_install_we_cannot_find(tmp_path, monkeypatch):
    """A network path, another drive, a version Autodesk registers somewhere new.

    C: on the client's machine is nearly full, so Max will land on another
    drive; this is the escape hatch when the search still misses it.
    """
    from gemopt.maxbridge import MAX_EXE_ENV, find_3dsmaxbatch

    fake = tmp_path / "3dsmaxbatch.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv(MAX_EXE_ENV, str(fake))
    assert find_3dsmaxbatch() == fake

    monkeypatch.setenv(MAX_EXE_ENV, str(tmp_path / "not_there.exe"))
    assert find_3dsmaxbatch() is None or find_3dsmaxbatch().exists()


def test_the_importer_declares_the_global_we_set():
    """`gemSaveMax` is passed in from outside; the script has to declare it."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert re.search(r"^global gemSaveMax", source, re.M)
    assert "saveMaxFile" in source, "nothing in the script actually saves"


def test_auto_puts_the_max_beside_the_gem():
    """The documented shorthand has to be handled, not silently ignored."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'savePath == "auto"' in source


def test_finding_max_never_raises():
    """Called on every machine, including ones with no Autodesk anything."""
    assert max_is_available() is None or max_is_available().exists()


def test_the_max_file_is_part_of_finishing_a_run():
    """The client asked for it to be in the folder, not behind a button.

    So the worker builds it as the last step of the job, alongside the .gem
    and .obj, and the interface has no control for it at all.
    """
    import inspect

    from gemopt.app import FA, App, _worker

    worker = inspect.getsource(_worker)
    assert "gem_to_max" in worker, "the run never builds a .max"
    assert "max_file" in worker, "the result never reports whether it got one"
    assert "_pattern_image" in worker, "the run never hands the picture across"
    assert "half_width" in worker, "the picture frame never reaches the script"

    # nothing in the window should offer it as a separate action
    window = inspect.getsource(App)
    for gone in ("max_btn", "auto_max_var", "_make_max"):
        assert gone not in window, f"{gone} is still in the interface"

    for key in ("making_max", "max_note", "no_max"):
        assert FA.get(key, "").strip(), f"{key} has no text"
