"""Turn a solved stone into a real ``.max`` file.

``.max`` is Autodesk's closed binary format.  Nothing outside 3ds Max can
write one -- there is no library, and the format is not documented -- so the
only honest way to produce it is to have Max do the writing.

What this does is drive that automatically: it finds ``3dsmaxbatch.exe``,
hands it the importer with the paths already filled in, and Max opens
head-less, builds the stone, saves the ``.max`` and exits.  No window, no
clicking, a few seconds.

If 3ds Max is not installed the file simply cannot be produced on that
machine, and this says so plainly rather than failing somewhere confusing.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .maxbridge import find_3dsmax_exe, find_3dsmaxbatch, maxscript_dir

__all__ = ["MaxResult", "gem_to_max", "max_is_available", "write_build_script"]


@dataclass
class MaxResult:
    """What happened, in a form the interface can show without interpreting."""

    ok: bool
    path: Path | None
    message: str
    log: str = ""
    script: Path | None = None


def max_is_available(explicit: str | None = None) -> Path | None:
    """The batch executable, if this machine has 3ds Max at all."""
    return find_3dsmaxbatch(explicit) or find_3dsmax_exe(explicit)


def _literal(value) -> str:
    """A MAXScript string literal.

    Paths go through a generated script rather than ``-mxsString``.  The
    documented form is ``-mxsString name:"value"`` -- but those quotes are
    stripped by the shell, and we launch without one, so they would arrive
    inside the string and turn every path into one Max cannot open.  Writing
    the assignments into a small script sidesteps the question entirely and
    handles spaces and non-ASCII folder names for free.
    """
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _number(value: float) -> str:
    """A MAXScript number, not a quoted string and not a binary dump."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"not a length: {value}")
    text = f"{number:.8f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-", "-0") else "0"


def _positive_length(value) -> float | None:
    """A half-width Max can store, or nothing when the caller has no frame."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def write_build_script(
    gem_path,
    out_path=None,
    material: str = "vray",
    ies: str | None = None,
    name: str = "gemopt_stone",
    script_path=None,
    pattern: str | None = None,
    half_width: float | None = None,
    pattern_face: str | None = None,
) -> Path:
    """Write the one-click ``.ms`` that builds this stone and saves the ``.max``.

    This is what makes the ``.max`` reachable on a machine this program never
    runs on: the client drags this file into 3ds Max -- Scripting, Run Script --
    and the stone is built and saved with no paths to type and nothing to
    configure.  It is written on every run, whether or not Max is installed
    here, because the machine that has Max is usually not this one.
    """
    gem_path = Path(gem_path).resolve()
    out_path = Path(out_path).resolve() if out_path else gem_path.with_suffix(".max")
    script_path = (Path(script_path).resolve() if script_path
                   else gem_path.with_name(gem_path.stem + "_build.ms"))

    lines = [
        "/*  gemopt -- build this stone in 3ds Max and save it as .max",
        "",
        "    Scripting menu -> Run Script -> pick this file.",
        "    The stone appears, and the .max is written next to it.",
        "*/",
        f"global gemFile = {_literal(gem_path)}",
        f"global gemSaveMax = {_literal(out_path)}",
        f"global gemMaterial = {_literal(material)}",
        f"global gemName = {_literal(name)}",
    ]
    if ies:
        lines.append(f"global gemIES = {_literal(Path(ies).resolve())}")
    # Globals survive for the whole Max session.  Clear them first, or the
    # previous stone's picture and frame would be applied to this one.
    # The headless flag is not written here: this file is what the client
    # runs by hand, and the rollout has to open.
    lines.append("global gemPattern = undefined")
    lines.append("global gemHalfWidth = undefined")
    lines.append("global gemPatternFace = undefined")
    if pattern:
        lines.append(f"global gemPattern = {_literal(Path(pattern).resolve())}")
    length = _positive_length(half_width)
    if length is not None:
        lines.append(f"global gemHalfWidth = {_number(length)}")
    if pattern_face == "pavilion":
        lines.append('global gemPatternFace = "pavilion"')
    lines.append(f"fileIn {_literal(maxscript_dir() / 'import_stone.ms')}")

    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path


def _headless_runner(script: Path, quit_after: bool) -> Path:
    """A throwaway script that builds the stone without opening the rollout.

    The file the client keeps does not carry this flag, so running it from
    the Scripting menu still shows the pattern window.
    """
    runner = script.with_name(script.stem + "_run.ms")
    lines = [
        "global gemoptHeadless = true",
        f"fileIn {_literal(script)}",
    ]
    if quit_after:
        lines.append("quitMax #noPrompt")
    runner.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return runner


def gem_to_max(
    gem_path,
    out_path=None,
    material: str = "vray",
    ies: str | None = None,
    name: str = "gemopt_stone",
    batch_exe: str | None = None,
    timeout: int = 420,
    pattern: str | None = None,
    half_width: float | None = None,
    pattern_face: str | None = None,
) -> MaxResult:
    """Build ``gem_path`` inside 3ds Max and save it as a ``.max`` file.

    Returns rather than raises: "3ds Max is not installed" is a normal answer
    on a machine that only runs the optimiser, and the caller should be able
    to say so without catching anything.
    """
    gem_path = Path(gem_path).resolve()
    if not gem_path.exists():
        return MaxResult(False, None, f"فایل پیدا نشد: {gem_path}")

    out_path = Path(out_path).resolve() if out_path else gem_path.with_suffix(".max")

    importer = maxscript_dir() / "import_stone.ms"
    if not importer.exists():
        return MaxResult(False, None, f"اسکریپت مکس پیدا نشد: {importer}")

    # Written first and kept, not written only when Max is here: the machine
    # that has Max is usually not the machine that ran the search, and this
    # file is what carries the stone across.
    try:
        script = write_build_script(gem_path, out_path, material=material,
                                    ies=ies, name=name, pattern=pattern,
                                    half_width=half_width,
                                    pattern_face=pattern_face)
    except OSError as exc:
        return MaxResult(False, None, f"نوشتن اسکریپت مکس ممکن نشد: {exc}")

    exe = max_is_available(batch_exe)
    if exe is None:
        return MaxResult(
            False, None,
            "3ds Max روی این سیستم پیدا نشد.\n"
            "فرمت .max فقط توسط خود 3ds Max قابل ساخت است، پس این فایل باید روی "
            "سیستمی ساخته شود که مکس روی آن نصب است.\n"
            f"برای همین کار این اسکریپت ساخته شد: {script.name}\n"
            "روی سیستمِ مکس → منوی Scripting → Run Script → همین فایل. "
            "سنگ ساخته می‌شود و .max کنارش نوشته می‌شود.\n"
            "فایل‌های .gem و .obj هم ساخته شده‌اند و هر دو در مکس باز می‌شوند.",
            script=script)

    # 3dsmaxbatch is the right tool and is what gets found first; 3dsmax.exe is
    # the fallback for installs that do not ship it, and it is driven
    # differently, so which one we got decides the command.  Either way the
    # launched file is a runner: the kept script must not know it is headless.
    quit_after = "batch" not in exe.name.lower()
    try:
        runner = _headless_runner(script, quit_after=quit_after)
    except OSError as exc:
        return MaxResult(False, None, f"نوشتن اسکریپت موقت ممکن نشد: {exc}",
                         script=script)
    if quit_after:
        # 3dsmax.exe runs a script too, but never exits on its own.  Without
        # the quit the run sits at the timeout with a window open.
        command = [str(exe), "-U", "MAXScript", str(runner), "-q", "-silent"]
    else:
        command = [str(exe), str(runner)]

    try:
        finished = subprocess.run(command, capture_output=True, text=True,
                                  timeout=timeout, errors="replace")
    except subprocess.TimeoutExpired:
        return MaxResult(False, None,
                         f"3ds Max در {timeout} ثانیه پاسخ نداد.\n"
                         f"می‌توانید {script.name} را دستی داخل مکس اجرا کنید.",
                         script=script)
    except OSError as exc:
        return MaxResult(False, None, f"اجرای 3ds Max ممکن نشد: {exc}",
                         script=script)
    finally:
        if runner is not None:
            runner.unlink(missing_ok=True)

    log = (finished.stdout or "") + (finished.stderr or "")
    if out_path.exists():
        size = out_path.stat().st_size / 1024
        return MaxResult(True, out_path,
                         f"فایل مکس ساخته شد ({size:.0f} کیلوبایت)", log,
                         script=script)

    return MaxResult(False, None,
                     "3ds Max اجرا شد ولی فایلی ننوشت.\n"
                     f"اسکریپت {script.name} را دستی داخل مکس اجرا کنید. "
                     "متن کامل در گزارش زیر است.",
                     log, script=script)
