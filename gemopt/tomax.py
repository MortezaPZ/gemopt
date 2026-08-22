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


def write_build_script(
    gem_path,
    out_path=None,
    material: str = "vray",
    ies: str | None = None,
    name: str = "gemopt_stone",
    script_path=None,
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
    lines.append(f"fileIn {_literal(maxscript_dir() / 'import_stone.ms')}")

    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path


def gem_to_max(
    gem_path,
    out_path=None,
    material: str = "vray",
    ies: str | None = None,
    name: str = "gemopt_stone",
    batch_exe: str | None = None,
    timeout: int = 420,
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
                                    ies=ies, name=name)
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
    # differently, so which one we got decides the command.
    if "batch" in exe.name.lower():
        command = [str(exe), str(script)]
        runner = None
    else:
        # 3dsmax.exe runs a script too, but never exits on its own, so it gets
        # a wrapper that quits when the work is done.  Without that the run
        # sits at the timeout with a window open and produces nothing.
        runner = script.with_name(script.stem + "_auto.ms")
        try:
            runner.write_text(
                f"fileIn {_literal(script)}\nquitMax #noPrompt\n", encoding="utf-8")
        except OSError as exc:
            return MaxResult(False, None, f"نوشتن اسکریپت موقت ممکن نشد: {exc}",
                             script=script)
        command = [str(exe), "-U", "MAXScript", str(runner), "-q", "-silent"]

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
