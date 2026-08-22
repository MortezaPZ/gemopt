"""The desktop application.

Everything the project can do, without a command line.  The client picks a
reference picture, says what the stone is made of and how many facets it
should have, presses one button and watches the search run; the result appears
beside the reference and can be sent straight to 3ds Max.

Built on tkinter because it ships with Python -- there is nothing to install
beyond what the engine already needs, and it starts by double-clicking a file
on Windows.

The search runs in a worker process, not in the interface.  A long
optimisation would otherwise freeze the window, and a frozen window is
indistinguishable from a crashed one to whoever is watching it.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import queue
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path

from .precision import DECIMALS, fmt, fmt_deg

# Persian labels, kept together so the wording can be changed in one place.
FA = {
    "title": "بهینه‌ساز تراش سنگ",
    "reference": "تصویر مرجع",
    "browse": "انتخاب فایل…",
    "target_kind": "هدف",
    "layout": "چیدمان عکس",
    "layout_single": "یک نما",
    "layout_pair": "دو نما کنار هم",
    "which_view": "این نما کدام است؟",
    "view_crown": "از بالا (تاج)",
    "view_pavilion": "از پایین (پاویون)",
    "target_photo": "عکس مرجع",
    "target_white": "بیشترین سفیدی",
    "target_mirror": "آینه",
    "material": "ماده",
    "custom_ior": "ضریب شکست دلخواه",
    "shape": "شکل",
    "shape_gem": "سنگ (برلیان)",
    "shape_vessel": "ظرف (لیوان / پارچ)",
    "facets": "تعداد ستون (فست دور تا دور)",
    "girdle_multiple": "نیم‌صفحه به ازای هر فست",
    "crown_rows": "ردیف اضافه — تاج",
    "pavilion_rows": "ردیف اضافه — پاویون",
    "effort": "دقت جستجو",
    # Measured on eight cores: glass is the lower figure, diamond the upper,
    # because more light survives to bounce again inside it.
    "effort_fast": "سریع  (۷ تا ۱۰ دقیقه)",
    "effort_normal": "متعادل  (۱۳ تا ۲۰ دقیقه)",
    "effort_deep": "دقیق  (۲۲ تا ۳۲ دقیقه)",
    "start": "شروع بهینه‌سازی",
    "stop": "توقف",
    "export": "خروجی برای 3ds Max",
    "open_folder": "باز کردن پوشه‌ی نتایج",
    "ready": "آماده",
    "hearts": "از پایین (پاویون)",
    "arrows": "از بالا (تاج)",
    "ref_col": "مرجع",
    "out_col": "خروجی",
    "similarity": "شباهت",
    "light": "بازگشت نور",
    "leak": "نشت",
    "results": "نتیجه",
    "pick_first": "اول یک تصویر مرجع انتخاب کنید",
    "running": "در حال جستجو…",
    "done": "تمام شد",
    "stopped": "متوقف شد",
    "failed": "خطا",
    "generation": "نسل",
    "exported": "ذخیره شد",
    "degraded": "حافظه کم آمد — ادامه روی یک هسته، کندتر ولی مطمئن",
    "making_max": "در حال ساخت فایل مکس…",
    "max_note": "فایل مکس هم ساخته شد",
    "max_ready": "فایل مکس ساخته شد",
    "no_max": "3ds Max روی این سیستم نیست — اسکریپت ساخت ‎.max کنار فایل‌هاست",
    "size": "اندازه‌ی سنگ",
    "diameter": "قطر کمربند (میلی‌متر)",
    "diameter_note": "اندازه‌ی واقعی مدل در مکس. مدل با همین قطر ساخته می‌شود.",
    "bad_diameter": "قطر باید عددی بین ۰٫۱ تا ۱۰۰ میلی‌متر باشد",
    "polishing": "تنظیم دقیق نتیجه…",
    "pointed_culet": "نوک تیز (بدون صفحه در پایین)",
    "pointed_culet_note": "بردارید تا پایین سنگ یک صفحه‌ی کوچک صاف داشته باشد.",
    "flat_girdle": "کمربند صاف (یکنواخت دور تا دور)",
    "flat_girdle_note": "بدون آن، کمربند زیر فست‌های اصلی نازک‌تر و بینشان ضخیم‌تر می‌شود.",
    "table_pct": "نسبت تیبل به قطر کمربند (٪)",
    "table_auto": "برنامه خودش انتخاب کند",
    "table_note": "درصدِ قطر تیبل نسبت به قطر کمربند. تیک را بردارید تا عدد خودتان ثابت بماند.",
    "bad_table": "نسبت تیبل باید عددی بین ۴۵ تا ۶۸ درصد باشد",
}

MATERIALS_FA = [
    ("شیشه (۱٫۵۲)", "glass", 1.52),
    ("کریستال سربی (۱٫۵۴)", "lead_crystal", 1.545),
    ("زیرکونیا (۲٫۲۰)", None, 2.20),
    ("الماس (۲٫۴۲)", "diamond", 2.4175),
    ("دلخواه", None, None),
]

EFFORT = {
    "fast": dict(generations=8, popsize=8),
    "normal": dict(generations=16, popsize=10),
    "deep": dict(generations=26, popsize=14),
}


REQUIRED = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("PIL", "pillow"),
)
"""Modules the app cannot start without, with the name pip installs them by."""


def _worker_count() -> int:
    """How many candidates to evaluate at once.

    One core is left for the interface so the window keeps repainting.  The
    real limit is usually memory rather than cores: every worker carries its
    own numpy, scipy and copy of the problem, roughly a third of a gigabyte,
    and a machine that also has 3ds Max open runs out long before it runs out
    of threads.  When that happens Windows kills a worker outright and the
    pool reports it as a BrokenProcessPool, which reads as a crash.
    """
    import os

    cores = max(1, (os.cpu_count() or 2) - 1)
    try:
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        free_gb = status.ullAvailPhys / (1024 ** 3)
        cores = min(cores, max(1, int(free_gb / 0.45)))
    except Exception:
        pass                                   # not Windows, or no way to ask
    return cores


def _make_dpi_aware() -> None:
    """Tell Windows we scale ourselves.

    Without this the system stretches the window bitmap on a high-DPI screen
    and every label comes out soft.  Declaring awareness costs one call and
    makes the text sharp; anywhere that is not Windows it does nothing.
    """
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except AttributeError:                     # older Windows
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass                                       # not Windows, or not allowed


@dataclass
class Job:
    """Everything the worker needs; must be picklable to cross the process."""

    photo: str
    single: bool
    view: str
    material: str | None
    ior: float | None
    target_kind: str
    n_base: int
    girdle_multiple: int
    extra_crown_rows: int
    extra_pavilion_rows: int
    generations: int
    popsize: int
    workers: int
    outdir: str
    # The search works on a unit stone -- radius 1 -- because the optics do not
    # care how big it is.  The exported model very much does: at radius 1 the
    # stone arrives in 3ds Max two units across, which next to a scene in
    # millimetres is a speck you cannot see and would swear had not imported.
    diameter_mm: float = 6.5
    polish: bool = True
    # A straight girdle rather than the hand-cut scallop, and an optional
    # locked table: both are the client's own requests, so both default to
    # what he asked for while staying switchable.
    flat_girdle: bool = True
    pointed_culet: bool = True          # no flat plane at the bottom
    table_pct: float | None = None      # None = let the search choose it


# --------------------------------------------------------------------------
# the worker -- runs in its own process so the window stays alive
# --------------------------------------------------------------------------


_EVAL: dict = {}


def _eval_init(photo: str, single: bool, view: str, material_name, ior,
               psf_sigma: float, n_base: int, girdle_multiple: int,
               extra_crown_rows: int, extra_pavilion_rows: int,
               max_bounces: int, flat_girdle: bool = True,
               table_pct: float | None = None,
               pointed_culet: bool = True) -> None:
    """Set one pool worker up with its own copy of the problem.

    The reference photo is parsed once per worker rather than once per
    candidate; everything after that is pure arithmetic on arrays it already
    holds.
    """
    import os

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"

    from .geometry import CutParams
    from .inverse import InverseProblem, load_reference
    from .optics import MATERIALS, Material
    from .params import SearchSpace

    material = (Material("custom", n_d=ior, dispersion=0.02)
                if material_name is None else MATERIALS[material_name])
    views = load_reference(photo, single=single, view=view)
    _EVAL["problem"] = InverseProblem(views=views, material=material,
                                      psf_sigma=psf_sigma, max_bounces=max_bounces)
    # Must match the parent's space exactly: the search vector is positional,
    # so a space with one more free parameter here than there would silently
    # read every angle out of the wrong slot.
    space = SearchSpace(base=CutParams(
        n_base=n_base, girdle_multiple=girdle_multiple,
        extra_crown_rows=extra_crown_rows,
        extra_pavilion_rows=extra_pavilion_rows,
        flat_girdle=flat_girdle))
    if table_pct is not None:
        space.lock("table", table_pct / 100.0)
    if pointed_culet:
        space.lock("culet", 0.0)
    else:
        space.enable("culet")
    _EVAL["space"] = space


def _eval_one(vector) -> float:
    import numpy as np

    params = _EVAL["space"].to_params(np.asarray(vector))
    return -_EVAL["problem"].score(params)[0]


def _worker(job: Job, progress: mp.Queue, stop: mp.Event) -> None:
    """Run one optimisation, reporting progress and honouring the stop flag."""
    import os

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"

    try:
        import numpy as np
        from scipy.optimize import differential_evolution

        from .geometry import CutParams, build_stone
        from .inverse import InverseProblem, load_reference
        from .maxbridge import write_gem, write_vertex_csv
        from .metrics import (Target, achievable_ceiling, mirror_target,
                              relative_match, uniform_target)
        from .optics import MATERIALS, Geometry, Material
        from .params import SearchSpace
        from .scope import Camera, render, save_png

        out = Path(job.outdir)
        out.mkdir(parents=True, exist_ok=True)

        material = (Material("custom", n_d=job.ior, dispersion=0.02)
                    if job.material is None else MATERIALS[job.material])

        views = load_reference(job.photo, single=job.single, view=job.view)
        progress.put(("status", "views loaded"))

        problem = InverseProblem(views=views, material=material,
                                 psf_sigma=2.5, max_bounces=10)
        space = SearchSpace(base=CutParams(
            n_base=job.n_base, girdle_multiple=job.girdle_multiple,
            extra_crown_rows=job.extra_crown_rows,
            extra_pavilion_rows=job.extra_pavilion_rows,
            flat_girdle=job.flat_girdle))
        if job.table_pct is not None:
            space.lock("table", job.table_pct / 100.0)
        if job.pointed_culet:
            space.lock("culet", 0.0)
        else:
            space.enable("culet")
        low, high = space.bounds()
        bounds = list(zip(low, high))
        caps = {v.view: achievable_ceiling(Target(polar=v.target, n_fold=8,
                                                  name=v.view)) for v in views}

        state = {"gen": 0, "best": 0.0}
        init_args = (job.photo, job.single, job.view, job.material, job.ior,
                     2.5, job.n_base, job.girdle_multiple,
                     job.extra_crown_rows, job.extra_pavilion_rows, 10,
                     job.flat_girdle, job.table_pct, job.pointed_culet)

        # Every candidate in a generation is independent, so they go out to a
        # pool.  Run serially this is a single core out of eight, which turns
        # a twenty minute search into most of an afternoon.
        from concurrent.futures import BrokenExecutor, ProcessPoolExecutor

        pool_state = {"pool": None}

        def evaluate_all(vectors):
            """Score a generation, in parallel if the machine can take it.

            If a worker is killed -- almost always memory -- the pool is
            unusable from then on.  Rather than surface that as a crash after
            the client has already waited, the run drops to doing the work in
            this process: slower, but it finishes and produces a stone.
            """
            pool = pool_state["pool"]
            if pool is not None:
                try:
                    return list(pool.map(_eval_one, vectors, chunksize=1))
                except (BrokenExecutor, OSError):
                    progress.put(("degraded",))
                    try:
                        pool.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                    pool_state["pool"] = None
            _eval_init(*init_args)
            return [_eval_one(v) for v in vectors]

        def batch(population):
            if stop.is_set():
                raise KeyboardInterrupt
            values = evaluate_all(list(population.T))
            state["gen"] += 1
            state["best"] = max(state["best"], -min(values))
            progress.put(("generation", state["gen"], job.generations,
                          state["best"]))
            return np.asarray(values)

        # Not a `with` block: leaving one calls shutdown a second time, and on
        # a pool that has already broken that raises on the way out and buries
        # the real result.
        try:
            pool_state["pool"] = ProcessPoolExecutor(
                max_workers=max(1, job.workers), initializer=_eval_init,
                initargs=init_args)
        except OSError:
            progress.put(("degraded",))          # could not start one at all

        try:
            result = differential_evolution(
                batch, bounds, seed=0, popsize=job.popsize,
                maxiter=job.generations, tol=0.0, mutation=(0.4, 1.0),
                recombination=0.85, polish=False, init="sobol",
                vectorized=True, x0=space.start(),
            )
            best_vector = result.x

            # Differential evolution stops where its population can no longer
            # tell two candidates apart, which on a stone this size is a few
            # hundredths of a degree -- and a few hundredths of a degree is a
            # different cut, not a rounding detail.  A downhill run from the
            # winner costs about one generation and takes the answer down to
            # where the score itself stops moving.
            if job.polish and not stop.is_set():
                from scipy.optimize import minimize

                seen = {"n": 0}

                def refine_one(vector):
                    if stop.is_set():
                        raise KeyboardInterrupt
                    seen["n"] += 1
                    if seen["n"] % 20 == 0:
                        progress.put(("polish", seen["n"]))
                    return float(evaluate_all([np.asarray(vector, dtype=float)])[0])

                progress.put(("polish", 0))
                refined = minimize(refine_one, best_vector, method="Nelder-Mead",
                                   bounds=bounds,
                                   options={"xatol": 1e-5, "fatol": 1e-9,
                                            "maxfev": 600, "adaptive": True})
                if float(refined.fun) <= float(result.fun):
                    best_vector = np.clip(refined.x, low, high)
        except KeyboardInterrupt:
            progress.put(("stopped",))
            return
        finally:
            if pool_state["pool"] is not None:
                try:
                    pool_state["pool"].shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass

        params = space.to_params(best_vector)
        score, per_view, profile = problem.score(params)
        instrument = {v.view: problem.environment(profile, i)
                      for i, v in enumerate(problem.views)}

        mesh = build_stone(params)
        geom = Geometry(mesh)
        stats = {}
        for view in [v.view for v in views]:
            shot = render(mesh, material, Camera(resolution=420, view=view, samples=4),
                          instrument[view], geometry=geom)
            reference = {v.view: v.rgb for v in views}[view]
            save_png(_match(shot.image, reference), out / f"out_{view}.png")
            save_png(reference, out / f"ref_{view}.png", gamma=1.0)
            stats[view] = {"light_return": shot.light_return, "leakage": shot.leakage}

        shares = {v.view: relative_match(s, caps[v.view])
                  for v, s in zip(problem.views, per_view)}

        # build_stone works at girdle radius 1, so half the wanted diameter is
        # the factor that turns the solved shape into a real stone.
        scale = float(job.diameter_mm) / 2.0
        write_gem(out / "stone.gem", mesh, params=params, scale=scale,
                  ior=material.n_d)
        write_vertex_csv(out / "stone_vertices.csv", mesh, scale=scale)
        mesh.scale(scale).to_obj(out / "stone.obj")

        # The .max belongs to the output, not to a separate button: the client
        # asked for it to be in the folder when the run ends.  Only 3ds Max can
        # write that format, so this drives it head-lessly.  A machine without
        # Max simply does not get the file, and the run is no less finished.
        from .precision import DECIMALS, round_floats
        from .tomax import gem_to_max

        progress.put(("status", "max"))
        ies = Path("assets") / "075 SPARK 5000 840 30D.ies"
        max_result = gem_to_max(out / "stone.gem",
                                ies=str(ies) if ies.exists() else None,
                                material="vray")
        summary = {"params": params.to_dict(), "derived": params.derived(),
                   "score": score, "of_achievable": shares, "stats": stats,
                   "material": material.name, "ior": material.n_d,
                   "diameter_mm": float(job.diameter_mm), "scale": scale,
                   "decimals": DECIMALS,
                   "flat_girdle": bool(job.flat_girdle),
                   "pointed_culet": bool(job.pointed_culet),
                   "table_locked": job.table_pct is not None,
                   "max_file": str(max_result.path) if max_result.ok else None,
                   "max_script": str(max_result.script) if max_result.script else None,
                   "max_message": max_result.message}
        (out / "result.json").write_text(
            json.dumps(round_floats(summary), indent=2, ensure_ascii=False),
            encoding="utf-8")
        progress.put(("done", summary))

    except Exception:                                    # pragma: no cover
        progress.put(("failed", traceback.format_exc()))


def _match(image, reference):
    """Put the render on the photo's brightness scale (see compare_options)."""
    import numpy as np

    lum = np.array([0.2126, 0.7152, 0.0722])
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    inside = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (0.92 * h / 2) ** 2
    rh, rw = reference.shape[:2]
    ry, rx = np.mgrid[0:rh, 0:rw]
    ref_in = ((ry - rh / 2) ** 2 + (rx - rw / 2) ** 2) < (0.92 * rh / 2) ** 2
    src, ref = image @ lum, reference @ lum
    if src[inside].std() < 1e-6:
        return image
    gain = ref[ref_in].std() / src[inside].std()
    return np.clip(image * gain + (ref[ref_in].mean() - gain * src[inside].mean()), 0, 1)


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------


class App:
    """The whole interface.  One window, no menus, nothing to learn."""

    BG = "#f4f6fa"
    CARD = "#ffffff"
    INK = "#12305c"
    MUTED = "#6b7488"
    ACCENT = "#12305c"
    GOOD = "#14603a"

    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        _make_dpi_aware()

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title(FA["title"])
        self.root.configure(bg=self.BG)
        self._centre(1180, 800)

        self.photo_path: Path | None = None
        self.outdir = Path("results") / "app"
        self.process: mp.Process | None = None
        self.queue: mp.Queue | None = None
        self.stop_flag = None
        self._images: dict[str, object] = {}
        self.result: dict | None = None

        self._style()
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(120, self._pump)

    def _centre(self, width: int, height: int) -> None:
        """Open centred, and never larger than the screen it opens on.

        Asking for a window taller than the display pushes its bottom under
        the taskbar, which is where the start button used to end up on a
        smaller monitor.  Fitting to the screen and letting the control column
        scroll keeps everything reachable at any resolution.
        """
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(width, screen_w - 40)
        height = min(height, screen_h - 90)          # room for the taskbar
        self.root.minsize(min(940, width), min(560, height))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 3)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    # -- chrome ------------------------------------------------------------

    def _style(self) -> None:
        s = self.ttk.Style()
        try:
            s.theme_use("clam")
        except Exception:                                # pragma: no cover
            pass
        s.configure(".", background=self.BG, font=("Tahoma", 10))
        s.configure("Card.TFrame", background=self.CARD, relief="flat")
        s.configure("TLabel", background=self.BG, foreground="#1b2230")
        s.configure("Card.TLabel", background=self.CARD)
        s.configure("Card.TCheckbutton", background=self.BG, font=("Tahoma", 8))
        s.configure("H1.TLabel", background=self.BG, foreground=self.INK,
                    font=("Tahoma", 15, "bold"))
        s.configure("Muted.TLabel", background=self.CARD, foreground=self.MUTED,
                    font=("Tahoma", 8))
        s.configure("Big.TButton", font=("Tahoma", 11, "bold"), padding=9)
        s.configure("TButton", padding=5)
        s.configure("Horizontal.TProgressbar", background=self.ACCENT,
                    troughcolor="#e3e8f0", borderwidth=0, thickness=8)

    def _card(self, parent, title: str):
        wrap = self.ttk.Frame(parent, style="Card.TFrame", padding=12)
        head = self.ttk.Label(wrap, text=title, style="Card.TLabel",
                              font=("Tahoma", 10, "bold"), foreground=self.INK)
        head.pack(anchor="e", pady=(0, 8))
        return wrap

    def _build(self) -> None:
        tk, ttk = self.tk, self.ttk
        root = self.root

        ttk.Label(root, text=FA["title"], style="H1.TLabel").pack(
            anchor="e", padx=18, pady=(14, 2))
        ttk.Label(root, text="تصویر مرجع را بدهید، ماده و تعداد فست را انتخاب کنید، و "
                             "بگذارید سیستم بهترین تراش را پیدا کند",
                  background=self.BG, foreground=self.MUTED).pack(anchor="e", padx=18)

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, padx=14, pady=12)
        body.columnconfigure(0, weight=1, minsize=560)   # results, on the left
        body.columnconfigure(1, weight=0, minsize=340)   # controls, on the right
        body.rowconfigure(0, weight=1)

        side = self._scrollable(body)
        self._build_controls(side)

        panel = self._card(body, FA["results"])
        panel.grid(row=0, column=0, sticky="nsew")
        self._build_results(panel)

    def _scrollable(self, body):
        """A control column that can be scrolled.

        The window is sized to the screen, but a short screen still leaves
        controls below the fold -- and a button nobody can reach is the same
        as a button that does not exist.  The wheel is bound while the pointer
        is over the column so it scrolls without having to grab the bar.
        """
        tk, ttk = self.tk, self.ttk

        holder = ttk.Frame(body)
        holder.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)

        canvas = tk.Canvas(holder, bg=self.BG, highlightthickness=0, width=330)
        bar = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        window = canvas.create_window((0, 0), window=inner, anchor="nw", width=330)
        canvas.configure(yscrollcommand=bar.set)
        canvas.grid(row=0, column=0, sticky="nsew")

        def resize(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # the bar only earns its space when there is something to scroll
            needed = inner.winfo_reqheight() > canvas.winfo_height()
            if needed and not bar.winfo_ismapped():
                bar.grid(row=0, column=1, sticky="ns")
            elif not needed and bar.winfo_ismapped():
                bar.grid_remove()

        inner.bind("<Configure>", resize)
        canvas.bind("<Configure>", lambda e: (canvas.itemconfig(window, width=e.width),
                                              resize()))

        def wheel(event):
            if inner.winfo_reqheight() > canvas.winfo_height():
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", wheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))
        return inner

    def _build_controls(self, parent) -> None:
        tk, ttk = self.tk, self.ttk

        # The action block sits at the top.  On a short screen the column
        # scrolls, and whatever is at the bottom is the thing nobody finds --
        # so the button that actually starts the work goes first, above the
        # settings it depends on.
        actions = ttk.Frame(parent)
        actions.pack(fill="x", pady=(0, 4))
        self.start_btn = ttk.Button(actions, text=FA["start"], style="Big.TButton",
                                    command=self._start)
        self.start_btn.pack(fill="x")
        self.stop_btn = ttk.Button(actions, text=FA["stop"], command=self._stop,
                                   state="disabled")
        self.stop_btn.pack(fill="x", pady=(6, 0))

        self.progress = ttk.Progressbar(parent, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(8, 3))
        self.status_var = tk.StringVar(value=FA["ready"])
        ttk.Label(parent, textvariable=self.status_var, foreground=self.MUTED,
                  anchor="e").pack(fill="x")
        self.export_btn = ttk.Button(parent, text=FA["open_folder"],
                                     command=self._open_folder, state="disabled")
        self.export_btn.pack(fill="x", pady=(6, 10))

        card = self._card(parent, FA["reference"])
        card.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(card, style="Card.TFrame")
        row.pack(fill="x")
        self.path_var = tk.StringVar(value="—")
        ttk.Button(row, text=FA["browse"], command=self._choose).pack(side="right")
        ttk.Label(row, textvariable=self.path_var, style="Muted.TLabel",
                  anchor="e").pack(side="right", fill="x", expand=True, padx=8)

        self.layout_var = tk.StringVar(value="single")
        self.view_var = tk.StringVar(value="crown")
        layout = ttk.Frame(card, style="Card.TFrame")
        layout.pack(fill="x", pady=(8, 0))
        for label, value in ((FA["layout_single"], "single"),
                             (FA["layout_pair"], "pair")):
            ttk.Radiobutton(layout, text=label, value=value,
                            variable=self.layout_var,
                            command=self._layout_changed).pack(side="right",
                                                               padx=(0, 12))
        self.view_frame = ttk.Frame(card, style="Card.TFrame")
        ttk.Label(self.view_frame, text=FA["which_view"],
                  style="Card.TLabel").pack(anchor="e")
        for label, value in ((FA["view_crown"], "crown"),
                             (FA["view_pavilion"], "pavilion")):
            ttk.Radiobutton(self.view_frame, text=label, value=value,
                            variable=self.view_var,
                            command=self._reload_reference).pack(anchor="e")

        self.target_var = tk.StringVar(value="photo")
        kinds = ttk.Frame(card, style="Card.TFrame")
        kinds.pack(fill="x", pady=(8, 0))
        for label, value in ((FA["target_photo"], "photo"),
                             (FA["target_white"], "white"),
                             (FA["target_mirror"], "mirror")):
            ttk.Radiobutton(kinds, text=label, value=value,
                            variable=self.target_var).pack(side="right", padx=(0, 12))

        card = self._card(parent, FA["material"])
        card.pack(fill="x", pady=(0, 10))
        self.material_var = tk.StringVar(value=MATERIALS_FA[0][0])
        combo = ttk.Combobox(card, textvariable=self.material_var, state="readonly",
                             values=[m[0] for m in MATERIALS_FA], justify="right")
        combo.pack(fill="x")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._toggle_ior())
        self.ior_frame = ttk.Frame(card, style="Card.TFrame")
        self.ior_var = tk.StringVar(value="1.55")
        ttk.Entry(self.ior_frame, textvariable=self.ior_var, width=8,
                  justify="center").pack(side="right")
        ttk.Label(self.ior_frame, text=FA["custom_ior"],
                  style="Card.TLabel").pack(side="right", padx=6)

        # Without this the model leaves at radius 1 and lands in Max two units
        # across, which reads as "the 3D output is broken" rather than "the
        # stone is 2mm".
        card = self._card(parent, FA["size"])
        card.pack(fill="x", pady=(0, 10))
        size_row = ttk.Frame(card, style="Card.TFrame")
        size_row.pack(fill="x")
        self.diameter_var = tk.StringVar(value="6.5")
        ttk.Entry(size_row, textvariable=self.diameter_var, width=10,
                  justify="center").pack(side="right")
        ttk.Label(size_row, text=FA["diameter"],
                  style="Card.TLabel").pack(side="right", padx=6)
        ttk.Label(card, text=FA["diameter_note"],
                  style="Muted.TLabel").pack(anchor="e", pady=(6, 0))

        # The client asked for this ratio by name -- table against girdle
        # diameter -- and wanted to be able to move it, so it gets a box of
        # its own rather than staying buried in the search.
        table_row = ttk.Frame(card, style="Card.TFrame")
        table_row.pack(fill="x", pady=(10, 0))
        self.table_var = tk.StringVar(value="53")
        self.table_entry = ttk.Entry(table_row, textvariable=self.table_var,
                                     width=10, justify="center")
        self.table_entry.pack(side="right")
        ttk.Label(table_row, text=FA["table_pct"],
                  style="Card.TLabel").pack(side="right", padx=6)

        self.table_auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(card, text=FA["table_auto"], variable=self.table_auto_var,
                        style="Card.TCheckbutton",
                        command=self._toggle_table).pack(anchor="e", pady=(6, 0))
        ttk.Label(card, text=FA["table_note"],
                  style="Muted.TLabel").pack(anchor="e", pady=(4, 0))
        self._toggle_table()

        card = self._card(parent, FA["shape"])
        card.pack(fill="x", pady=(0, 10))
        grid = ttk.Frame(card, style="Card.TFrame")
        grid.pack(fill="x")
        self.facets_var = tk.IntVar(value=8)
        self.girdle_var = tk.IntVar(value=2)
        self.crown_rows_var = tk.IntVar(value=0)
        self.pav_rows_var = tk.IntVar(value=0)
        self._spin(grid, FA["facets"], self.facets_var, 4, 24, 0)
        self._spin(grid, FA["girdle_multiple"], self.girdle_var, 1, 4, 1)
        self._spin(grid, FA["crown_rows"], self.crown_rows_var, 0, 4, 2)
        self._spin(grid, FA["pavilion_rows"], self.pav_rows_var, 0, 4, 3)
        self.facet_note = ttk.Label(card, text="", style="Muted.TLabel")
        self.facet_note.pack(anchor="e", pady=(6, 0))
        for var in (self.facets_var, self.girdle_var,
                    self.crown_rows_var, self.pav_rows_var):
            var.trace_add("write", lambda *_: self._update_facet_note())
        self._update_facet_note()

        self.pointed_culet_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(card, text=FA["pointed_culet"],
                        variable=self.pointed_culet_var,
                        style="Card.TCheckbutton").pack(anchor="e", pady=(8, 0))
        ttk.Label(card, text=FA["pointed_culet_note"],
                  style="Muted.TLabel").pack(anchor="e", pady=(4, 0))

        self.flat_girdle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(card, text=FA["flat_girdle"],
                        variable=self.flat_girdle_var,
                        style="Card.TCheckbutton").pack(anchor="e", pady=(8, 0))
        ttk.Label(card, text=FA["flat_girdle_note"],
                  style="Muted.TLabel").pack(anchor="e", pady=(4, 0))

        card = self._card(parent, FA["effort"])
        card.pack(fill="x", pady=(0, 10))
        self.effort_var = tk.StringVar(value="normal")
        for label, value in ((FA["effort_fast"], "fast"),
                             (FA["effort_normal"], "normal"),
                             (FA["effort_deep"], "deep")):
            ttk.Radiobutton(card, text=label, value=value,
                            variable=self.effort_var).pack(anchor="e")


    def _spin(self, parent, label: str, var, lo: int, hi: int, row: int) -> None:
        ttk = self.ttk
        ttk.Label(parent, text=label, style="Card.TLabel").grid(
            row=row, column=1, sticky="e", pady=3)
        ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=6,
                    justify="center").grid(row=row, column=0, sticky="w", pady=3)
        parent.columnconfigure(1, weight=1)

    def _build_results(self, parent) -> None:
        ttk = self.ttk
        grid = ttk.Frame(parent, style="Card.TFrame")
        grid.pack(fill="both", expand=True)

        for col, text in ((1, FA["ref_col"]), (2, FA["out_col"])):
            ttk.Label(grid, text=text, style="Card.TLabel",
                      font=("Tahoma", 9, "bold"), foreground=self.INK).grid(
                row=0, column=col, pady=(0, 6))

        self.canvases = {}
        self.row_labels = {}
        for row, (view, label) in enumerate(
                (("pavilion", FA["hearts"]), ("crown", FA["arrows"])), start=1):
            tag = ttk.Label(grid, text=label, style="Card.TLabel")
            tag.grid(row=row, column=3, padx=(10, 4), sticky="e")
            self.row_labels[view] = tag
            for col, kind in ((1, "ref"), (2, "out")):
                canvas = self.tk.Canvas(grid, width=232, height=232, bg="#101216",
                                        highlightthickness=1,
                                        highlightbackground="#d8dde6")
                canvas.grid(row=row, column=col, padx=5, pady=5)
                self.canvases[f"{kind}_{view}"] = canvas
        self._show_rows()

        self.metrics_var = self.tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.metrics_var, style="Card.TLabel",
                  foreground=self.GOOD, font=("Tahoma", 10, "bold"),
                  justify="right").pack(anchor="e", pady=(8, 0))
        self.params_var = self.tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.params_var, style="Muted.TLabel",
                  justify="right").pack(anchor="e")

    # -- behaviour ---------------------------------------------------------

    def _layout_changed(self) -> None:
        """A single photo needs to say which face it shows; a pair does not."""
        if self.layout_var.get() == "single":
            self.view_frame.pack(fill="x", pady=(6, 0))
        else:
            self.view_frame.pack_forget()
        self._reload_reference()

    def _reload_reference(self) -> None:
        if self.photo_path is not None:
            self._show_reference()

    def _show_rows(self) -> None:
        """Show a row per face we actually have a reference for.

        With one photograph there is only one face to compare, and an empty
        second row invites the question of what is missing from it.
        """
        single = self.layout_var.get() == "single"
        wanted = {self.view_var.get()} if single else {"crown", "pavilion"}
        for view, tag in self.row_labels.items():
            widgets = [tag] + [self.canvases[f"{k}_{view}"] for k in ("ref", "out")]
            if view in wanted:
                for w in widgets:
                    w.grid()
            else:
                for w in widgets:
                    w.grid_remove()

    def _blank_unused(self) -> None:
        self._show_rows()

    def _toggle_table(self) -> None:
        """Grey the table box out while the search owns that number.

        Left editable it would look like an input that is being ignored --
        which is exactly the complaint that started this.
        """
        self.table_entry.config(
            state="disabled" if self.table_auto_var.get() else "normal")

    def _toggle_ior(self) -> None:
        custom = self.material_var.get() == MATERIALS_FA[-1][0]
        if custom:
            self.ior_frame.pack(fill="x", pady=(8, 0))
        else:
            self.ior_frame.pack_forget()

    def _update_facet_note(self) -> None:
        """Live preview of the count, straight from the geometry itself.

        Asking the real builder rather than repeating its arithmetic here
        means the preview cannot drift away from what actually gets cut.
        """
        from .geometry import CutParams, facet_count

        try:
            params = CutParams(n_base=int(self.facets_var.get()),
                               girdle_multiple=int(self.girdle_var.get()),
                               extra_crown_rows=int(self.crown_rows_var.get()),
                               extra_pavilion_rows=int(self.pav_rows_var.get()))
        except Exception:
            return
        if params.validate():
            self.facet_note.config(text="این ترکیب ساخته نمی‌شود")
            return
        n, m = params.n_base, params.girdle_multiple
        rows = 3 + params.extra_crown_rows + 2 + params.extra_pavilion_rows
        self.facet_note.config(
            text=f"{n} ستون · {rows} ردیف · {m * n} نیم‌صفحه در هر طرف · "
                 f"مجموع {facet_count(params)} فست")

    def _choose(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title=FA["reference"],
            filetypes=[("تصویر", "*.jpg *.jpeg *.png *.bmp"), ("همه", "*.*")])
        if path:
            self.photo_path = Path(path)
            self.path_var.set(self.photo_path.name)
            self._show_reference()

    def _show_reference(self) -> None:
        try:
            from .inverse import load_reference

            for view in load_reference(str(self.photo_path),
                                       single=self.layout_var.get() == "single",
                                       view=self.view_var.get()):
                self._draw(f"ref_{view.view}", view.rgb)
            self._blank_unused()
            self.status_var.set(FA["ready"])
        except Exception as exc:
            self.status_var.set(f"{FA['failed']}: {exc}")

    def _draw(self, key: str, array) -> None:
        import numpy as np
        from PIL import Image, ImageTk

        canvas = self.canvases[key]
        size = int(canvas["width"])
        img = Image.fromarray((np.clip(array, 0, 1) * 255).astype("uint8"))
        img = img.resize((size, size), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._images[key] = photo          # keep a reference or tkinter drops it
        canvas.delete("all")
        canvas.create_image(size // 2, size // 2, image=photo)

    def _start(self) -> None:
        if self.photo_path is None:
            self.status_var.set(FA["pick_first"])
            return
        if self.process is not None and self.process.is_alive():
            return

        label = self.material_var.get()
        name, ior = next((m[1], m[2]) for m in MATERIALS_FA if m[0] == label)
        if name is None and ior is None:
            try:
                ior = float(self.ior_var.get())
            except ValueError:
                self.status_var.set("ضریب شکست معتبر نیست")
                return

        try:
            diameter = float(str(self.diameter_var.get()).strip().replace("٫", "."))
        except ValueError:
            diameter = 0.0
        if not 0.1 <= diameter <= 100.0:
            self.status_var.set(FA["bad_diameter"])
            return

        table_pct = None
        if not self.table_auto_var.get():
            try:
                table_pct = float(str(self.table_var.get()).strip()
                                  .replace("٫", ".").replace("٪", ""))
            except ValueError:
                table_pct = 0.0
            # the bounds the search itself uses, so a locked value cannot ask
            # for a stone the geometry would reject
            if not 45.0 <= table_pct <= 68.0:
                self.status_var.set(FA["bad_table"])
                return

        effort = EFFORT[self.effort_var.get()]
        self.outdir = Path("results") / "app"
        job = Job(photo=str(self.photo_path),
                  single=self.layout_var.get() == "single",
                  view=self.view_var.get(),
                  material=name, ior=ior,
                  target_kind=self.target_var.get(),
                  n_base=int(self.facets_var.get()),
                  girdle_multiple=int(self.girdle_var.get()),
                  extra_crown_rows=int(self.crown_rows_var.get()),
                  extra_pavilion_rows=int(self.pav_rows_var.get()),
                  generations=effort["generations"], popsize=effort["popsize"],
                  workers=_worker_count(), outdir=str(self.outdir),
                  diameter_mm=diameter,
                  flat_girdle=bool(self.flat_girdle_var.get()),
                  pointed_culet=bool(self.pointed_culet_var.get()),
                  table_pct=table_pct)

        ctx = mp.get_context("spawn")
        self.queue = ctx.Queue()
        self.stop_flag = ctx.Event()
        # Not daemonic: this process starts a pool of its own, and a daemonic
        # parent is not allowed children.  Closing the window kills it instead.
        self.process = ctx.Process(target=_worker,
                                   args=(job, self.queue, self.stop_flag))
        self.process.start()

        self.progress.config(value=0)
        self.status_var.set(FA["running"])
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

    def _stop(self) -> None:
        if self.stop_flag is not None:
            self.stop_flag.set()
        self.status_var.set(FA["stopped"])
        self.stop_btn.config(state="disabled")

    def _pump(self) -> None:
        """Drain the worker's messages.  Runs on the tk loop, never blocks."""
        if self.queue is not None:
            try:
                while True:
                    message = self.queue.get_nowait()
                    self._handle(message)
            except queue.Empty:
                pass
        self.root.after(120, self._pump)

    def _handle(self, message) -> None:
        kind = message[0]
        if kind == "status" and len(message) > 1 and message[1] == "max":
            self.status_var.set(FA["making_max"])
        elif kind == "generation":
            gen, total, best = message[1], message[2], message[3]
            self.progress.config(value=100.0 * gen / max(total, 1))
            self.status_var.set(f"{FA['generation']} {gen} از {total} — "
                                f"بهترین {best:.{DECIMALS}f}")
        elif kind == "polish":
            self.progress.config(value=100.0)
            self.status_var.set(FA["polishing"])
        elif kind == "done":
            self._finish(message[1])
        elif kind == "degraded":
            self.status_var.set(FA["degraded"])
        elif kind == "stopped":
            self.status_var.set(FA["stopped"])
            self._idle()
        elif kind == "failed":
            self.status_var.set(FA["failed"])
            self._show_error(message[1])
            self._idle()

    def _idle(self) -> None:
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _finish(self, summary: dict) -> None:
        import numpy as np
        from PIL import Image

        self.result = summary
        for view in ("pavilion", "crown"):
            path = self.outdir / f"out_{view}.png"
            if self.layout_var.get() == "single" and view != self.view_var.get():
                continue
            if path.exists():
                self._draw(f"out_{view}",
                           np.asarray(Image.open(path).convert("RGB"),
                                      dtype=float) / 255.0)

        shares = summary.get("of_achievable", {})
        stats = summary.get("stats", {})
        self.metrics_var.set(
            f"{FA['similarity']}:  {FA['hearts']} {shares.get('pavilion', 0):.0%}"
            f"   ·   {FA['arrows']} {shares.get('crown', 0):.0%}\n"
            f"{FA['light']}: {stats.get('crown', {}).get('light_return', 0):.0%}"
            f"   ·   {FA['leak']}: {stats.get('crown', {}).get('leakage', 0):.0%}")
        d = summary.get("derived", {})
        self.params_var.set(
            f"زاویه‌ی تاج {fmt_deg(d.get('crown_angle', 0))}  ·  "
            f"زاویه‌ی پاویون {fmt_deg(d.get('pavilion_angle', 0))}  ·  "
            f"table {fmt(d.get('table_pct', 0))}٪  ·  "
            f"عمق {fmt(d.get('total_depth_pct', 0))}٪  ·  "
            f"قطر {fmt(summary.get('diameter_mm', 0))} م‌م  ·  "
            f"{int(d.get('facet_count', 0))} فست")
        made_max = summary.get("max_file")
        self.status_var.set(f"{FA['done']} — {FA['max_note']}" if made_max
                            else f"{FA['done']} — {FA['no_max']}")
        self.export_btn.config(state="normal")
        self._idle()

    def _show_error(self, text: str) -> None:
        from tkinter import messagebox

        messagebox.showerror(FA["failed"], text[-1500:])

    def _open_folder(self) -> None:
        import subprocess
        import sys

        path = self.outdir.resolve()
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", str(path)])
        else:
            # WSL: hand the Windows-side path to Explorer
            win = str(path).replace("/mnt/c/", "C:\\").replace("/", "\\")
            subprocess.Popen(["explorer.exe", win])

    def _close(self) -> None:
        """Stop the search before the window goes, or it outlives the app."""
        if self.stop_flag is not None:
            self.stop_flag.set()
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=3)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    mp.freeze_support()
    App().run()


if __name__ == "__main__":
    main()
