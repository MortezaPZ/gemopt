"""Command line entry points.

Everything the client needs to do can be done from a terminal without opening
Python:

    python -m gemopt template                  make the parameter spreadsheet
    python -m gemopt optimise --material glass run a search
    python -m gemopt render                    picture one set of parameters
    python -m gemopt score mystone.gem         measure a stone exported from Max
    python -m gemopt export best.json          write .gem / .obj / vertex CSV
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .evaluate import EvalConfig, evaluate, make_reference
from .geometry import CutParams, STANDARD_BRILLIANT, brilliant_for_n, build_stone
from .ies import load_ies
from .maxbridge import maxscript_dir, read_gem, read_obj, write_gem, write_vertex_csv
from .metrics import ScoreWeights, load_target
from .optics import MATERIALS, Material
from .optimize import optimise, optimise_over_facets
from .params import SearchSpace, bind_symmetry, load_space, write_template
from .pattern import layout_from_n
from .pattern_maxscript import write_pattern_ms
from .diagram import assert_n_fold_symmetric, draw_pattern, save_pattern_png
from .report import write_report, write_trials_csv
from .scope import HEARTS_ARROWS_SCOPE, IDEALSCOPE, Camera, IESEnvironment, render, save_png

SCOPES = {"hearts": HEARTS_ARROWS_SCOPE, "idealscope": IDEALSCOPE}


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------


def _material(name: str, ior: float | None, dispersion: float | None) -> Material:
    if name in MATERIALS:
        base = MATERIALS[name]
    else:
        try:
            base = Material("custom", n_d=float(name))
        except ValueError:
            raise SystemExit(
                f"unknown material '{name}'. Use one of {', '.join(MATERIALS)} "
                f"or give a refractive index as a number."
            )
    return Material(
        name=base.name,
        n_d=ior if ior is not None else base.n_d,
        dispersion=dispersion if dispersion is not None else base.dispersion,
        absorption=base.absorption,
    )


def _config(args) -> EvalConfig:
    environment = SCOPES[args.scope]
    if getattr(args, "ies", None):
        profile = load_ies(args.ies)
        environment = IESEnvironment(profile=profile)
        print(f"lighting: {profile.name or Path(args.ies).stem} "
              f"({profile.lumens:.0f} lm, {profile.beam_angle:.0f}deg beam)")

    weights = ScoreWeights()
    if getattr(args, "weight_pattern", None) is not None:
        weights.pattern = args.weight_pattern
    if getattr(args, "weight_return", None) is not None:
        weights.light_return = args.weight_return
    if getattr(args, "weight_leakage", None) is not None:
        weights.leakage = args.weight_leakage

    return EvalConfig(
        material=_material(args.material, args.ior, args.dispersion),
        environment=environment,
        view=args.view,
        resolution=args.resolution,
        samples=getattr(args, "samples", 1),
        weights=weights,
        n_fold=getattr(args, "facets", None) or 8,
    )


def _resolve_target(args, config: EvalConfig):
    if getattr(args, "target", None):
        target = load_target(args.target, n_fold=config.n_fold)
        print(f"target: {args.target}")
        if config.view == "both":
            # One photograph cannot stand in for both sides of the stone, so
            # say so rather than silently scoring the hearts against a
            # picture of the arrows.
            print("  note: --view both with a single photo compares BOTH views "
                  "against that one image; give --view crown or pavilion for a "
                  "photo of one side.")
        return target
    print("target: simulated ideal diamond (no reference photo given)")
    return make_reference(config)


def _params_from(path: str | None, n_base: int | None = None,
                 n_d: float | None = None) -> CutParams:
    params = STANDARD_BRILLIANT
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        params = CutParams.from_dict(data.get("parameters", data))
    if n_base is not None:
        rest = {k: v for k, v in params.to_dict().items() if k != "n_base"}
        if path is None and n_d is not None:
            rest.pop("pavilion_angle", None)
            rest.pop("crown_angle", None)
            params = brilliant_for_n(int(n_base), n_d=n_d, **rest)
        else:
            params = brilliant_for_n(int(n_base), **rest)
    elif n_d is not None and path is None:
        params = brilliant_for_n(params.n_base, n_d=n_d)
    return params


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_template(args) -> int:
    written = write_template(args.output)
    print(f"parameter sheet -> {written}")
    print("Edit the min/max columns, set enabled to 0 to freeze a parameter, "
          "then pass it with --space.")
    return 0


def cmd_render(args) -> int:
    config = _config(args)
    params = _params_from(args.params, getattr(args, "facets", None),
                         n_d=config.material.n_d)
    problems = params.validate()
    if problems:
        print("these parameters are not a cuttable stone:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    mesh = build_stone(params)
    result = render(
        mesh, config.material,
        camera=Camera(resolution=args.resolution, view=args.view, samples=args.samples),
        environment=config.environment, dispersion=args.dispersion_render,
    )
    save_png(result.image, args.output)
    print(f"image -> {args.output}")
    print(f"light return {result.light_return:.1%}   leakage {result.leakage:.1%}   "
          f"surface {result.surface:.1%}")
    for key, value in params.derived().items():
        print(f"  {key:26s} {value:.2f}" if isinstance(value, float) else f"  {key:26s} {value}")
    return 0


def cmd_optimise(args) -> int:
    config = _config(args)
    config.target = _resolve_target(args, config)

    space = load_space(args.space) if args.space else SearchSpace(
        base=brilliant_for_n(int(args.facets or 8), n_d=config.material.n_d))
    facet_counts = args.facet_sweep or [args.facets or 8]
    if len(facet_counts) == 1:
        space = bind_symmetry(space, int(facet_counts[0]))
        layout = layout_from_n(int(facet_counts[0]))
        print(
            f"N={facet_counts[0]}  sector={layout.sector_deg:.4f}deg  "
            f"stroke={layout.stroke_width:.5f}  "
            f"r_valley={layout.valley_radius:.4f}  "
            f"halos={layout.halo_count}"
        )

    print(f"searching {len(space.free())} parameters: "
          f"{', '.join(s.name for s in space.free())}")

    baseline = evaluate(space.to_params(space.start()), config, keep_render=True)
    print(f"starting score {baseline.score:.4f}")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    def progress(n, best):
        if n % 20 == 0 or n == args.evaluations:
            print(f"  {n:5d}/{args.evaluations}  best {best:.4f}", flush=True)

    if len(facet_counts) > 1:
        runs = optimise_over_facets(
            space, config, facet_counts=facet_counts,
            max_evaluations=args.evaluations, seed=args.seed,
            backend=args.backend, progress=progress,
        )
        print("\nlayouts, best first:")
        for run in runs:
            print(f"  {run.label:28s} {run.best.score:.4f}")
        result = runs[0]
        for run in runs:
            write_trials_csv(out / f"trials_{run.label.split('-')[0]}fold.csv", run.history)
    else:
        result = optimise(
            space, config, max_evaluations=args.evaluations, seed=args.seed,
            backend=args.backend, progress=progress,
            label=f"{facet_counts[0]}-fold",
        )

    best = result.best
    print(f"\nbest score {best.score:.4f} after {result.evaluations} stones "
          f"({result.seconds:.0f} s)")
    for key, value in best.parts.items():
        print(f"  {key:16s} {value:.4f}")
    print("winning cut:")
    for key, value in best.params.derived().items():
        print(f"  {key:26s} {value:.2f}" if isinstance(value, float) else f"  {key:26s} {value}")

    write_trials_csv(out / "trials.csv", result.history)
    for view, view_render in (best.renders or {}).items():
        name = "best.png" if len(best.renders) == 1 else f"best_{view}.png"
        save_png(view_render.image, out / name)
    for view, view_render in (baseline.renders or {}).items():
        name = "baseline.png" if len(baseline.renders) == 1 else f"baseline_{view}.png"
        save_png(view_render.image, out / name)
    for view in config.views:
        target = config.target_for(view)
        if target is not None and target.preview is not None:
            name = "target.png" if len(config.views) == 1 else f"target_{view}.png"
            save_png(target.preview, out / name)

    mesh = build_stone(best.params)
    write_gem(out / "best.gem", mesh, best.params, scale=args.scale, metrics=best.parts)
    mesh.scale(args.scale).to_obj(out / "best.obj")
    write_vertex_csv(out / "best_vertices.csv", mesh, scale=args.scale)
    write_report(out / "report.html", result, config, baseline=baseline,
                 title=f"Gem cut optimisation - {config.material.name}")

    print(f"\nwritten to {out}/")
    for name in ("report.html", "trials.csv", "best.png", "best.gem", "best.obj",
                 "best_vertices.csv", "best.json"):
        if (out / name).exists():
            print(f"  {name}")
    print(f"\nTo see it in 3ds Max, run {maxscript_dir() / 'import_stone.ms'} "
          f"and pick {out / 'best.gem'}")
    return 0


def cmd_score(args) -> int:
    """Measure a stone that came out of 3ds Max."""
    path = Path(args.stone)
    mesh = read_obj(path) if path.suffix.lower() == ".obj" else read_gem(path)
    print(f"{path.name}: {len(mesh.vertices)} vertices, {len(mesh.polygons)} facets, "
          f"{'closed' if mesh.is_watertight() else 'NOT closed -- results will be unreliable'}")

    # normalise to unit girdle radius so the optics match the built-in stones
    radius = float(np.linalg.norm(mesh.vertices[:, :2], axis=1).max())
    if radius <= 0:
        print("degenerate mesh", file=sys.stderr)
        return 1
    mesh = mesh.scale(1.0 / radius)

    config = _config(args)
    result = render(
        mesh, config.material,
        camera=Camera(resolution=args.resolution, view=args.view, samples=args.samples),
        environment=config.environment,
    )
    save_png(result.image, args.output)
    print(f"image -> {args.output}")
    print(f"light return {result.light_return:.1%}   leakage {result.leakage:.1%}   "
          f"surface {result.surface:.1%}")
    return 0


def cmd_export(args) -> int:
    params = _params_from(args.params, getattr(args, "facets", None))
    mesh = build_stone(params)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    write_gem(out / "stone.gem", mesh, params, scale=args.scale)
    mesh.scale(args.scale).to_obj(out / "stone.obj")
    write_vertex_csv(out / "stone_vertices.csv", mesh, scale=args.scale)
    print(f"written to {out}/  (stone.gem, stone.obj, stone_vertices.csv, stone.json)")
    return 0


def cmd_diagram(args) -> int:
    """PNG + MaxScript from the same PatternLayout, then the rotation test."""
    n = int(args.facets)
    layout = layout_from_n(n)
    out = Path(args.output)
    save_pattern_png(out, layout, size=args.size)
    ms = write_pattern_ms(out.with_suffix(".ms"), layout)
    image = draw_pattern(layout, size=args.size)
    err = assert_n_fold_symmetric(image, n, limit=args.limit)
    print(f"N={n}")
    print(f"  sector_deg     {layout.sector_deg:.8f}")
    print(f"  stroke_width   {layout.stroke_width:.8f}")
    print(f"  valley_radius  {layout.valley_radius:.8f}")
    print(f"  halo_count     {layout.halo_count}")
    print(f"  rotation_off   {layout.rotation_offset_deg:.8f}")
    print(f"  png            {out}")
    print(f"  maxscript      {ms}")
    print(f"  symmetry_err   {err:.6f}  (pass, limit {args.limit})")
    return 0


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--material", default="glass",
                   help="glass, lead_crystal, diamond, cubic_zirconia, or a refractive index")
    p.add_argument("--ior", type=float, default=None, help="override the refractive index")
    p.add_argument("--dispersion", type=float, default=None, help="override the dispersion")
    p.add_argument("--view", default="crown", choices=("crown", "pavilion"),
                   help="crown shows the arrows, pavilion shows the hearts")
    p.add_argument("--scope", default="hearts", choices=tuple(SCOPES),
                   help="which virtual instrument to look through")
    p.add_argument("--ies", default=None, help="light with this photometric file instead")
    p.add_argument("--resolution", type=int, default=192)
    p.add_argument("--samples", type=int, default=1, help="samples per pixel")


def _cmd_tomax(args) -> None:
    """Only 3ds Max can write its own format, so this drives it head-lessly."""
    from .tomax import gem_to_max

    result = gem_to_max(args.gem, args.out, material=args.material,
                        ies=args.ies, batch_exe=args.batch_exe)
    print(result.message)
    if result.script is not None:
        print(f"اسکریپت مکس: {result.script}")
    if result.log.strip():
        print(result.log.strip()[-1500:])
    if not result.ok:
        raise SystemExit(1)


def _cmd_maxcheck(args) -> None:
    """Say whether 3ds Max can be driven from here, and what was searched.

    "The .max was not created" has exactly one common cause, and it is this.
    Answering it needs to be one command, not a debugging session over the
    phone.
    """
    from .maxbridge import (MAX_EXE_ENV, _install_dirs, find_3dsmax_exe,
                            find_3dsmaxbatch)

    batch = find_3dsmaxbatch(args.batch_exe)
    app = find_3dsmax_exe(args.batch_exe)

    if batch is not None:
        print(f"✓ 3dsmaxbatch پیدا شد: {batch}")
        print("  فایل .max خودکار ساخته می‌شود.")
        return
    if app is not None:
        print(f"✓ 3dsmax.exe پیدا شد: {app}")
        print("  3dsmaxbatch نیست، پس مکس کامل اجرا می‌شود — کندتر ولی کار می‌کند.")
        return

    print("✗ 3ds Max پیدا نشد.")
    folders = _install_dirs()
    print(f"  {len(folders)} پوشه‌ی نصب بررسی شد"
          + (":" if folders else " — هیچ نصبی از مکس روی این سیستم نیست."))
    for folder in folders:
        print(f"    {folder}")
    print(f"\n  اگر مکس نصب است ولی اینجا نیست، مسیرش را دستی بدهید:")
    print(f"    set {MAX_EXE_ENV}=D:\\...\\3dsmaxbatch.exe")
    print("  یا فایل stone_build.ms را داخل خود مکس اجرا کنید "
          "(Scripting → Run Script).")
    raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemopt",
        description="Parametric gem cut generation, optical scoring and optimisation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("template", help="write the parameter spreadsheet")
    p.add_argument("-o", "--output", default="parameters.xlsx")
    p.set_defaults(func=cmd_template)

    p = sub.add_parser("render", help="render one set of parameters")
    _add_common(p)
    p.add_argument("--params", default=None, help="a stone .json; defaults to the standard brilliant")
    p.add_argument("--facets", type=int, default=None, help="override n_base")
    p.add_argument("--dispersion-render", action="store_true", help="trace R/G/B separately")
    p.add_argument("-o", "--output", default="stone.png")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("optimise", aliases=["optimize"], help="search for the best cut")
    _add_common(p)
    p.add_argument("--both-views", dest="view", action="store_const", const="both",
                   help="score every candidate through the crown AND the pavilion "
                        "(twice the cost, but neither pattern gets sacrificed)")
    p.add_argument("--space", default=None, help="parameter sheet from `template`")
    p.add_argument("--target", default=None, help="reference photograph to match")
    p.add_argument("--evaluations", type=int, default=400)
    p.add_argument("--facets", type=int, default=8, help="base symmetry N (57 facets = 8)")
    p.add_argument("--facet-sweep", type=int, nargs="+", default=None,
                   help="run a separate search per layout, e.g. --facet-sweep 6 8 10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="auto", choices=("auto", "de", "optuna"))
    p.add_argument("--scale", type=float, default=6.5,
                   help="girdle diameter in scene units for the exports")
    p.add_argument("--weight-pattern", type=float, default=None)
    p.add_argument("--weight-return", type=float, default=None)
    p.add_argument("--weight-leakage", type=float, default=None)
    p.add_argument("-o", "--outdir", default="results")
    p.set_defaults(func=cmd_optimise)

    p = sub.add_parser("tomax", help="build a .max file from a .gem (needs 3ds Max)")
    p.add_argument("gem", help="the .gem file to convert")
    p.add_argument("-o", "--out", help="defaults to the same name with .max")
    p.add_argument("--material", default="vray", choices=("vray", "physical", "none"))
    p.add_argument("--ies", help="photometric file for a light in the scene")
    p.add_argument("--batch-exe", help="path to 3dsmaxbatch.exe if it is not found")
    p.set_defaults(func=_cmd_tomax)

    p = sub.add_parser("maxcheck", help="report whether 3ds Max can be driven from here")
    p.add_argument("--batch-exe", default=None,
                   help="path to 3dsmaxbatch.exe to test instead of searching")
    p.set_defaults(func=_cmd_maxcheck)

    p = sub.add_parser("score", help="measure a .gem or .obj exported from 3ds Max")
    _add_common(p)
    p.add_argument("stone")
    p.add_argument("-o", "--output", default="scored.png")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("export", help="write .gem / .obj / vertex CSV for a stone")
    p.add_argument("--params", default=None)
    p.add_argument("--facets", type=int, default=None, help="override n_base")
    p.add_argument("--scale", type=float, default=6.5,
                   help="multiplier on the unit stone (girdle radius 1)")
    p.add_argument("--diameter", type=float, default=None,
                   help="girdle diameter in millimetres; overrides --scale")
    p.add_argument("-o", "--outdir", default="export")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("diagram", help="N-fold kite diagram PNG + MaxScript")
    p.add_argument("--facets", type=int, default=8)
    p.add_argument("-o", "--output", default="gemstone_facets.png")
    p.add_argument("--size", type=int, default=900)
    p.add_argument("--limit", type=float, default=0.04,
                   help="max mean-abs error for the 360/N rotation test")
    p.set_defaults(func=cmd_diagram)

    return parser


def main(argv: list[str] | None = None) -> int:
    # The messages are Persian and Windows consoles still default to a code
    # page that cannot encode them, which turns a finished run into a traceback
    # about charmap.  The .bat files set the code page; this covers everyone
    # who types the command themselves.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):       # not a real console
            pass

    args = build_parser().parse_args(argv)
    # --diameter is the number the client actually has; --scale is the internal
    # multiplier it becomes, and the unit stone has radius 1.
    if getattr(args, "diameter", None):
        args.scale = float(args.diameter) / 2.0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
