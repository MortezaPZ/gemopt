"""Find the cut for the client's material, through the recovered instrument.

Stage one (``solve_reference.py``) worked out what the reference photograph
shows and, with it, what the viewer looks like.  This takes that viewer as
fixed -- it is one physical device, and re-solving it per candidate would let
a bad stone hide behind a flattering instrument -- and searches for the cut
that comes closest to the same picture in glass, or whatever the client
actually cuts.

The objective is deliberately not pattern similarity alone.  Similarity is
measured on contrast-normalised images, so a stone that leaks almost all its
light can still have the little that remains in roughly the right places and
score well; optimising on that produces a cut that reproduces the *shape* of
the reference while being, physically, a dead piece of glass.  Light return is
part of the score for that reason (see ``FixedInstrumentProblem``).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from gemopt.inverse import FixedInstrumentProblem, load_views
from gemopt.optics import MATERIALS, Material
from gemopt.params import SearchSpace

_STATE: dict = {}


def _limit_threads() -> None:
    """One BLAS thread per worker; the parallelism lives at the process level."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def _init(photo, profile_path, material_name, ior, min_return, return_weight,
          psf_sigma, max_bounces):
    _limit_threads()
    material = (Material("custom", n_d=ior, dispersion=0.01) if ior
                else MATERIALS[material_name])
    _STATE["p"] = FixedInstrumentProblem(
        views=load_views(photo),
        profile=np.load(profile_path),
        material=material,
        min_return=min_return,
        return_weight=return_weight,
        psf_sigma=psf_sigma,
        max_bounces=max_bounces,
    )
    _STATE["s"] = SearchSpace(base=CutParams(n_base=n_base,
                                             girdle_multiple=girdle_multiple))


def _objective(vector) -> float:
    return -_STATE["p"].score(_STATE["s"].to_params(np.asarray(vector)))[0]


def main() -> None:
    _limit_threads()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photo", default="assets/reference-stone.jpg")
    ap.add_argument("--solved", default="solved")
    ap.add_argument("--material", default="glass")
    ap.add_argument("--ior", type=float)
    ap.add_argument("--min-return", type=float, default=0.55)
    ap.add_argument("--return-weight", type=float, default=0.45)
    ap.add_argument("--generations", type=int, default=22)
    ap.add_argument("--popsize", type=int, default=20)
    ap.add_argument("--facets", type=int, default=8,
                    help="base symmetry N: 8 gives 8 arrows, 16 gives 16")
    ap.add_argument("--girdle-multiple", type=int, default=2,
                    help="girdle facets per main facet; 2 is the classic 57-facet cut")
    ap.add_argument("--psf-sigma", type=float, default=2.5,
                    help="camera blur in pixels; fitted against the reference")
    ap.add_argument("--max-bounces", type=int, default=10)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default=None,
                    help="defaults to <solved>/stage2.json")
    args = ap.parse_args()

    solved = Path(args.solved)
    profile_path = str(solved / "profile.npy")
    out_path = Path(args.out) if args.out else solved / "stage2.json"

    space = SearchSpace()
    low, high = space.bounds()
    bounds = list(zip(low, high))
    started = time.time()
    generation = [0]

    init_args = (args.photo, profile_path, args.material, args.ior,
                 args.min_return, args.return_weight, args.psf_sigma,
                 args.max_bounces)

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=init_args) as pool:
        def batch(population):
            values = list(pool.map(_objective, list(population.T), chunksize=1))
            generation[0] += 1
            print(f"  generation {generation[0]:3d}  best {-min(values):.4f}  "
                  f"({time.time() - started:.0f}s)", flush=True)
            return np.asarray(values)

        result = differential_evolution(
            batch, bounds, seed=args.seed, popsize=args.popsize,
            maxiter=args.generations, tol=0.0, mutation=(0.4, 1.0),
            recombination=0.85, polish=False, init="sobol", vectorized=True,
            x0=space.start(),
        )

    params = space.to_params(result.x)
    _init(*init_args)
    score, patterns, returns = _STATE["p"].score(params)

    print(f"\nbest combined {score:.4f}")
    for view, pattern, ret in zip(_STATE["p"].views, patterns, returns):
        print(f"    {view.view:9s} pattern {pattern:.4f}   light return {ret:.1%}")
    for key, value in params.derived().items():
        print(f"    {key:24s} {value:.3f}")

    material = _STATE["p"].material
    out_path.write_text(json.dumps({
        "params": params.to_dict(),
        "derived": params.derived(),
        "score": score,
        "pattern": patterns,
        "returns": returns,
        "material": material.name,
        "ior": material.n_d,
        "min_return": args.min_return,
        "return_weight": args.return_weight,
        "psf_sigma": args.psf_sigma,
        "seconds": time.time() - started,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
