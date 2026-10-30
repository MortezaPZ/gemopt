"""Push the diamond cut as close to the reference photograph as it will go.

The stone in the photograph is a diamond, so this is the case where the
material is not fighting us and the only question left is the cut.  Every
candidate is scored with its own best-fitting instrument -- the viewer is
genuinely unknown, so solving for it rather than guessing removes it as a
source of error -- and with the camera blur in the model, which is what took
the match from about 79% of achievable to about 90%.
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

from gemopt.geometry import CutParams
from gemopt.inverse import InverseProblem, load_views
from gemopt.metrics import Target, achievable_ceiling, relative_match
from gemopt.optics import MATERIALS, Material
from gemopt.params import SearchSpace

_STATE: dict = {}


def _limit_threads() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def _init(photo, ior, psf_sigma, smooth, max_bounces, n_base, girdle_multiple):
    _limit_threads()
    material = MATERIALS["diamond"] if ior is None else Material(
        "custom", n_d=ior, dispersion=0.044)
    _STATE["p"] = InverseProblem(views=load_views(photo), material=material,
                                 psf_sigma=psf_sigma, smooth=smooth,
                                 max_bounces=max_bounces)
    _STATE["s"] = SearchSpace(base=CutParams(n_base=n_base,
                                             girdle_multiple=girdle_multiple))


def _objective(vector) -> float:
    return -_STATE["p"].score(_STATE["s"].to_params(np.asarray(vector)))[0]


def main() -> None:
    _limit_threads()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photo", default="assets/reference-stone.jpg")
    ap.add_argument("--ior", type=float, help="defaults to diamond")
    ap.add_argument("--facets", type=int, default=8,
                    help="base symmetry N: 8 gives 8 arrows, 16 gives 16")
    ap.add_argument("--girdle-multiple", type=int, default=2,
                    help="girdle facets per main facet; 2 is the classic 57-facet cut")
    ap.add_argument("--psf-sigma", type=float, default=2.5)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--max-bounces", type=int, default=10)
    ap.add_argument("--generations", type=int, default=26)
    ap.add_argument("--popsize", type=int, default=14)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="solved/psf_diamond.json")
    args = ap.parse_args()

    space = SearchSpace(base=CutParams(n_base=args.facets,
                                       girdle_multiple=args.girdle_multiple))
    low, high = space.bounds()
    bounds = list(zip(low, high))
    init_args = (args.photo, args.ior, args.psf_sigma, args.smooth,
                 args.max_bounces, args.facets, args.girdle_multiple)

    views = load_views(args.photo)
    caps = {v.view: achievable_ceiling(Target(polar=v.target, n_fold=8, name=v.view))
            for v in views}

    started = time.time()
    generation = [0]

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
    score, per_view, profile = _STATE["p"].score(params)

    print(f"\nbest {score:.4f}")
    for view, sim in zip(_STATE["p"].views, per_view):
        print(f"    {view.view:9s} similarity {sim:.4f}  "
              f"= {relative_match(sim, caps[view.view]):.1%} of achievable")
    for key, value in params.derived().items():
        print(f"    {key:24s} {value:.3f}")

    out = Path(args.out)
    np.save(out.with_name(out.stem + "_profile.npy"), profile)
    out.write_text(json.dumps({
        "params": params.to_dict(),
        "derived": params.derived(),
        "score": score,
        "similarity": list(per_view),
        "of_achievable": {v.view: relative_match(s, caps[v.view])
                          for v, s in zip(_STATE["p"].views, per_view)},
        "material": _STATE["p"].material.name,
        "ior": _STATE["p"].material.n_d,
        "psf_sigma": args.psf_sigma,
        "n_base": args.facets,
        "girdle_multiple": args.girdle_multiple,
        "seconds": time.time() - started,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
