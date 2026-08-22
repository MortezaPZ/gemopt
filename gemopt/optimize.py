"""Searching the parameter space.

The objective is cheap by rendering standards -- under a second -- but it is
not cheap enough to brute force: nine free parameters at ten steps each would
be a billion stones.  It is also not smooth, because facets appear and
disappear from view as angles cross the critical angle, so gradient methods
are the wrong tool.

Differential evolution fits: it only ever compares objective values, it
copes with the discontinuities, and it parallelises trivially if that is
needed later.  Optuna is used instead when it is installed and asked for,
since the client may want its dashboard.

Discrete structure -- how many facets in a row -- is handled by running the
whole search once per structure and comparing the winners, rather than
letting a continuous optimiser fumble with an integer.  That is also how a
cutter thinks about it: pick the layout, then perfect the angles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from .evaluate import EvalConfig, Evaluation, evaluate
from .geometry import CutParams
from .params import SearchSpace

__all__ = ["OptimizeResult", "optimise", "optimise_over_facets"]


@dataclass
class OptimizeResult:
    best: Evaluation
    history: list[dict] = field(default_factory=list)
    evaluations: int = 0
    seconds: float = 0.0
    label: str = ""

    def best_rows(self, n: int = 10) -> list[dict]:
        ok = [r for r in self.history if not r.get("rejected")]
        return sorted(ok, key=lambda r: -r["score"])[:n]


def optimise(
    space: SearchSpace,
    config: EvalConfig,
    max_evaluations: int = 400,
    population: int = 24,
    seed: int = 0,
    backend: str = "auto",
    progress: Callable[[int, float], None] | None = None,
    label: str = "",
) -> OptimizeResult:
    """Maximise the score over the free parameters of ``space``."""
    backend = _resolve_backend(backend)
    started = time.time()
    history: list[dict] = []
    best: Evaluation | None = None
    counter = {"n": 0}

    def objective(vector: np.ndarray) -> Evaluation:
        nonlocal best
        params = space.to_params(vector)
        ev = evaluate(params, config)
        counter["n"] += 1
        row = ev.row()
        row["trial"] = counter["n"]
        history.append(row)
        if best is None or ev.score > best.score:
            best = ev
        if progress is not None:
            progress(counter["n"], best.score if best else 0.0)
        return ev

    if backend == "optuna":
        _run_optuna(space, objective, max_evaluations, seed)
    else:
        _run_differential_evolution(space, objective, max_evaluations, population, seed)

    if best is None:                                    # pragma: no cover
        raise RuntimeError("no candidate was evaluated")

    # re-run the winner keeping its render, so callers get the picture too
    best = evaluate(best.params, config, keep_render=True)
    return OptimizeResult(
        best=best,
        history=history,
        evaluations=counter["n"],
        seconds=time.time() - started,
        label=label,
    )


def _resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    return "de"


# --------------------------------------------------------------------------
# differential evolution
# --------------------------------------------------------------------------


def _run_differential_evolution(
    space: SearchSpace,
    objective: Callable[[np.ndarray], Evaluation],
    max_evaluations: int,
    population: int,
    seed: int,
) -> None:
    """A compact rand/1/bin differential evolution.

    Written out rather than delegated to scipy so the evaluation budget is
    exact -- scipy's version counts generations, not calls, and the client
    needs "run 400 stones and stop" to mean that.  The seeded first member
    puts the known good proportions into the initial population, so the
    search starts from the best guess a human would make.
    """
    rng = np.random.default_rng(seed)
    low, high = space.bounds()
    dim = len(low)
    if dim == 0:
        objective(np.zeros(0))
        return

    pop_size = max(8, min(population, max(8, max_evaluations // 8)))
    pop = rng.uniform(low, high, size=(pop_size, dim))
    pop[0] = np.clip(space.start(), low, high)

    fitness = np.empty(pop_size)
    used = 0
    for i in range(pop_size):
        if used >= max_evaluations:
            fitness[i:] = -np.inf
            break
        fitness[i] = objective(pop[i]).score
        used += 1

    f_scale, crossover = 0.7, 0.9
    while used < max_evaluations:
        for i in range(pop_size):
            if used >= max_evaluations:
                break
            choices = [j for j in range(pop_size) if j != i]
            a, b, c = rng.choice(choices, size=3, replace=False)
            mutant = pop[a] + f_scale * (pop[b] - pop[c])
            mutant = np.clip(mutant, low, high)

            cross = rng.random(dim) < crossover
            if not cross.any():
                cross[rng.integers(dim)] = True
            trial = np.where(cross, mutant, pop[i])

            score = objective(trial).score
            used += 1
            if score >= fitness[i]:
                pop[i] = trial
                fitness[i] = score


# --------------------------------------------------------------------------
# optuna
# --------------------------------------------------------------------------


def _run_optuna(
    space: SearchSpace,
    objective: Callable[[np.ndarray], Evaluation],
    max_evaluations: int,
    seed: int,
) -> None:
    try:
        import optuna
    except ImportError as exc:                  # pragma: no cover
        raise SystemExit(
            "the optuna backend needs optuna installed:  pip install optuna\n"
            "the default backend needs nothing extra -- drop --backend to use it"
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    free = space.free()

    def trial_fn(trial: "optuna.Trial") -> float:
        vector = []
        for spec in free:
            if spec.kind == "int":
                vector.append(trial.suggest_int(spec.name, int(spec.low), int(spec.high)))
            elif spec.step > 0.0:
                vector.append(trial.suggest_float(spec.name, spec.low, spec.high, step=spec.step))
            else:
                vector.append(trial.suggest_float(spec.name, spec.low, spec.high))
        return objective(np.asarray(vector, dtype=np.float64)).score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.enqueue_trial({s.name: s.default for s in free})
    study.optimize(trial_fn, n_trials=max_evaluations)


# --------------------------------------------------------------------------
# structure sweep
# --------------------------------------------------------------------------


def optimise_over_facets(
    space: SearchSpace,
    config: EvalConfig,
    facet_counts: Iterable[int] = (8,),
    max_evaluations: int = 400,
    **kwargs,
) -> list[OptimizeResult]:
    """Run one full search per facet layout and return them all, best first.

    This is the answer to "what if the number of facets per row changes?" --
    the continuous parameters mean different things for different layouts, so
    each layout gets its own search and they are compared only at the end, on
    score.
    """
    results: list[OptimizeResult] = []
    for n in facet_counts:
        base = CutParams.from_dict({**space.base.to_dict(), "n_base": int(n)})
        sub = SearchSpace(specs=space.specs, base=base)
        cfg = EvalConfig(**{**vars(config), "n_fold": int(n)})
        results.append(
            optimise(sub, cfg, max_evaluations=max_evaluations,
                     label=f"{int(n)}-fold ({7 * int(n) + 1} facets)", **kwargs)
        )
    return sorted(results, key=lambda r: -r.best.score)
