"""One number for one stone.

This is the objective function.  Everything else in the package feeds into
it: parameters become a mesh, the mesh is viewed through a simulated scope,
the resulting picture is compared with the reference, and the physical
numbers from the tracer are folded in.

Keeping it in one place matters, because the optimiser, the command line and
the reports must all agree on what "better" means.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import CutParams, Mesh, build_stone
from .metrics import (
    POLAR_R,
    POLAR_THETA,
    ScoreWeights,
    Target,
    is_uniform,
    contrast,
    pattern_similarity,
    symmetry_error,
    to_polar,
)
from .optics import CROWN_GLASS, Geometry, Material
from .scope import (
    HEARTS_ARROWS_SCOPE,
    Camera,
    Environment,
    RenderResult,
    render,
)

__all__ = ["EvalConfig", "Evaluation", "evaluate", "make_reference"]


@dataclass
class EvalConfig:
    """Everything that stays fixed while the parameters are searched.

    Holding the viewing setup constant is not a detail -- if the lighting or
    the camera moved between candidates their scores would not be comparable
    and the whole search would be meaningless.
    """

    material: Material = CROWN_GLASS
    environment: Environment | dict[str, Environment] = field(
        default_factory=lambda: HEARTS_ARROWS_SCOPE
    )
    """The instrument, or one per view.

    The hearts and arrows views are taken through the viewer from opposite
    ends, and a calibration against real photographs comes back with
    different band angles for each, so they have to be allowed to differ."""
    view: str = "pavilion"
    """``"pavilion"`` shows the hearts, ``"crown"`` shows the arrows, and
    ``"both"`` scores the stone through both and averages.

    A cut that draws clean arrows can still leak badly when turned over, so
    optimising one view alone quietly sacrifices the other.  ``"both"`` costs
    twice as much per candidate and is the honest choice when the client
    cares about both patterns."""

    resolution: int = 160
    """Search at this size; render the winners larger afterwards."""

    samples: int = 1
    max_bounces: int = 14
    dispersion: bool = False
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    target: Target | dict[str, Target] | None = None
    """One reference, or one per view when ``view`` is ``"both"``."""

    n_fold: int = 8

    @property
    def views(self) -> tuple[str, ...]:
        return ("crown", "pavilion") if self.view == "both" else (self.view,)

    def target_for(self, view: str) -> Target | None:
        if isinstance(self.target, dict):
            return self.target.get(view)
        return self.target

    def environment_for(self, view: str) -> Environment:
        if isinstance(self.environment, dict):
            return self.environment[view]
        return self.environment

    def camera(
        self,
        resolution: int | None = None,
        samples: int | None = None,
        view: str | None = None,
    ) -> Camera:
        return Camera(
            resolution=resolution or self.resolution,
            view=view or (self.view if self.view != "both" else "crown"),
            samples=samples or self.samples,
        )


@dataclass
class Evaluation:
    params: CutParams
    score: float
    parts: dict[str, float]
    render: RenderResult | None = None
    renders: dict[str, RenderResult] = field(default_factory=dict)
    """Every view that was scored, when ``keep_render`` was asked for."""

    mesh: Mesh | None = None
    rejected: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected

    def row(self) -> dict:
        """Flat dictionary for the trial log / Excel export."""
        out = {"score": self.score}
        out.update({f"m_{k}": v for k, v in self.parts.items()})
        out.update(self.params.to_dict())
        out.update({f"d_{k}": v for k, v in self.params.derived().items()})
        out["rejected"] = "; ".join(self.rejected)
        return out


def evaluate(
    params: CutParams,
    config: EvalConfig,
    keep_render: bool = False,
    resolution: int | None = None,
    samples: int | None = None,
) -> Evaluation:
    """Score one candidate stone.

    Invalid geometry is rejected *before* any ray is traced -- a stone that
    could not be cut is worth nothing, and checking costs microseconds where
    tracing costs a second.
    """
    problems = params.validate()
    if problems:
        return Evaluation(params=params, score=0.0, parts={}, rejected=problems)

    mesh = build_stone(params)
    if not mesh.is_watertight():
        return Evaluation(
            params=params, score=0.0, parts={},
            rejected=["mesh is not closed -- parameters produced a degenerate shape"],
        )

    # Build the ray-tracing structure once and reuse it for every view; it is
    # the same stone either way and rebuilding it is pure waste.
    geom = Geometry(mesh)
    per_view: dict[str, dict[str, float]] = {}
    renders: dict[str, RenderResult] = {}

    for view in config.views:
        cam = config.camera(resolution, samples, view=view)
        result = render(
            mesh,
            config.material,
            camera=cam,
            environment=config.environment_for(view),
            max_bounces=config.max_bounces,
            dispersion=config.dispersion,
            geometry=geom,
        )
        renders[view] = result

        pol = to_polar(
            result.image,
            centre=(cam.resolution / 2.0 - 0.5, cam.resolution / 2.0 - 0.5),
            radius=cam.resolution / 2.0 / cam.extent * 0.99,
            n_r=POLAR_R,
            n_theta=POLAR_THETA,
        )
        target = config.target_for(view)
        per_view[view] = {
            "light_return": result.light_return,
            "leakage": result.leakage,
            "surface": result.surface,
            "symmetry": symmetry_error(pol, config.n_fold),
            "contrast": contrast(pol),
            # A uniform target ("as white as possible") has no structure to
            # correlate against, so the pattern term becomes the light the
            # stone actually sends back -- which is what that brief means.
            "pattern": (result.light_return if is_uniform(target)
                        else pattern_similarity(pol, target) if target else 0.0),
        }

    keys = next(iter(per_view.values())).keys()
    parts = {k: float(np.mean([v[k] for v in per_view.values()])) for k in keys}
    if len(per_view) > 1:
        for view, values in per_view.items():
            parts.update({f"{view}_{k}": v for k, v in values.items()})

    # Average the *scores*, not just the ingredients: a stone that is superb
    # from one side and hopeless from the other should not be able to hide
    # behind a flattering mean.
    score = float(np.mean([config.weights.combine(v) for v in per_view.values()]))

    primary = renders[config.views[0]]
    return Evaluation(
        params=params,
        score=score,
        parts=parts,
        render=primary if keep_render else None,
        renders=renders if keep_render else {},
        mesh=mesh if keep_render else None,
    )


def make_reference(
    config: EvalConfig,
    params: CutParams | None = None,
    material: Material | None = None,
    resolution: int = 320,
) -> Target:
    """Build the target pattern by simulating an ideal diamond.

    The client's goal is "reproduce this picture, but in glass".  Rendering
    the ideal stone through the *same* virtual instrument gives a target that
    is framed, lit and coloured identically to the candidates -- so the
    comparison measures the cut and nothing else.  A photograph can be used
    instead via :func:`gemopt.metrics.load_target`, but it carries the
    camera's own lighting and white balance with it.
    """
    from .geometry import STANDARD_BRILLIANT
    from .optics import DIAMOND

    from .metrics import _normalise

    params = params or STANDARD_BRILLIANT
    mesh = build_stone(params)
    geom = Geometry(mesh)

    targets: dict[str, Target] = {}
    for view in config.views:
        cam = Camera(resolution=resolution, view=view, samples=2)
        result = render(
            mesh,
            material or DIAMOND,
            camera=cam,
            environment=config.environment_for(view),
            max_bounces=config.max_bounces,
            geometry=geom,
        )
        pol = to_polar(
            result.image,
            centre=(resolution / 2.0 - 0.5, resolution / 2.0 - 0.5),
            radius=resolution / 2.0 / cam.extent * 0.99,
            n_r=POLAR_R,
            n_theta=POLAR_THETA,
        )
        targets[view] = Target(
            polar=_normalise(pol),
            name=f"ideal diamond reference ({view})",
            n_fold=params.n_base,
            preview=result.image,
        )

    return targets if len(targets) > 1 else targets[config.views[0]]
