"""gemopt -- parametric gem cut generation, optical scoring and optimisation.

Built for the workflow the client already has: 3ds Max holds the model, but
the search runs entirely in Python because tracing light through a stone
numerically is thousands of times faster than rendering it.
"""

from .geometry import CutParams, Mesh, STANDARD_BRILLIANT, build_stone, facet_count
from .optics import (
    CROWN_GLASS,
    CUBIC_ZIRCONIA,
    DIAMOND,
    LEAD_CRYSTAL,
    MATERIALS,
    Geometry,
    Material,
    trace,
)
from .ies import IESProfile, load_ies
from .scope import (
    HEARTS_ARROWS_SCOPE,
    IDEALSCOPE,
    Camera,
    IESEnvironment,
    ScopeEnvironment,
    render,
    save_png,
)
from .metrics import ScoreWeights, Target, load_target
from .evaluate import EvalConfig, Evaluation, evaluate, make_reference
from .params import SearchSpace, load_space, write_template
from .optimize import optimise, optimise_over_facets

__version__ = "0.1.0"

__all__ = [
    "CutParams", "Mesh", "STANDARD_BRILLIANT", "build_stone", "facet_count",
    "Material", "Geometry", "trace", "MATERIALS",
    "DIAMOND", "CROWN_GLASS", "LEAD_CRYSTAL", "CUBIC_ZIRCONIA",
    "IESProfile", "load_ies",
    "Camera", "render", "save_png", "ScopeEnvironment", "IESEnvironment",
    "HEARTS_ARROWS_SCOPE", "IDEALSCOPE",
    "ScoreWeights", "Target", "load_target",
    "EvalConfig", "Evaluation", "evaluate", "make_reference",
    "SearchSpace", "load_space", "write_template",
    "optimise", "optimise_over_facets",
    "__version__",
]
