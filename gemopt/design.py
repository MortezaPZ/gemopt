"""Transfer a fitted cut to a new count while retaining its proportions.

When the photograph itself repeats, the design preview repeats that motif
at the new count. A single drawing is shown unchanged. The optical image is
always a ray trace of the exported destination mesh; these are distinct
images because changing facet normals can change light paths.
"""

from dataclasses import dataclass, replace

import numpy as np

from .fold import _fold, remap_image_fold
from .geometry import CutParams, build_stone
from .inverse import InverseProblem, ViewData, _align_framed
from .metrics import _normalise, to_polar
from .optics import Geometry
from .scope import Camera, RenderResult, render


def transfer_cut(source: CutParams, dest_n: int) -> CutParams:
    """Change only the repetition count, keeping every fitted dimension."""
    return replace(source, n_base=_fold(dest_n))


def retarget_views(views: list[ViewData], dest_n: int) -> list[ViewData]:
    """Transform from native pixels every time, never from a previous edit."""
    out = []
    for v in views:
        rgb = v.rgb_native if v.rgb_native is not None else v.rgb
        if not v.repeats:
            # One motif stays where it was drawn. The column count does not tile it.
            polar = to_polar(rgb, radius=min(rgb.shape[:2]) * 0.49,
                             n_r=v.target.shape[0], n_theta=v.target.shape[1])
            data = ViewData(view=v.view, target=_normalise(polar), rgb=np.array(rgb, copy=True),
                            radius_scale=v.radius_scale, source_fold=v.source_fold,
                            repeats=False)
            data.rgb_native = np.array(rgb, copy=True)
            data.prepare(dest_n)
            out.append(data)
            continue
        polar = to_polar(rgb, radius=min(rgb.shape[:2]) * 0.49,
                         n_r=v.target.shape[0], n_theta=v.target.shape[1])
        data = ViewData(view=v.view, target=_normalise(polar), rgb=rgb.copy(),
                        radius_scale=v.radius_scale)
        _align_framed(data, dest_n, source_n=v.source_fold)
        data.prepare(dest_n)
        out.append(data)
    return out


@dataclass
class DesignRender:
    source: RenderResult
    destination: RenderResult
    pattern: np.ndarray
    optics: np.ndarray
    source_display: np.ndarray


def render_design(source: CutParams, dest_n: int, problem: InverseProblem,
                  profile: np.ndarray, view_index: int = 0,
                  resolution: int = 420, samples: int = 4) -> DesignRender:
    """Produce a form-preserving preview and the true destination trace.

    Use the same fitted instrument and material for both traces. Source fit
    and display framing are independent of the requested destination count.
    """
    from .app import DISPLAY_GAMMA, _crop_to_stone, _match

    view = problem.views[view_index]
    source_mesh = build_stone(source)
    environment = problem.environment(profile, view_index)
    camera = Camera(resolution=resolution, view=view.view, samples=samples)
    shot = render(source_mesh, problem.material, camera, environment,
                  geometry=Geometry(source_mesh))
    display = np.power(np.clip(shot.image, 0.0, 1.0), 1.0 / DISPLAY_GAMMA)
    matched = _match(display, view.rgb)
    if not view.repeats:
        # One drawing stays one drawing. The column search must not tile it.
        photo = view.rgb_native if view.rgb_native is not None else view.rgb
        shown = _crop_to_stone(np.asarray(photo, dtype=np.float64))
    else:
        shown = None
    if dest_n == source.n_base:
        pattern = shown if shown is not None else matched
        return DesignRender(shot, shot, pattern, matched, matched)

    mesh = build_stone(transfer_cut(source, dest_n))
    dest = render(mesh, problem.material, camera, environment, geometry=Geometry(mesh))
    dest_display = np.power(np.clip(dest.image, 0.0, 1.0), 1.0 / DISPLAY_GAMMA)
    if shown is not None:
        reference = np.asarray(view.rgb_native if view.rgb_native is not None else view.rgb,
                               dtype=np.float64)
        return DesignRender(shot, dest, shown, _match(dest_display, reference), matched)
    reference = remap_image_fold(view.rgb, source.n_base, dest_n)
    return DesignRender(shot, dest,
                        remap_image_fold(matched, source.n_base, dest_n),
                        _match(dest_display, reference), matched)
