"""Turning a run into something a client can read.

Two outputs.  A CSV of every stone tried, because that is the audit trail --
"show me what you searched" is a fair question and the answer should not be
"trust me".  And a single self-contained HTML page with the pictures side by
side, the winning numbers in gemmological terms, and how the score climbed.

The HTML embeds its images, so the whole report is one file that can be
emailed or opened from a USB stick.
"""

from __future__ import annotations

import base64
import csv
import html
import io
from datetime import datetime
from pathlib import Path

import numpy as np

from .evaluate import EvalConfig, Evaluation
from .optimize import OptimizeResult

__all__ = ["write_trials_csv", "write_report"]


def write_trials_csv(path, history: list[dict]) -> Path:
    """Every candidate, in the order it was tried."""
    path = Path(path)
    if not history:
        path.write_text("", encoding="utf-8")
        return path

    columns: list[str] = []
    for row in history:
        for key in row:
            if key not in columns:
                columns.append(key)
    front = [c for c in ("trial", "score") if c in columns]
    columns = front + [c for c in columns if c not in front]

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    return path


def _png_data_uri(image: np.ndarray, gamma: float = 2.2) -> str:
    from PIL import Image

    arr = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    arr = np.power(arr, 1.0 / gamma)
    buf = io.BytesIO()
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _convergence_uri(history: list[dict]) -> str | None:
    """Best-so-far against trial number, drawn without matplotlib.

    A line chart is a handful of pixels of work and pulling in a plotting
    library for it would add a heavy dependency to a package that otherwise
    needs only numpy.
    """
    scores = [r.get("score", 0.0) or 0.0 for r in history]
    if len(scores) < 2:
        return None

    running = np.maximum.accumulate(np.asarray(scores, dtype=np.float64))
    w, h = 720, 180
    canvas = np.ones((h, w, 3))

    lo, hi = float(running.min()), float(running.max())
    span = max(hi - lo, 1e-6)
    xs = np.linspace(0, w - 1, len(running))
    ys = (h - 1) - (running - lo) / span * (h - 12) - 6

    for band in range(1, 4):
        canvas[int(h * band / 4), :] = 0.90

    px = np.clip(np.round(xs).astype(int), 0, w - 1)
    py = np.clip(np.round(ys).astype(int), 0, h - 1)
    for i in range(len(px) - 1):
        x0, x1 = px[i], px[i + 1]
        y0, y1 = py[i], py[i + 1]
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for s in range(n + 1):
            x = int(round(x0 + (x1 - x0) * s / n))
            y = int(round(y0 + (y1 - y0) * s / n))
            canvas[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = (0.13, 0.42, 0.85)

    return _png_data_uri(canvas, gamma=1.0)


_CSS = """
body{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1c2330}
.wrap{max-width:1080px;margin:0 auto;padding:32px 24px 64px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:17px;margin:32px 0 12px;color:#334}
.sub{color:#667;font-size:14px;margin-bottom:24px}
.cards{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px}
.card{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:16px 18px;flex:1;min-width:150px}
.card .k{font-size:12px;color:#78808c;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:24px;font-weight:650;margin-top:4px}
.imgs{display:flex;gap:20px;flex-wrap:wrap}
.imgs figure{margin:0;background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:12px;text-align:center}
.imgs img{width:260px;height:260px;object-fit:contain;background:#20242b;border-radius:6px;display:block}
.imgs figcaption{font-size:13px;color:#667;margin-top:8px}
table{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e3e6ea;border-radius:10px;overflow:hidden;font-size:14px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #eef0f3}
th{background:#f0f2f5;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em;color:#5a6472}
tr:last-child td{border-bottom:none}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.note{background:#fff8e6;border:1px solid #f0dfae;border-radius:10px;padding:14px 16px;font-size:14px;margin-top:12px}
.scroll{overflow-x:auto}
"""


def _table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    out = ["<div class='scroll'><table><thead><tr>"]
    out += [f"<th>{html.escape(str(h))}</th>" for h in headers]
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for i, cell in enumerate(row):
            cls = " class='num'" if i >= numeric_from else ""
            out.append(f"<td{cls}>{html.escape(str(cell))}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


_METRIC_LABELS = {
    "pattern": "Pattern match with the target",
    "light_return": "Light returned to the viewer",
    "leakage": "Light leaking through",
    "surface": "Reflected off the surface",
    "symmetry": "Symmetry error (lower is better)",
    "contrast": "Pattern contrast",
}

_PARAM_LABELS = {
    "table_pct": "Table %",
    "crown_height_pct": "Crown height %",
    "pavilion_depth_pct": "Pavilion depth %",
    "girdle_thickness_pct": "Girdle thickness %",
    "total_depth_pct": "Total depth %",
    "crown_angle": "Crown angle (deg)",
    "pavilion_angle": "Pavilion angle (deg)",
    "facet_count": "Facets (trade count)",
    "facet_count_with_girdle": "Facets incl. girdle band",
}


def write_report(
    path,
    result: OptimizeResult,
    config: EvalConfig,
    baseline: Evaluation | None = None,
    title: str = "Gem cut optimisation",
) -> Path:
    """Write the single-file HTML summary of a run."""
    path = Path(path)
    best = result.best
    parts: list[str] = []

    parts.append(f"<h1>{html.escape(title)}</h1>")
    parts.append(
        "<div class='sub'>{material} &middot; {view} view &middot; {n} stones evaluated "
        "&middot; {secs:.0f} s &middot; {when}</div>".format(
            material=html.escape(config.material.name),
            view=html.escape(config.view),
            n=result.evaluations,
            secs=result.seconds,
            when=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
    )

    # headline numbers
    cards = [("Best score", f"{best.score:.4f}")]
    if baseline is not None and baseline.ok:
        gain = best.score - baseline.score
        cards.append(("Starting score", f"{baseline.score:.4f}"))
        cards.append(("Improvement", f"{gain:+.4f}"))
    cards.append(("Light return", f"{best.parts.get('light_return', 0):.1%}"))
    cards.append(("Leakage", f"{best.parts.get('leakage', 0):.1%}"))
    parts.append("<div class='cards'>")
    for key, value in cards:
        parts.append(f"<div class='card'><div class='k'>{key}</div><div class='v'>{value}</div></div>")
    parts.append("</div>")

    # pictures
    figures = []
    multi = len(config.views) > 1
    for view in config.views:
        suffix = f" ({view})" if multi else ""
        target = config.target_for(view)
        if target is not None and target.preview is not None:
            figures.append((target.preview, f"Target &mdash; {html.escape(target.name)}"))
        if baseline is not None and baseline.renders.get(view) is not None:
            figures.append((baseline.renders[view].image, f"Starting point{suffix}"))
        if best.renders.get(view) is not None:
            figures.append((best.renders[view].image, f"Best found{suffix}"))
    if not figures and best.render is not None:
        figures.append((best.render.image, "Best found"))
    if figures:
        parts.append("<h2>Patterns</h2><div class='imgs'>")
        for image, caption in figures:
            parts.append(
                f"<figure><img src='{_png_data_uri(image)}' alt='{caption}'>"
                f"<figcaption>{caption}</figcaption></figure>"
            )
        parts.append("</div>")

    # convergence
    chart = _convergence_uri(result.history)
    if chart:
        parts.append("<h2>How the search progressed</h2>")
        parts.append(
            f"<figure style='margin:0'><img src='{chart}' "
            "style='width:100%;border:1px solid #e3e6ea;border-radius:10px;background:#fff'>"
            "<figcaption style='font-size:13px;color:#667;margin-top:6px'>"
            "Best score so far against number of stones evaluated</figcaption></figure>"
        )

    # winning geometry
    parts.append("<h2>Winning cut</h2>")
    derived = best.params.derived()
    rows = [[_PARAM_LABELS.get(k, k), f"{v:.2f}" if isinstance(v, float) else v]
            for k, v in derived.items()]
    parts.append(_table(["Property", "Value"], rows))

    parts.append("<h2>Parameters (as fed to the model)</h2>")
    prows = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)]
             for k, v in best.params.to_dict().items()]
    parts.append(_table(["Parameter", "Value"], prows))

    # metric breakdown
    parts.append("<h2>Score breakdown</h2>")
    mrows = []
    for key, value in best.parts.items():
        base_txt = ""
        if baseline is not None and baseline.ok:
            base_txt = f"{baseline.parts.get(key, 0.0):.3f}"
        mrows.append([_METRIC_LABELS.get(key, key), f"{value:.3f}", base_txt])
    parts.append(_table(["Metric", "Best", "Starting point"], mrows))

    # leaderboard
    top = result.best_rows(10)
    if top:
        parts.append("<h2>Top 10 candidates</h2>")
        cols = ["trial", "score", "crown_angle", "pavilion_angle", "table",
                "lower_girdle_length", "m_light_return", "m_leakage", "m_pattern"]
        cols = [c for c in cols if c in top[0]]
        headers = [c.replace("m_", "").replace("_", " ") for c in cols]
        rows = [[f"{r[c]:.4f}" if isinstance(r.get(c), float) else r.get(c, "") for c in cols]
                for r in top]
        parts.append(_table(headers, rows))

    parts.append(
        "<div class='note'><b>How to read this.</b> The score is a weighted blend of how "
        "closely the pattern matches the target, how much light comes back to the viewer, "
        "how little leaks away, and how well the stone holds its symmetry. It is a "
        "comparison tool, not a grade: what matters is that the winning cut scores higher "
        "than the starting one under identical lighting and viewing conditions.</div>"
    )

    doc = (
        "<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style>"
        f"<div class='wrap'>{''.join(parts)}</div>"
    )
    path.write_text(doc, encoding="utf-8")
    return path
