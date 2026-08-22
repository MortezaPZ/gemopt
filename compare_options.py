"""Put every option the client can choose between on one page.

The question is no longer "can it be done" but "which trade do you want".
There are three levers and they pull against each other:

* the material -- physics sets a hard ceiling and no amount of cutting beats it;
* brightness against likeness -- a stone can be made to draw the pattern more
  faithfully by letting light leak out of it, which looks better in a polar
  plot and worse in the hand;
* and the cut itself, which is the only one we actually optimise.

So each row is one fully optimised stone, rendered through the same recovered
instrument, scored the same way.  The client picks.
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

from gemopt.geometry import CutParams, build_stone
from gemopt.inverse import FixedInstrumentProblem, InverseProblem, load_views
from gemopt.maxbridge import write_gem, write_vertex_csv
from gemopt.metrics import Target, achievable_ceiling, relative_match
from gemopt.optics import MATERIALS, Geometry, Material
from gemopt.scope import Camera, render, save_png

EDGE = Path("/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")

PAGE = """<meta charset="utf-8">
<title>گزینه‌های تراش</title>
<style>
@page {{ size: A4; margin: 10mm; }}
body {{ font-family: Tahoma, "Segoe UI", sans-serif; direction: rtl;
       text-align: right; color: #1b2230; margin: 0; font-size: 9.5pt; }}
h1 {{ font-size: 15.5pt; color: #12305c; margin: 0 0 2px; }}
.sub {{ color: #55617a; font-size: 9pt; margin-bottom: 9px; }}
table {{ width: 100%; border-collapse: collapse; }}
td, th {{ padding: 4px 5px; text-align: center; vertical-align: middle; }}
th {{ font-size: 10pt; color: #12305c; border-bottom: 2px solid #12305c; }}
td.lbl {{ text-align: right; width: 25%; line-height: 1.6; }}
td.lbl b {{ color: #12305c; font-size: 10pt; }}
td.lbl span {{ color: #6b7488; font-size: 8pt; }}
img {{ width: 118px; height: 118px; border-radius: 50%;
      border: 1px solid #d8dde6; background: #000; }}
tr {{ border-bottom: 1px solid #e6e9ef; }}
.n {{ font-size: 8.5pt; color: #45506b; font-variant-numeric: tabular-nums; }}
.good {{ color: #14603a; font-weight: bold; }}
.bad {{ color: #a8324a; font-weight: bold; }}
.note {{ margin-top: 8px; border: 1px solid #d8dde6; border-right: 4px solid #12305c;
        background: #f7f9fc; padding: 8px 12px; border-radius: 5px; font-size: 9pt;
        line-height: 1.7; }}
.note b {{ color: #12305c; }}
.warn {{ border-right-color: #b8860b; background: #fdf9ef; }}
.warn b {{ color: #8a6508; }}
.pick {{ background: #f2faf6; }}
</style>

<h1>گزینه‌های تراش — کدام را می‌خواهید؟</h1>
<div class="sub">همه با یک دستگاه دید یکسان رندر و با یک معیار نمره‌گذاری شده‌اند &nbsp;•&nbsp;
درصدها نسبت به سقف قابل دستیابی ({ceil_pav} برای قلب‌ها، {ceil_crown} برای پیکان‌ها)</div>

<table>
<tr><th></th><th>قلب‌ها</th><th>پیکان‌ها</th><th style="width:15%">شباهت</th>
    <th style="width:16%">درخشش</th></tr>
{rows}
</table>

{notes}
"""

ROW = """<tr class="{cls}">
  <td class="lbl"><b>{title}</b><br><span>{subtitle}</span></td>
  <td><img src="{pav}"></td>
  <td><img src="{crown}"></td>
  <td class="n">قلب‌ها <span class="{pav_cls}">{pav_pct}</span><br>
      پیکان‌ها <span class="{crown_cls}">{crown_pct}</span></td>
  <td class="n">بازگشت نور<br><span class="{ret_cls}">{ret}</span><br>
      <span style="font-size:7.5pt">نشت {leak}</span></td>
</tr>"""


def display_blur(image: np.ndarray, resolution: int, sigma_cells: float,
                 polar_r: int = 64, extent: float = 0.94) -> np.ndarray:
    """Show the picture the model actually predicts, blur included.

    The instrument blurs -- that is why the match jumped from 79% to 90% once
    it was modelled.  Rendering pixel-sharp and then claiming a blurred-model
    score would be showing one thing and measuring another, so the same blur is
    applied here, converted from polar cells into pixels of this render.
    """
    from scipy.ndimage import gaussian_filter

    if sigma_cells <= 0:
        return image
    radii_per_cell = extent / polar_r
    pixels_per_radius = resolution / 2.0 / 1.02
    sigma_px = sigma_cells * radii_per_cell * pixels_per_radius
    return gaussian_filter(image, (sigma_px, sigma_px, 0), mode="nearest")


def match_contrast(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Put the render on the same brightness scale as the photograph.

    The score is a correlation, which is blind to overall brightness and
    contrast -- and so, deliberately, is the instrument solve: nothing in the
    fit pins down the absolute scale of the recovered profile, only its shape.
    That leaves the rendered picture sitting at an arbitrary contrast, which
    reads as "washed out" next to the photo even when the structure agrees at
    92%.  Rescaling to the reference's mean and spread removes a gauge that was
    never determined rather than flattering the result: every pixel moves by
    the same affine map, so nothing about the pattern changes.
    """
    lum = np.array([0.2126, 0.7152, 0.0722])
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    inside = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (0.92 * h / 2) ** 2

    src = image @ lum
    ref = reference @ lum
    rh, rw = reference.shape[:2]
    ry, rx = np.mgrid[0:rh, 0:rw]
    ref_inside = ((ry - rh / 2) ** 2 + (rx - rw / 2) ** 2) < (0.92 * rh / 2) ** 2

    s_std = src[inside].std()
    if s_std < 1e-6:
        return image
    gain = ref[ref_inside].std() / s_std
    offset = ref[ref_inside].mean() - gain * src[inside].mean()
    return np.clip(image * gain + offset, 0.0, 1.0)


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def material_of(spec: dict) -> Material:
    if spec.get("ior") and spec.get("material") in (None, "custom"):
        return Material("custom", n_d=spec["ior"], dispersion=0.01)
    name = spec.get("material", "glass")
    return MATERIALS.get(name, Material(name, n_d=spec.get("ior", 1.52), dispersion=0.01))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photo", default="1 (1).jpg")
    ap.add_argument("--solved", default="solved")
    ap.add_argument("-d", "--dir", default="results")
    ap.add_argument("-o", "--out", default="docs/options")
    ap.add_argument("--resolution", type=int, default=430)
    ap.add_argument("--psf-sigma", type=float, default=2.5,
                    help="camera blur, in polar cells; must match the search")
    args = ap.parse_args()

    solved, out_dir = Path(args.solved), Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    views = load_views(args.photo)
    caps = {v.view: achievable_ceiling(Target(polar=v.target, n_fold=8, name=v.view))
            for v in views}
    profile = np.load(solved / "profile.npy")
    # Rebuild the instrument from the profile actually in use, so the pictures
    # are the model's own prediction rather than a leftover from an earlier fit.
    dia_spec = json.loads((solved / "stage1.json").read_text(encoding="utf-8"))
    rebuilt = InverseProblem(views=views, psf_sigma=args.psf_sigma, max_bounces=10)
    instrument = {v.view: rebuilt.environment(profile, i)
                  for i, v in enumerate(rebuilt.views)}

    reference_rgb = {v.view: v.rgb for v in views}
    for v in views:
        save_png(v.rgb, out_dir / f"reference_{v.view}.png", gamma=1.0)

    options = [
        ("reference", "۱. عکس مرجع شما", "الماس واقعی — هدف", None, None),
        ("diamond", "۲. صحت‌سنجی", "همان سنگ، شبیه‌سازی‌شده در الماس",
         solved / "stage1.json", "diamond"),
        ("baseline", "۳. نقطه‌ی شروع", "زوایای استاندارد الماس روی شیشه",
         None, "glass"),
        ("balanced", "۴. شیشه — متعادل", "شباهت و درخشش با هم",
         solved / "psf_glass_balanced.json", None),
        ("appearance", "۵. شیشه — ظاهر در اولویت", "شباهت بیشتر، سنگ کم‌نورتر",
         solved / "psf_glass_appearance.json", None),
        ("zirconia", "۶. زیرکونیا", "ماده‌ی متفاوت، ضریب شکست ۲٫۲۰",
         solved / "psf_zirconia.json", None),
    ]

    rows, records = [], []
    for key, title, subtitle, spec_path, forced in options:
        if key == "reference":
            rows.append(ROW.format(
                cls="", title=title, subtitle=subtitle,
                pav=data_uri(out_dir / "reference_pavilion.png"),
                crown=data_uri(out_dir / "reference_crown.png"),
                pav_cls="", crown_cls="", ret_cls="",
                pav_pct="—", crown_pct="—", ret="—", leak="—"))
            continue

        if spec_path is not None and not spec_path.exists():
            print(f"skipping {key}: {spec_path} not found")
            continue

        if spec_path is None:
            params, material = CutParams(), MATERIALS["glass"]
        else:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            params = CutParams.from_dict(spec["params"])
            material = MATERIALS["diamond"] if forced == "diamond" else material_of(spec)

        scorer = FixedInstrumentProblem(views=views, profile=profile,
                                        material=material,
                                        psf_sigma=args.psf_sigma, max_bounces=10)
        _, patterns, _ = scorer.score(params)

        mesh = build_stone(params)
        geom = Geometry(mesh)
        shares, energy = {}, {}
        for view in ("crown", "pavilion"):
            cam = Camera(resolution=args.resolution, view=view, samples=3)
            result = render(mesh, material, cam, instrument[view], geometry=geom)
            shown = display_blur(result.image, args.resolution, args.psf_sigma)
            shown = match_contrast(shown, reference_rgb[view])
            save_png(shown, out_dir / f"{key}_{view}.png")
            energy[view] = {"return": result.light_return, "leak": result.leakage}
        for v, pattern in zip(scorer.views, patterns):
            shares[v.view] = relative_match(pattern, caps[v.view])

        good = lambda x: "good" if x >= 0.45 else ("bad" if x < 0.25 else "")
        rows.append(ROW.format(
            cls="pick" if key == "zirconia" else "",
            title=title, subtitle=subtitle,
            pav=data_uri(out_dir / f"{key}_pavilion.png"),
            crown=data_uri(out_dir / f"{key}_crown.png"),
            pav_cls=good(shares["pavilion"]), crown_cls=good(shares["crown"]),
            pav_pct=f"{shares['pavilion']:.0%}", crown_pct=f"{shares['crown']:.0%}",
            ret_cls="good" if energy["crown"]["return"] >= 0.8 else "bad",
            ret=f"{energy['crown']['return']:.0%}",
            leak=f"{energy['crown']['leak']:.0%}"))

        records.append({"key": key, "material": material.name, "ior": material.n_d,
                        "shares": shares, "energy": energy,
                        "params": params.to_dict(), "derived": params.derived()})
        print(f"{key:12s} {material.name:9s} hearts {shares['pavilion']:5.1%}  "
              f"arrows {shares['crown']:5.1%}  return {energy['crown']['return']:5.1%}")

        if key in ("balanced", "appearance", "zirconia"):
            write_gem(out_dir / f"{key}.gem", mesh, params=params)
            write_vertex_csv(out_dir / f"{key}_vertices.csv", mesh)
            mesh.to_obj(out_dir / f"{key}.obj")

    notes = """
<div class="note"><b>سطر ۲ صحت‌سنجی است.</b> سیستم تناسبات سنگِ داخل عکس شما را از روی
خود عکس بازسازی کرد و شبیه‌سازی‌اش به مرجع نزدیک شد. یعنی موتور درست کار می‌کند و
اختلاف سطرهای بعدی از <b>ماده</b> می‌آید، نه از سیستم.</div>

<div class="note"><b>سطر ۴ در برابر سطر ۵.</b> هر دو شیشه‌اند و هر دو کاملاً بهینه شده‌اند؛
تنها فرقشان این است که در سطر ۵ به سیستم گفتیم ظاهر مهم‌تر از درخشش است. شباهت بالاتر
می‌رود ولی نور بیشتری از سنگ بیرون می‌زند. این یک <b>انتخاب سلیقه‌ای</b> است، نه یک
مسئله‌ی فنی.</div>

<div class="note warn"><b>سطر ۶ جهش واقعی است.</b> زیرکونیا ضریب شکست ۲٫۲۰ دارد در برابر
۱٫۵۲ شیشه، پس زاویه‌ی بحرانی‌اش خیلی کوچک‌تر است و نور به‌جای نشت از کف، برمی‌گردد و
الگو می‌سازد. اگر امکان تغییر ماده هست، بیشترین سود همین‌جاست — بیشتر از هر مقدار
بهینه‌سازی روی شیشه.</div>

<div class="note"><b>درباره‌ی تصاویر.</b> رندرها با همان تاری‌ای نمایش داده شده‌اند که
در مدل هست، و روشنایی و کنتراستشان روی مقیاس عکس مرجع تنظیم شده است. معیار شباهت
همبستگی است و ذاتاً به مقیاس مطلق کاری ندارد؛ این تنظیم فقط همان مقیاسِ تعیین‌نشده را
هم‌تراز می‌کند و هیچ چیزی از الگو را تغییر نمی‌دهد.</div>

<div class="note warn"><b>چرا هیچ سطری ۱۰۰٪ نیست.</b> شبیه‌سازی ما کاملاً متقارن است و
عکس واقعی هرگز نیست؛ نور ناهمگون، دوربین کمی کج و فشرده‌سازی JPEG بخشی از تصویر را
غیرقابل بازتولید می‌کند. سقف واقعی همان عددی است که بالای صفحه نوشته شده و درصدها
نسبت به آن حساب شده‌اند.</div>
"""

    html = PAGE.format(ceil_pav=f"{caps['pavilion']:.2f}",
                       ceil_crown=f"{caps['crown']:.2f}",
                       rows="\n".join(rows), notes=notes)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".html").write_text(html, encoding="utf-8")
    (out_dir / "options.json").write_text(
        json.dumps({"ceilings": caps, "options": records}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    win = str(out.with_suffix(".html").resolve()).replace("/mnt/c/", "C:/")
    pdf = str(out.with_suffix(".pdf").resolve()).replace("/mnt/c/", "C:\\").replace("/", "\\")
    subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", f"file:///{win}"],
                   capture_output=True, timeout=180)
    print(f"\nwrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
