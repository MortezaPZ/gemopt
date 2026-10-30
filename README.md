# gemopt

Search stone-cut parameters, trace light through each candidate, score it against a reference photo, and export the winner to 3ds Max.

## Overview

gemopt varies the cut, renders the light return, and compares that image with a photo you supply. The same numbers are written to `result.json`, `stone.gem`, and `stone.obj`.

## Features

- Reference photo, maximum whiteness, or a mirror target
- Refractive index for glass (1.52), lead crystal (1.54), zirconia (2.20), diamond (2.42), or a custom value
- Facet count, girdle facets, extra crown rows, and extra pavilion rows
- Flat girdle and a fixed table-to-diameter ratio
- Five decimal places on exported coordinates
- Fast, balanced, and precise search presets
- `stone.gem` for 3ds Max, plus `stone.obj`, a vertex CSV, and a MaxScript builder
- A hollow faceted vessel model (`gemopt/vessel.py`) from the command line. It is not in the graphical UI

## Technology Stack

- Python 3.10+
- NumPy and SciPy
- Pillow
- 3ds Max, when it is installed on the same machine

## Architecture

The optimizer proposes a cut. The light engine traces rays through that solid and writes an image. A score compares the image with the reference, or rewards returned light when the target is whiteness. The best parameter set is exported.

`stone.gem` is the Max exchange file. `stone_build.ms` is always written and can build a `.max` file. If 3ds Max is on this machine, `stone.max` is written too. `python -m gemopt maxcheck` reports whether Max was found.

Facet rows are selected in Max by polygon material id (`row_pavilion_main` and the other row names).

## Installation

Python 3.10 or newer, on `PATH`.

```bash
pip install -r requirements.txt
```

On Windows, start the graphical app with `run-app.bat`, or run `python -m gemopt`. `check-max.bat` reports whether 3ds Max can be launched. `build-max-file.bat` sends `results\app\stone.gem` through `python -m gemopt tomax`.

## Usage

```bash
python -m gemopt template -o parameters.xlsx
python -m gemopt optimise --material glass
python -m gemopt render --params best.json
python -m gemopt score mystone.gem
python -m gemopt export --diameter 6.5
python -m gemopt maxcheck
python -m gemopt --help
```

`optimise_diamond.py` and `optimise_target.py` are the long parallel searches.

Recorded search times on the development notes:

| Preset | Glass | Diamond |
| --- | --- | --- |
| Fast | 7 min | 10 min |
| Balanced | 13 min | 20 min |
| Precise | 22 min | 32 min |

| File | Role |
| --- | --- |
| `stone.gem` | Model for 3ds Max |
| `stone.max` | Max file, only if Max is installed here |
| `stone_build.ms` | Script that builds the `.max` file |
| `stone.obj` | The same solid in a standard mesh format |
| `stone_vertices.csv` | Coordinates for a spreadsheet |
| `result.json` | Parameters and scores |
| `out_*.png` / `ref_*.png` | Render and reference |

## Testing

```bash
python -m pytest tests/ -q
```

The suite covers geometry, the light engine against Snell's law, the critical angle, Fresnel, and energy conservation, the Max exchange format, `.max` construction through a stand-in Max executable, and the UI. The stand-in checks argument order, path escaping, and whether the output file appeared.

## Limitations

`import_stone.ms` was checked against the file format the script parses. It was not executed inside 3ds Max on the development machine, because Max was not installed there. Existing shop Max scripts were not in the repository when the change notes were written, so row names may still need to be aligned with those scripts.

The vessel generator is command-line only.
