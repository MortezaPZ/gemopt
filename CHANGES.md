# Cut and export changes

## Extra facets and a triangulated pavilion

Extra crown rows sit between the bezel and the girdle. Extra pavilion rows sit between the main pavilion facets and the girdle. The pavilion is triangulated so those rows meet the girdle without a gap.

## Culet

The culet stays flat. It is not pinched to a single point.

## Flat girdle

A flat girdle is a separate control from the facet rows.

## Table to girdle diameter

The table size is a ratio of the girdle diameter, and that ratio is one of the search parameters.

## Real stone size

Export can set a physical diameter. `python -m gemopt export --diameter 6.5` is the command form. The same size is written into `result.json`, `stone.gem`, and `stone.obj`.

## Five decimal places

Coordinates are written to five decimal places in every export, so the JSON, the gem file, and the OBJ do not disagree by rounding.

## Search precision

The graphical presets are fast, balanced, and precise. Recorded times on the notes that shipped with this file: glass 7 / 13 / 22 minutes, diamond 10 / 20 / 32 minutes.

## Automatic Max file

`check-max.bat` reports whether 3ds Max can be launched. `build-max-file.bat` runs `python -m gemopt tomax` on `results\app\stone.gem` unless another `.gem` path is passed. `stone_build.ms` is always written. `stone.max` is written only when Max is installed here.

The `.max` build path is tested with a stand-in Max executable on disk, so argument order and path escaping are checked without a licensed Max install. Running `import_stone.ms` inside 3ds Max itself was not done on the development machine.

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers geometry, optics against known analytical answers, the Max exchange format, the stand-in Max build, and the UI.
