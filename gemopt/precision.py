"""How many decimals every number leaves this program with.

The client reads the numbers, not the code: the angle on screen, the column in
the spreadsheet and the coordinate in the ``.gem`` all have to agree, and they
have to be fine enough that two stones which really are different never look
like the same stone.  Two decimals on an angle is not fine enough -- 34.50 and
34.5049 are a visibly different cut and printed identically.

One constant, used everywhere, so the answer to "how precise is this" is a
single number rather than a survey of format strings.
"""

from __future__ import annotations

__all__ = ["DECIMALS", "fmt", "fmt_deg", "round_floats"]

DECIMALS = 5


def fmt(value, decimals: int = DECIMALS) -> str:
    """A number as text, at the project's precision."""
    return f"{float(value):.{decimals}f}"


def fmt_deg(value, decimals: int = DECIMALS) -> str:
    """An angle, with its degree sign."""
    return f"{float(value):.{decimals}f}°"


def round_floats(obj, decimals: int = DECIMALS):
    """Round every float inside a nest of dicts/lists, for writing JSON.

    JSON keeps full binary precision, which prints ``34.500000000000004`` and
    makes the client think the number is wrong.  Rounding on the way out keeps
    the file readable without pretending to a precision the search never had.
    """
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, decimals) for v in obj]
    return obj
