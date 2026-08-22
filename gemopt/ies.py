"""IESNA LM-63 photometric file reader.

The client works with a real luminaire -- ``075 SPARK 5000 840 30D.ies``, a
54 W 4000 K spot with a 30 degree beam -- and the way that lamp concentrates
its light changes which facets light up.  Reading the actual photometry means
the simulation is lit the same way the 3ds Max scene is, instead of by a
generic point source.

Only what we need is implemented: the angular intensity distribution.  Files
with a single horizontal plane (rotationally symmetric, which is the usual
case for a spot) are handled exactly; files with several planes are averaged
over the horizontal angle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["IESProfile", "load_ies"]


@dataclass
class IESProfile:
    """Candela as a function of the angle from the beam axis."""

    vertical_angles: np.ndarray   # degrees, 0 = straight down the beam
    candela: np.ndarray           # same length, averaged over horizontal angle
    lumens: float = 0.0
    name: str = ""

    def intensity(self, angle_deg: np.ndarray) -> np.ndarray:
        """Interpolated candela at arbitrary angles, 0 outside the table."""
        return np.interp(
            np.abs(np.asarray(angle_deg, dtype=np.float64)),
            self.vertical_angles,
            self.candela,
            left=self.candela[0],
            right=0.0,
        )

    def normalised(self, angle_deg: np.ndarray) -> np.ndarray:
        """Intensity scaled so the peak is 1.0 -- what the sampler wants."""
        peak = float(self.candela.max())
        if peak <= 0.0:
            return np.ones_like(np.asarray(angle_deg, dtype=np.float64))
        return self.intensity(angle_deg) / peak

    @property
    def beam_angle(self) -> float:
        """Full angle containing the half-peak intensity -- the '30D' number."""
        peak = float(self.candela.max())
        if peak <= 0.0:
            return 180.0
        above = self.vertical_angles[self.candela >= 0.5 * peak]
        return float(2.0 * above.max()) if len(above) else 0.0

    @property
    def cutoff_angle(self) -> float:
        """Beyond this angle the luminaire emits nothing measurable."""
        nz = self.vertical_angles[self.candela > 0.0]
        return float(nz.max()) if len(nz) else 90.0


def load_ies(path) -> IESProfile:
    """Parse an LM-63 file (1986/1991/1995/2002 layouts all work).

    The format puts a free-form keyword block first, then ``TILT=``, then a
    stream of whitespace-separated numbers whose meaning is positional.  We
    key off ``TILT=`` rather than line numbers because the keyword block
    length varies between vendors.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    name = ""
    for line in raw.splitlines():
        upper = line.upper()
        if upper.startswith("[LUMINAIRE]"):
            name = line.split("]", 1)[1].strip()
        if upper.startswith("TILT="):
            if not upper.startswith("TILT=NONE"):
                raise NotImplementedError(
                    "IES files with a TILT table are not supported; "
                    "re-export with TILT=NONE"
                )
            body = raw.split(line, 1)[1]
            break
    else:
        raise ValueError("not an IES file: no TILT= line found")

    nums: list[float] = []
    for token in body.split():
        try:
            nums.append(float(token))
        except ValueError:
            continue

    if len(nums) < 13:
        raise ValueError("IES file truncated")

    n_lamps = int(nums[0])
    lumens_per_lamp = nums[1]
    multiplier = nums[2]
    n_vert = int(nums[3])
    n_horiz = int(nums[4])
    # nums[5] = photometric type, 6 = units, 7..9 = luminous dimensions
    ballast = nums[10]
    # nums[11] = future use, 12 = input watts
    cursor = 13

    vert = np.asarray(nums[cursor:cursor + n_vert], dtype=np.float64)
    cursor += n_vert
    horiz = np.asarray(nums[cursor:cursor + n_horiz], dtype=np.float64)
    cursor += n_horiz

    need = n_vert * n_horiz
    values = np.asarray(nums[cursor:cursor + need], dtype=np.float64)
    if len(values) < need:
        raise ValueError("IES candela table is shorter than its header claims")
    grid = values.reshape(n_horiz, n_vert) * multiplier * ballast

    # Average over horizontal planes; for a rotationally symmetric spot there
    # is only one plane and this is a no-op.
    candela = grid.mean(axis=0)

    order = np.argsort(vert)
    return IESProfile(
        vertical_angles=vert[order],
        candela=candela[order],
        lumens=float(n_lamps * lumens_per_lamp),
        name=name,
    )
