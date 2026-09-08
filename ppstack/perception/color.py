"""RGB to HSV, and hue-band thresholding, in plain numpy.

Thresholding in HSV rather than RGB is the entire reason this pipeline survives
a lighting gradient. In RGB, dimming a red block moves all three channels, so
any fixed box around "red" fails at the dark end of the table. In HSV, dimming
moves V and leaves H almost alone, so a band on hue plus a floor on saturation
tracks the same object across the whole frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> float HSV with H in [0, 360), S and V in [0, 1]."""
    arr = np.asarray(rgb, dtype=float) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    v = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    c = v - mn

    # Guard the divides: a grey pixel has c == 0 and an undefined hue, and a
    # black pixel has v == 0 and an undefined saturation. Both get 0.
    safe_c = np.where(c == 0, 1.0, c)
    h = np.select(
        [c == 0, v == r, v == g],
        [0.0, ((g - b) / safe_c) % 6.0, (b - r) / safe_c + 2.0],
        default=(r - g) / safe_c + 4.0,
    ) * 60.0
    s = np.where(v == 0, 0.0, c / np.where(v == 0, 1.0, v))
    return np.stack([h % 360.0, s, v], axis=-1)


@dataclass(frozen=True)
class ColorBand:
    """A named acceptance region in HSV.

    `hue_lo > hue_hi` means the band wraps through 0, which is the case that
    catches people out: red straddles the seam, so a plain `lo <= h <= hi` test
    silently matches nothing.
    """

    name: str
    hue_lo: float
    hue_hi: float
    sat_min: float = 0.35
    val_min: float = 0.15
    val_max: float = 1.01

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        if self.hue_lo <= self.hue_hi:
            in_hue = (h >= self.hue_lo) & (h <= self.hue_hi)
        else:
            in_hue = (h >= self.hue_lo) | (h <= self.hue_hi)
        return in_hue & (s >= self.sat_min) & (v >= self.val_min) & (v <= self.val_max)


# Bands tuned against the synthetic scene's palette under its lighting gradient.
DEFAULT_BANDS: tuple[ColorBand, ...] = (
    ColorBand("red", 345.0, 15.0, sat_min=0.45, val_min=0.15),
    ColorBand("yellow", 40.0, 70.0, sat_min=0.40, val_min=0.25),
    ColorBand("green", 90.0, 160.0, sat_min=0.40, val_min=0.15),
    ColorBand("blue", 200.0, 260.0, sat_min=0.45, val_min=0.15),
)

# Obstacles are dark and unsaturated, so they are found by absence of colour
# rather than by hue. A hue band would be meaningless on a near-grey pixel.
OBSTACLE_BAND = ColorBand("obstacle", 0.0, 360.0, sat_min=0.0, val_min=0.0, val_max=0.30)
