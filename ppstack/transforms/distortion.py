"""Radial lens distortion, and estimating it from the same fiducials.

A homography is exact for a pinhole camera looking at a plane. Real lenses are
not pinholes: a wide-angle or cheap lens bows straight lines, and the further a
point sits from the optical centre the worse it gets. Fitting a homography to
distorted pixels does not fail loudly. It produces a fit that is good in the
middle of the image and steadily worse towards the edges, which is exactly
where a table's corners are.

The model here is the standard one-parameter radial term, written in the
direction the image warp actually needs:

    p_undistorted = c + (p - c) * (1 + k1 * |p - c|^2)

with the radius normalised by half the image diagonal so that k1 is
dimensionless and comparable across resolutions. Writing it this way, rather
than the more common undistorted-to-distorted direction, means undistorting a
pixel is a direct evaluation instead of a Newton solve on every access.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .homography import DegenerateCorrespondences, fit_homography, reprojection_error


class DistortionDiverged(ValueError):
    """Raised when the distortion model is outside the radius it is valid on."""


@dataclass(frozen=True)
class RadialDistortion:
    """One-parameter radial model about an optical centre."""

    k1: float
    center: tuple[float, float]
    norm: float          # radius normaliser, half the image diagonal

    @classmethod
    def for_image(cls, image_size: tuple[int, int], k1: float = 0.0) -> "RadialDistortion":
        w, h = image_size
        return cls(k1, (w / 2.0, h / 2.0), 0.5 * float(np.hypot(w, h)))

    def undistort(self, uv: np.ndarray) -> np.ndarray:
        """Map observed pixels to the ideal pinhole pixels."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        if self.k1 == 0.0:
            return uv.copy()
        d = uv - np.asarray(self.center)
        r2 = (d**2).sum(axis=1, keepdims=True) / (self.norm**2)
        return np.asarray(self.center) + d * (1.0 + self.k1 * r2)

    def distort(self, uv: np.ndarray, *, iterations: int = 20) -> np.ndarray:
        """The inverse map, by fixed-point iteration.

        Only the synthetic camera needs this direction, and only to place a
        known world point in the rendered image, so a short iteration is
        cheaper and clearer than inverting the cubic in closed form.

        The iteration only contracts while 1 + k1*r^2 stays comfortably
        positive. Past that the model has folded the image over on itself and
        the fixed point does not exist, so this raises rather than returning the
        plausible-looking garbage a silent divergence produces.
        """
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        if self.k1 == 0.0:
            return uv.copy()
        c = np.asarray(self.center)
        guess = uv.copy()
        for _ in range(iterations):
            d = guess - c
            r2 = (d**2).sum(axis=1, keepdims=True) / (self.norm**2)
            denom = 1.0 + self.k1 * r2
            if np.any(denom <= 0.05):
                raise DistortionDiverged(
                    f"k1={self.k1:+.3f} folds the image at radius "
                    f"{float(np.sqrt(r2.max())):.2f} (normalised); the model is only "
                    "valid while 1 + k1*r^2 stays positive"
                )
            guess = c + (uv - c) / denom
        residual = float(np.abs(self.undistort(guess) - uv).max())
        if not np.isfinite(residual) or residual > 1e-3:
            raise DistortionDiverged(
                f"the inverse distortion did not converge (residual {residual:.3g} px)"
            )
        return guess


def estimate_k1(
    pixel_pts: np.ndarray,
    table_pts: np.ndarray,
    image_size: tuple[int, int],
    *,
    bracket: tuple[float, float] = (-0.6, 0.6),
    tolerance: float = 1e-5,
) -> tuple[RadialDistortion, float]:
    """Recover k1 from the calibration fiducials alone. Returns (model, rms).

    The trick is that k1 does not need its own measurements. For any candidate
    k1 the fiducials can be undistorted, a homography fitted to them, and the
    reprojection residual computed; the true k1 is the one that makes the
    plane-to-plane map actually be a homography, so it minimises that residual.
    The whole thing is one scalar to search over.

    The objective is smooth and unimodal near the optimum, so golden-section
    search is used: it needs no derivative, evaluates once per iteration, and
    cannot overshoot the way a Newton step on a noisy residual can.

    This only works with more than four fiducials. With exactly four the
    homography fits any k1 perfectly and the objective is flat.
    """
    pixel_pts = np.asarray(pixel_pts, dtype=float)
    table_pts = np.asarray(table_pts, dtype=float)
    if len(pixel_pts) <= 4:
        raise ValueError(
            "estimating distortion needs more than 4 fiducials; with exactly 4 the "
            "homography absorbs any k1 and the residual is flat"
        )

    def rms(k1: float) -> float:
        model = RadialDistortion.for_image(image_size, k1)
        undistorted = model.undistort(pixel_pts)
        try:
            H = fit_homography(undistorted, table_pts)
        except DegenerateCorrespondences:
            return float("inf")
        return float(np.sqrt((reprojection_error(H, undistorted, table_pts) ** 2).mean()))

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = bracket
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = rms(c), rms(d)
    while b - a > tolerance:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = rms(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = rms(d)
    best = 0.5 * (a + b)
    return RadialDistortion.for_image(image_size, best), rms(best)
