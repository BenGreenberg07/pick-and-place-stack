"""The full pixel -> table -> base chain, plus its health checks."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from ..frames import Point2D, RigidTransform2D
from .homography import apply_homography, fit_homography, reprojection_error


class CalibrationError(RuntimeError):
    """Raised when the calibration cannot be trusted for the current frame."""


@dataclass
class WorkspaceBounds:
    """Axis-aligned extent of the usable table, in table-frame millimetres."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, p: Point2D) -> bool:
        p.require("table")
        return self.x_min <= p.x <= self.x_max and self.y_min <= p.y <= self.y_max

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass
class CameraCalibration:
    """Pixels to millimetres to arm joint space, with the residual kept around.

    `max_residual_mm` is the acceptance threshold from the fit. A calibration
    that fits its own fiducials to 30 mm is not a calibration, and the pipeline
    refuses to run on one rather than quietly picking at the wrong place.

    That check only means something if the fit was overdetermined. A homography
    fitted to exactly four points reproduces those four points exactly, so its
    residual is zero however badly they were measured. `from_correspondences`
    therefore warns when it is handed the bare minimum.
    """

    H_pixel_to_table: np.ndarray
    table_to_base: RigidTransform2D
    bounds: WorkspaceBounds
    fit_residuals_mm: np.ndarray
    max_residual_mm: float = 3.0

    @property
    def rms_residual_mm(self) -> float:
        return float(np.sqrt((self.fit_residuals_mm**2).mean()))

    def validate(self) -> None:
        worst = float(self.fit_residuals_mm.max())
        if worst > self.max_residual_mm:
            raise CalibrationError(
                f"calibration residual is {worst:.2f} mm, above the "
                f"{self.max_residual_mm:.2f} mm threshold; recalibrate before picking"
            )

    # ----------------------------------------------------------- transforms

    def pixel_to_table(self, u: float, v: float) -> Point2D:
        xy = apply_homography(self.H_pixel_to_table, np.array([[u, v]]))[0]
        return Point2D(float(xy[0]), float(xy[1]), "table")

    def pixels_to_table(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised pixel -> table for whole footprints. Returns (N, 2) mm."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        if len(uv) == 0:
            return np.empty((0, 2))
        return apply_homography(self.H_pixel_to_table, uv)

    def table_to_pixel(self, p: Point2D) -> tuple[float, float]:
        p.require("table")
        H_inv = np.linalg.inv(self.H_pixel_to_table)
        uv = apply_homography(H_inv, np.array([[p.x, p.y]]))[0]
        return float(uv[0]), float(uv[1])

    def pixel_to_base(self, u: float, v: float) -> Point2D:
        """The whole chain, which is the one call the pipeline actually makes."""
        return self.table_to_base.apply(self.pixel_to_table(u, v))

    def base_to_table(self, p: Point2D) -> Point2D:
        return self.table_to_base.apply_inverse(p)

    def pixel_scale_mm(self, u: float, v: float) -> float:
        """Local mm-per-pixel at (u, v), from the Jacobian of the homography.

        Under perspective this is not constant across the image, so an area
        threshold in pixels means different things at the near and far edges of
        the table. The detector uses this to convert areas honestly.
        """
        h = 0.5
        p0 = self.pixel_to_table(u, v)
        px = self.pixel_to_table(u + h, v)
        py = self.pixel_to_table(u, v + h)
        J = np.array([[(px.x - p0.x) / h, (py.x - p0.x) / h],
                      [(px.y - p0.y) / h, (py.y - p0.y) / h]])
        return float(np.sqrt(abs(np.linalg.det(J))))

    # ---------------------------------------------------------------- build

    @classmethod
    def from_correspondences(
        cls,
        pixel_pts,
        table_pts,
        *,
        table_to_base: RigidTransform2D,
        bounds: WorkspaceBounds,
        max_residual_mm: float = 3.0,
    ) -> "CameraCalibration":
        pixel_pts = np.asarray(pixel_pts, float)
        table_pts = np.asarray(table_pts, float)
        if len(pixel_pts) == 4:
            warnings.warn(
                "calibrating from exactly 4 correspondences: the fit is exact by "
                "construction, so its residual is always ~0 and cannot detect a "
                "bad calibration. Use 6 or more fiducials.",
                stacklevel=2,
            )
        H = fit_homography(pixel_pts, table_pts)
        res = reprojection_error(H, pixel_pts, table_pts)
        return cls(H, table_to_base, bounds, res, max_residual_mm)

    # ------------------------------------------------------------------ i/o

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "H_pixel_to_table": self.H_pixel_to_table.tolist(),
                    "table_to_base": asdict(self.table_to_base),
                    "bounds": asdict(self.bounds),
                    "fit_residuals_mm": self.fit_residuals_mm.tolist(),
                    "max_residual_mm": self.max_residual_mm,
                },
                indent=2,
            )
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraCalibration":
        d = json.loads(Path(path).read_text())
        return cls(
            H_pixel_to_table=np.asarray(d["H_pixel_to_table"], float),
            table_to_base=RigidTransform2D(**d["table_to_base"]),
            bounds=WorkspaceBounds(**d["bounds"]),
            fit_residuals_mm=np.asarray(d["fit_residuals_mm"], float),
            max_residual_mm=float(d["max_residual_mm"]),
        )
