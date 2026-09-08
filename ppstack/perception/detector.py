"""Blob detection: an image in, frame-tagged detections out.

The detector deliberately does not know about the arm, the table, or the
planner. It reports what it sees in pixels plus a confidence, and the pipeline
is responsible for turning that into millimetres. Keeping the boundary here is
what makes it possible to swap the camera for the synthetic scene, or the numpy
backend for OpenCV, without touching anything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .color import DEFAULT_BANDS, OBSTACLE_BAND, ColorBand, rgb_to_hsv
from .morphology import Region, closing, connected_components, opening, region_properties


@dataclass
class Detection:
    """One object, still in pixel coordinates."""

    color: str
    centroid_px: tuple[float, float]
    angle_rad: float
    area_px: int
    eccentricity: float
    bbox: tuple[int, int, int, int]
    is_obstacle: bool
    # (N, 2) array of (u, v) pixels belonging to this object, subsampled. The
    # planner rasterises these rather than a bounding circle, which is the
    # difference between modelling a wall as a wall and as a large disc.
    footprint_px: np.ndarray = field(default=None, repr=False)

    @property
    def u(self) -> float:
        return self.centroid_px[0]

    @property
    def v(self) -> float:
        return self.centroid_px[1]


@dataclass
class DetectorConfig:
    footprint_samples: int = 2500
    min_area_px: int = 250
    max_area_px: int = 60_000
    open_radius: int = 2      # deletes specks and sensor noise
    close_radius: int = 2     # fills specular highlights inside a block
    # Rejects long thin streaks (a shadow edge, a cable) that are not blocks.
    # It applies to graspable objects only: a block has to be roughly square to
    # fit the gripper, but an obstacle is allowed to be a wall.
    max_eccentricity: float = 0.92
    detect_obstacles: bool = True


class BlockDetector:
    """HSV threshold -> morphological cleanup -> components -> moments.

    Backends differ only in who does the connected-component labelling:
        "numpy"  the two-pass union-find in this package, no dependencies
        "opencv" cv2.connectedComponents, which matters on masks that shatter
                 into thousands of noise components
        "auto"   opencv when importable, numpy otherwise

    The moment summary is shared, and a parity test pins the two backends to
    identical output, which is the only reason it is safe to have two of them.
    On a clean 640x480 frame the two are within a few milliseconds of each
    other, because the HSV conversion and the morphology dominate either way.
    """

    def __init__(
        self,
        bands: tuple[ColorBand, ...] = DEFAULT_BANDS,
        config: DetectorConfig | None = None,
        backend: str = "auto",
    ) -> None:
        self.bands = bands
        self.config = config or DetectorConfig()
        self.backend = self._resolve_backend(backend)

    @staticmethod
    def _resolve_backend(requested: str) -> str:
        if requested == "numpy":
            return "numpy"
        try:
            import cv2  # noqa: F401
        except ImportError:
            if requested == "opencv":
                raise ImportError(
                    "backend='opencv' requires opencv-python; "
                    "install it or use backend='numpy'"
                ) from None
            return "numpy"
        return "opencv"

    # ------------------------------------------------------------------ api

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) RGB image, got shape {rgb.shape}")
        hsv = rgb_to_hsv(rgb)

        out: list[Detection] = []
        bands = list(self.bands)
        if self.config.detect_obstacles:
            bands.append(OBSTACLE_BAND)

        for band in bands:
            mask = band.mask(hsv)
            if not mask.any():
                continue
            mask = opening(mask, self.config.open_radius)
            mask = closing(mask, self.config.close_radius)
            for region in self._regions(mask):
                if not (self.config.min_area_px <= region.area_px <= self.config.max_area_px):
                    continue
                is_obstacle = band.name == "obstacle"
                if not is_obstacle and region.eccentricity > self.config.max_eccentricity:
                    continue
                out.append(
                    Detection(
                        color=band.name,
                        centroid_px=region.centroid_px,
                        angle_rad=region.angle_rad,
                        area_px=region.area_px,
                        eccentricity=region.eccentricity,
                        bbox=region.bbox,
                        is_obstacle=is_obstacle,
                        footprint_px=_sample_footprint(
                            region, self.config.footprint_samples
                        ),
                    )
                )
        # Stable order so a run is reproducible and diffable.
        out.sort(key=lambda d: (d.color, -d.area_px))
        return out

    # ------------------------------------------------------------- backends

    def _regions(self, mask: np.ndarray) -> list[Region]:
        if self.backend == "opencv":
            return self._regions_opencv(mask)
        labels, count = connected_components(mask, connectivity=8)
        return region_properties(labels, count)

    @staticmethod
    def _regions_opencv(mask: np.ndarray) -> list[Region]:
        """cv2 does the labelling; the moment summary is shared with the numpy
        path so the two backends cannot drift apart in what they report."""
        import cv2

        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return region_properties(labels, count - 1)


def _sample_footprint(region: Region, limit: int) -> np.ndarray:
    """Region pixels as absolute (u, v), thinned to at most roughly `limit`.

    The thinning is a 2D stride over the region's bounding box, not a stride
    over the raster order. Striding the raster order leaves the kept points
    spread far apart along each row and tightly packed down the column, and the
    grid rasterised from them ends up full of phantom doorways. A 2D stride
    keeps a regular lattice whose spacing is known, so it can be chosen finer
    than the grid resolution and no hole ever appears.
    """
    if region.mask is None:
        return np.empty((0, 2), dtype=float)
    step = 1
    if limit and region.area_px > limit:
        step = int(np.ceil(np.sqrt(region.area_px / limit)))
    sub = region.mask[::step, ::step]
    vs, us = np.nonzero(sub)
    us = us * step + region.bbox[0]
    vs = vs * step + region.bbox[1]
    return np.stack([us, vs], axis=1).astype(float)
