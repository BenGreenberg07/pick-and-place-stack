"""A synthetic overhead camera, with ground truth.

The point of this module is measurable accuracy. A webcam demo can only be
eyeballed: the blob looks like it is on the block, so the pipeline "works".
Here the true block poses in table millimetres are known exactly, the image is
rendered from them through a real perspective warp, and the pipeline's output
can be scored in millimetres against the truth. That turns the whole repo from
a demo into an experiment, and it is what makes the end-to-end error number in
the README meaningful.

It also means every test runs without a camera, deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..transforms.homography import apply_homography

# Reference block colours as (R, G, B), 0-255, before lighting is applied.
PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (196, 42, 38),
    "green": (44, 160, 68),
    "blue": (40, 78, 190),
    "yellow": (222, 190, 44),
    "obstacle": (46, 46, 52),
}


@dataclass(frozen=True)
class Block:
    """A square block on the table, in table-frame millimetres."""

    x: float
    y: float
    theta: float
    size: float
    color: str

    @property
    def is_obstacle(self) -> bool:
        return self.color == "obstacle"

    def corners(self) -> np.ndarray:
        h = self.size / 2.0
        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])
        local = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
        return local @ R.T + (self.x, self.y)


@dataclass
class SceneSpec:
    """Everything needed to render a frame and to score the result against it."""

    blocks: list[Block]
    table_size_mm: tuple[float, float] = (400.0, 300.0)
    image_size: tuple[int, int] = (640, 480)
    table_color: tuple[int, int, int] = (208, 202, 190)
    # Perspective strength. 0 gives a pure scale-and-shift (an orthographic
    # camera); larger values tilt the camera so the far edge of the table
    # compresses, which is what a real overhead rig actually looks like.
    perspective: float = 0.28
    lighting_gradient: float = 0.35
    noise_sigma: float = 4.0
    speckle_count: int = 40
    seed: int = 0
    _H_table_to_pixel: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._H_table_to_pixel = self._build_camera()

    # ----------------------------------------------------------- the camera

    def _build_camera(self) -> np.ndarray:
        """Homography mapping table millimetres to image pixels.

        Built from the four table corners so the whole table fits the frame with
        a margin, then tilted: the far edge (large y) is pulled inward by
        `perspective`, which is exactly what a camera looking down at an angle
        does to a plane.
        """
        W, H = self.image_size
        tw, th = self.table_size_mm
        margin = 0.07
        x0, x1 = margin * W, (1 - margin) * W
        y0, y1 = (1 - margin) * H, margin * H  # table +y is image -v
        shrink = self.perspective * 0.5 * (x1 - x0)
        src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], float)
        dst = np.array(
            [[x0, y0], [x1, y0], [x1 - shrink, y1], [x0 + shrink, y1]], float
        )
        from .._dlt import fit  # local import keeps the dependency direction clean

        return fit(src, dst)

    @property
    def H_table_to_pixel(self) -> np.ndarray:
        return self._H_table_to_pixel

    @property
    def H_pixel_to_table(self) -> np.ndarray:
        return np.linalg.inv(self._H_table_to_pixel)

    def table_corners_px(self) -> np.ndarray:
        tw, th = self.table_size_mm
        return apply_homography(
            self._H_table_to_pixel,
            np.array([[0, 0], [tw, 0], [tw, th], [0, th]], float),
        )

    def calibration_correspondences(
        self, noise_px: float = 0.0, seed: int = 12, grid: int = 3
    ):
        """Fiducials for calibrating: a `grid` x `grid` lattice on the table,
        returned in both pixels and millimetres.

        Deliberately more than the four points a homography needs. A four-point
        fit passes exactly through its own four points no matter how badly they
        were measured, so its residual is identically zero and tells you
        nothing about whether the calibration is any good. Only redundant
        correspondences make the residual a real diagnostic, and the pipeline
        relies on that residual to refuse to run on a bad calibration.

        `noise_px` simulates a human clicking the fiducials imprecisely.
        """
        if grid < 2:
            raise ValueError("need at least a 2x2 fiducial grid")
        tw, th = self.table_size_mm
        xs = np.linspace(0.06 * tw, 0.94 * tw, grid)
        ys = np.linspace(0.06 * th, 0.94 * th, grid)
        table_pts = np.array([[x, y] for y in ys for x in xs], dtype=float)
        pixel_pts = apply_homography(self._H_table_to_pixel, table_pts)
        if noise_px > 0:
            pixel_pts = pixel_pts + np.random.default_rng(seed).normal(
                0, noise_px, pixel_pts.shape
            )
        return pixel_pts, table_pts

    # ------------------------------------------------------------ rendering

    def render(self, seed: int | None = None) -> np.ndarray:
        """Render an (H, W, 3) uint8 RGB frame.

        Every pixel is inverse-mapped into table coordinates and tested against
        each block's rectangle there. Doing it this way rather than drawing
        rotated rectangles in image space means the perspective distortion is
        genuinely present in the image and the pipeline has to undo it.
        """
        rng = np.random.default_rng(self.seed if seed is None else seed)
        W, H = self.image_size
        vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        px = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(float)
        table = apply_homography(self.H_pixel_to_table, px)
        tx, ty = table[:, 0], table[:, 1]

        img = np.empty((H * W, 3), dtype=float)
        img[:] = self.table_color

        tw, th = self.table_size_mm
        off_table = (tx < 0) | (tx > tw) | (ty < 0) | (ty > th)
        img[off_table] = (120, 118, 116)

        for b in self.blocks:
            c, s = np.cos(-b.theta), np.sin(-b.theta)
            dx, dy = tx - b.x, ty - b.y
            lx, ly = c * dx - s * dy, s * dx + c * dy
            h = b.size / 2.0
            inside = (np.abs(lx) <= h) & (np.abs(ly) <= h)
            img[inside] = PALETTE[b.color]

        img = img.reshape(H, W, 3)

        # Multiplicative lighting: bright near the lamp at the top-left corner,
        # falling off across the frame. This is what breaks a naive fixed-value
        # threshold, and why the detector works in hue and saturation instead.
        gy, gx = np.mgrid[0:H, 0:W]
        radial = np.sqrt((gx / W - 0.25) ** 2 + (gy / H - 0.2) ** 2)
        light = 1.0 + self.lighting_gradient * (0.5 - radial)
        img *= light[..., None]

        if self.speckle_count:
            ys = rng.integers(0, H, self.speckle_count)
            xs = rng.integers(0, W, self.speckle_count)
            for y, x in zip(ys, xs):
                img[max(0, y - 1): y + 2, max(0, x - 1): x + 2] = PALETTE["red"]

        if self.noise_sigma:
            img += rng.normal(0, self.noise_sigma, img.shape)

        return np.clip(img, 0, 255).astype(np.uint8)

    # --------------------------------------------------------- ground truth

    @property
    def targets(self) -> list[Block]:
        return [b for b in self.blocks if not b.is_obstacle]

    @property
    def obstacles(self) -> list[Block]:
        return [b for b in self.blocks if b.is_obstacle]
