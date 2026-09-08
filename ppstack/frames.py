"""Coordinate frames and frame-tagged poses.

The single most common way a pick-and-place stack goes wrong is that a number
computed in one frame gets used in another. Pixels are y-down, millimetres on
the table are y-up, and the arm's base frame is translated and rotated relative
to the table. Nothing in the type system stops you from subtracting a pixel
from a millimetre, so this module adds the tag by hand and checks it.

Frames used in this project:

    pixel   image coordinates, origin top-left, +x right, +y DOWN, units px
    table   workspace plane, origin at the table's front-left fiducial,
            +x right, +y AWAY from the camera operator, units mm
    base    arm frame, origin at joint 0, +x along the arm's zero direction,
            units mm. Related to `table` by a fixed rigid transform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

FRAMES = ("pixel", "table", "base")


class FrameMismatch(ValueError):
    """Raised when a quantity is used in a frame it was not expressed in."""


def _check(frame: str) -> str:
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r}; expected one of {FRAMES}")
    return frame


@dataclass(frozen=True)
class Point2D:
    """A position tagged with the frame it is expressed in."""

    x: float
    y: float
    frame: str

    def __post_init__(self) -> None:
        _check(self.frame)

    def require(self, frame: str) -> "Point2D":
        if self.frame != frame:
            raise FrameMismatch(
                f"expected a point in frame {frame!r} but got one in {self.frame!r}"
            )
        return self

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, other: "Point2D") -> float:
        if self.frame != other.frame:
            raise FrameMismatch(
                f"cannot take a distance between frames {self.frame!r} and {other.frame!r}"
            )
        return math.hypot(self.x - other.x, self.y - other.y)

    def in_frame(self, frame: str) -> "Point2D":
        """Relabel without transforming. Only for use inside transform code."""
        return replace(self, frame=_check(frame))


@dataclass(frozen=True)
class Pose2D:
    """A position plus a planar orientation, tagged with its frame."""

    x: float
    y: float
    theta: float
    frame: str

    def __post_init__(self) -> None:
        _check(self.frame)

    @property
    def point(self) -> Point2D:
        return Point2D(self.x, self.y, self.frame)

    def require(self, frame: str) -> "Pose2D":
        if self.frame != frame:
            raise FrameMismatch(
                f"expected a pose in frame {frame!r} but got one in {self.frame!r}"
            )
        return self


@dataclass(frozen=True)
class RigidTransform2D:
    """A rigid transform taking points from `parent` into `child`.

    Stored as the rotation and translation that map a point p expressed in
    `parent` to `child`:  p_child = R(theta) @ (p_parent - t)
    """

    tx: float
    ty: float
    theta: float
    parent: str
    child: str

    def apply(self, p: Point2D) -> Point2D:
        p.require(self.parent)
        c, s = math.cos(-self.theta), math.sin(-self.theta)
        dx, dy = p.x - self.tx, p.y - self.ty
        return Point2D(c * dx - s * dy, s * dx + c * dy, self.child)

    def inverse(self) -> "RigidTransform2D":
        """The transform taking points from `child` back into `parent`.

        Forward:  p_child  = R(-theta) (p_parent - t)
        Inverse:  p_parent = R(theta) p_child + t

        Rewriting the inverse in this same (t, theta) form gives
        theta' = -theta and t' = -R(-theta) t.
        """
        c, s = math.cos(self.theta), math.sin(self.theta)
        return RigidTransform2D(
            tx=-(c * self.tx + s * self.ty),
            ty=(s * self.tx - c * self.ty),
            theta=-self.theta,
            parent=self.child,
            child=self.parent,
        )

    def apply_inverse(self, p: Point2D) -> Point2D:
        p.require(self.child)
        c, s = math.cos(self.theta), math.sin(self.theta)
        return Point2D(c * p.x - s * p.y + self.tx, s * p.x + c * p.y + self.ty, self.parent)
