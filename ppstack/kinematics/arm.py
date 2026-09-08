"""Planar serial manipulator: forward kinematics and the analytic Jacobian."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..frames import Point2D, Pose2D


class JointLimitViolation(ValueError):
    """Raised when a joint vector falls outside the arm's mechanical limits."""


@dataclass
class PlanarArm:
    """An n-link planar revolute arm rooted at the origin of the `base` frame.

    Joint angles are relative: q[0] is measured from the base +x axis, and each
    subsequent q[i] is measured from the previous link. That convention keeps
    the 2-link closed-form inverse kinematics in its textbook shape, where q[1]
    is the interior elbow angle straight out of the Law of Cosines.
    """

    link_lengths: tuple[float, ...]
    joint_limits: tuple[tuple[float, float], ...] | None = None
    _limits: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.link_lengths = tuple(float(v) for v in self.link_lengths)
        if len(self.link_lengths) < 2:
            raise ValueError("need at least two links")
        if any(v <= 0 for v in self.link_lengths):
            raise ValueError("link lengths must be positive")
        if self.joint_limits is None:
            self.joint_limits = tuple((-np.pi, np.pi) for _ in self.link_lengths)
        if len(self.joint_limits) != len(self.link_lengths):
            raise ValueError("one (lo, hi) limit pair is required per joint")
        self._limits = np.asarray(self.joint_limits, dtype=float)
        if np.any(self._limits[:, 0] >= self._limits[:, 1]):
            raise ValueError("every joint limit must satisfy lo < hi")

    # ---------------------------------------------------------------- basics

    @property
    def n_joints(self) -> int:
        return len(self.link_lengths)

    @property
    def max_reach(self) -> float:
        return float(sum(self.link_lengths))

    @property
    def min_reach(self) -> float:
        """Radius of the dead zone around the base that the arm cannot enter.

        For a 2-link arm this is |L1 - L2|. For more links, folding the shorter
        links against the longest one gives max(0, L_max - sum(others)).
        """
        longest = max(self.link_lengths)
        rest = sum(self.link_lengths) - longest
        return float(max(0.0, longest - rest))

    def within_limits(self, q) -> bool:
        q = np.asarray(q, dtype=float)
        return bool(np.all(q >= self._limits[:, 0] - 1e-9) and np.all(q <= self._limits[:, 1] + 1e-9))

    def clamp_to_limits(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return np.clip(q, self._limits[:, 0], self._limits[:, 1])

    def check_limits(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if not self.within_limits(q):
            bad = [
                i
                for i in range(self.n_joints)
                if not (self._limits[i, 0] - 1e-9 <= q[i] <= self._limits[i, 1] + 1e-9)
            ]
            raise JointLimitViolation(
                f"joints {bad} outside limits: q={np.round(np.degrees(q), 2).tolist()} deg, "
                f"limits={np.round(np.degrees(self._limits), 2).tolist()} deg"
            )
        return q

    @property
    def limit_centers(self) -> np.ndarray:
        return self._limits.mean(axis=1)

    # ------------------------------------------------------------------ f.k.

    def joint_positions(self, q) -> np.ndarray:
        """(n+1, 2) array of joint origins, starting with the base at (0, 0)."""
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n_joints,):
            raise ValueError(f"expected {self.n_joints} joint angles, got {q.shape}")
        angles = np.cumsum(q)
        pts = np.zeros((self.n_joints + 1, 2))
        for i, (a, L) in enumerate(zip(angles, self.link_lengths)):
            pts[i + 1] = pts[i] + (L * np.cos(a), L * np.sin(a))
        return pts

    def forward(self, q) -> Pose2D:
        """End-effector pose in the `base` frame."""
        q = np.asarray(q, dtype=float)
        pts = self.joint_positions(q)
        return Pose2D(float(pts[-1, 0]), float(pts[-1, 1]), float(np.sum(q)), "base")

    def fk_point(self, q) -> Point2D:
        return self.forward(q).point

    # -------------------------------------------------------------- jacobian

    def jacobian(self, q) -> np.ndarray:
        """(2, n) position Jacobian d(x, y) / dq, analytic.

        Column i is the velocity the tip picks up from a unit rate on joint i,
        which for a planar revolute joint is the tip position measured from that
        joint, rotated by 90 degrees.
        """
        pts = self.joint_positions(q)
        tip = pts[-1]
        J = np.zeros((2, self.n_joints))
        for i in range(self.n_joints):
            r = tip - pts[i]
            J[0, i] = -r[1]
            J[1, i] = r[0]
        return J

    def manipulability(self, q) -> float:
        """Yoshikawa's measure sqrt(det(J J^T)). Zero at a singularity."""
        J = self.jacobian(q)
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))
