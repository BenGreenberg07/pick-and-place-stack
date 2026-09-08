"""Inverse kinematics: closed form for the 2-link case, damped least squares
for the redundant case.

Two solvers live here on purpose. The analytic one is exact, instant, and
enumerates both elbow branches, but only exists for two links. The numeric one
handles any number of links, resolves the extra degrees of freedom with a
nullspace term, and degrades gracefully at singularities. The pipeline uses
the analytic solver when it can and falls back to the numeric one otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frames import Point2D
from .arm import PlanarArm


class IKUnreachable(ValueError):
    """Raised when no joint vector puts the tip on the requested target."""


@dataclass
class IKSolution:
    q: np.ndarray
    branch: str            # "elbow_up", "elbow_down", or "numeric"
    position_error: float  # mm between the achieved tip and the target
    iterations: int        # 0 for the closed-form solver
    manipulability: float


# --------------------------------------------------------------------- 2-link


def _wrap(a: np.ndarray | float):
    """Wrap angles into (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def solve_2link(arm: PlanarArm, target: Point2D, *, tol: float = 1e-6) -> list[IKSolution]:
    """All closed-form solutions for a 2-link planar arm, Law of Cosines.

    With L1, L2 the link lengths and r the distance from the base to the target,
    the triangle (base, elbow, tip) has the target-to-base side of length r, so

        cos(pi - q1) = (L1^2 + L2^2 - r^2) / (2 L1 L2)

    which rearranges to the standard

        cos(q1) = (r^2 - L1^2 - L2^2) / (2 L1 L2)

    The two signs of q1 are the elbow-up and elbow-down branches. q0 is then the
    bearing to the target minus the angle the first link is offset by inside
    that triangle.

    Returns every solution that is inside the joint limits, ordered elbow-down
    first for determinism. Raises IKUnreachable if the target is outside the
    annulus the arm can reach, and returns an empty list if the target is
    geometrically reachable but every branch violates a joint limit.
    """
    if arm.n_joints != 2:
        raise ValueError("solve_2link requires exactly two links")
    target.require("base")
    L1, L2 = arm.link_lengths
    x, y = target.x, target.y
    r = float(np.hypot(x, y))

    if r > L1 + L2 + tol:
        raise IKUnreachable(
            f"target at r={r:.2f} mm is beyond the arm's reach of {L1 + L2:.2f} mm"
        )
    if r < abs(L1 - L2) - tol:
        raise IKUnreachable(
            f"target at r={r:.2f} mm is inside the unreachable core of "
            f"radius {abs(L1 - L2):.2f} mm around the base"
        )

    # Clip absorbs the float error at the two workspace boundaries, where the
    # cosine lands a few ulps outside [-1, 1] and would otherwise give a NaN.
    cos_q1 = np.clip((r * r - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1.0, 1.0)
    q1_mag = float(np.arccos(cos_q1))

    solutions: list[IKSolution] = []
    for sign, name in ((+1.0, "elbow_down"), (-1.0, "elbow_up")):
        q1 = sign * q1_mag
        # Offset of link 1 from the base-to-target bearing, inside the triangle.
        q0 = float(np.arctan2(y, x) - np.arctan2(L2 * np.sin(q1), L1 + L2 * np.cos(q1)))
        q = _wrap(np.array([q0, q1]))
        if not arm.within_limits(q):
            continue
        tip = arm.forward(q)
        solutions.append(
            IKSolution(
                q=q,
                branch=name,
                position_error=float(np.hypot(tip.x - x, tip.y - y)),
                iterations=0,
                manipulability=arm.manipulability(q),
            )
        )
        if q1_mag < tol or abs(q1_mag - np.pi) < tol:
            break  # fully stretched or fully folded: the branches coincide
    return solutions


# ------------------------------------------------------- damped least squares


def solve_dls(
    arm: PlanarArm,
    target: Point2D,
    *,
    q_seed=None,
    restarts: int = 4,
    rng: np.random.Generator | None = None,
    tol: float = 1e-3,
    max_iterations: int = 200,
    damping: float = 5.0,
    nullspace_gain: float = 0.05,
    step_clip: float = 0.35,
) -> IKSolution:
    """Levenberg-Marquardt inverse kinematics for an arm with any link count.

    Each iteration solves the damped normal equations

        dq = J^T (J J^T + k^2 I)^-1 e

    which is the least-squares step that trades tracking accuracy for bounded
    joint rates. Plain pseudo-inverse IK blows up near singularities because
    J J^T loses rank; the k^2 I term keeps it invertible, so the arm slows down
    through a singularity instead of commanding an enormous joint velocity.

    For a redundant arm (more joints than task dimensions) the remaining
    freedom is spent by projecting a joint-limit-avoidance gradient through the
    nullspace projector (I - J^+ J), which cannot disturb the tip position.

    `q_seed` matters: seeding from the previous waypoint's solution is what
    keeps a Cartesian path from flipping elbow branches partway through.

    Local descent can stall against a joint limit even when a solution exists
    elsewhere in configuration space, so a failed descent is retried from
    `restarts` random seeds before the target is declared unreachable.
    """
    rng = rng or np.random.default_rng(0)
    last_error: IKUnreachable | None = None
    for attempt in range(restarts + 1):
        seed = q_seed if attempt == 0 else rng.uniform(arm._limits[:, 0], arm._limits[:, 1])
        try:
            return _dls_descent(
                arm, target, q_seed=seed, tol=tol, max_iterations=max_iterations,
                damping=damping, nullspace_gain=nullspace_gain, step_clip=step_clip,
            )
        except IKUnreachable as exc:
            last_error = exc
            if "outside the reachable annulus" in str(exc):
                raise
    assert last_error is not None
    raise last_error


def _dls_descent(
    arm: PlanarArm,
    target: Point2D,
    *,
    q_seed=None,
    tol: float,
    max_iterations: int,
    damping: float,
    nullspace_gain: float,
    step_clip: float,
) -> IKSolution:
    """One Levenberg-Marquardt descent from a single seed."""
    target.require("base")
    goal = np.array([target.x, target.y], dtype=float)
    r = float(np.linalg.norm(goal))
    if r > arm.max_reach + 1e-6 or r < arm.min_reach - 1e-6:
        raise IKUnreachable(
            f"target at r={r:.2f} mm is outside the reachable annulus "
            f"[{arm.min_reach:.2f}, {arm.max_reach:.2f}] mm"
        )

    if q_seed is None:
        # A slightly bent seed. Starting from all zeros puts a 2-link arm exactly
        # on its stretched-out singularity, where the first step is degenerate.
        q = np.full(arm.n_joints, 0.2)
        q[0] = float(np.arctan2(goal[1], goal[0]))
    else:
        q = np.asarray(q_seed, dtype=float).copy()
    q = arm.clamp_to_limits(q)

    centers = arm.limit_centers
    spans = arm._limits[:, 1] - arm._limits[:, 0]

    err_norm = np.inf
    it = 0
    for it in range(1, max_iterations + 1):
        tip = arm.forward(q)
        e = goal - np.array([tip.x, tip.y])
        err_norm = float(np.linalg.norm(e))
        if err_norm < tol:
            break

        J = arm.jacobian(q)
        JJt = J @ J.T + (damping**2) * np.eye(2)
        dq = J.T @ np.linalg.solve(JJt, e)

        # The nullspace term is switched off once the tip is essentially on
        # target. It is only first-order tip-preserving, so leaving it running
        # would keep nudging the tip by a fraction of a micron forever and the
        # loop would never satisfy `tol`. Posture is optimised on the way in.
        if arm.n_joints > 2 and nullspace_gain > 0.0 and err_norm > 20 * tol:
            # Gradient of a cost that pushes each joint toward the middle of its
            # range, normalised by span so wide and narrow joints weigh alike.
            grad = -(q - centers) / (0.5 * spans) ** 2
            J_pinv = J.T @ np.linalg.inv(JJt)
            dq = dq + (np.eye(arm.n_joints) - J_pinv @ J) @ (nullspace_gain * grad)

        # Clipping the step keeps a large initial error from throwing the arm
        # across its workspace on iteration one.
        norm = np.linalg.norm(dq)
        if norm > step_clip:
            dq *= step_clip / norm
        q = arm.clamp_to_limits(q + dq)

    q = _wrap(arm.clamp_to_limits(q))
    tip = arm.forward(q)
    err_norm = float(np.hypot(tip.x - goal[0], tip.y - goal[1]))
    if err_norm > max(tol * 10, 1e-3):
        raise IKUnreachable(
            f"damped least squares stalled {err_norm:.4f} mm from the target after "
            f"{it} iterations (likely blocked by a joint limit)"
        )
    return IKSolution(
        q=q,
        branch="numeric",
        position_error=err_norm,
        iterations=it,
        manipulability=arm.manipulability(q),
    )


# ------------------------------------------------------------ branch policy


def solve(
    arm: PlanarArm,
    target: Point2D,
    *,
    q_seed=None,
    prefer: str = "least_travel",
    **kwargs,
) -> IKSolution:
    """Solve IK and pick one solution, closed form where one exists.

    `prefer` chooses between the two elbow branches of a 2-link arm:
        "least_travel"    minimise joint motion from q_seed (default; this is
                          what keeps a trajectory continuous)
        "elbow_up"        force that branch, falling back if it is infeasible
        "elbow_down"      likewise
        "manipulability"  pick the branch furthest from a singularity
    """
    if arm.n_joints == 2:
        sols = solve_2link(arm, target)
        if not sols:
            raise IKUnreachable(
                "target is geometrically reachable but both elbow branches "
                "violate the arm's joint limits"
            )
        if prefer in ("elbow_up", "elbow_down"):
            for s in sols:
                if s.branch == prefer:
                    return s
            return sols[0]
        if prefer == "manipulability":
            return max(sols, key=lambda s: s.manipulability)
        if q_seed is None:
            return max(sols, key=lambda s: s.manipulability)
        seed = np.asarray(q_seed, dtype=float)
        return min(sols, key=lambda s: float(np.abs(_wrap(s.q - seed)).sum()))
    return solve_dls(arm, target, q_seed=q_seed, **kwargs)
