"""Prebuilt scenes, so the demo and the tests exercise the same situations."""

from __future__ import annotations

import numpy as np

from .frames import RigidTransform2D
from .kinematics.arm import PlanarArm
from .perception.scene import Block, SceneSpec
from .transforms.calibration import WorkspaceBounds

# The arm base sits 200 mm to the right of the table origin and 60 mm behind
# its front edge, rotated a quarter turn so its zero direction points across
# the table. Nothing about this is special; it is here so that the table and
# base frames genuinely differ and the transform chain has to be right.
TABLE_TO_BASE = RigidTransform2D(tx=200.0, ty=-60.0, theta=np.pi / 2, parent="table", child="base")

BOUNDS = WorkspaceBounds(x_min=10.0, x_max=390.0, y_min=10.0, y_max=290.0)


def default_arm() -> PlanarArm:
    """A 3-link arm, chosen so the redundancy resolver is actually exercised."""
    return PlanarArm(
        link_lengths=(170.0, 130.0, 70.0),
        joint_limits=((-2.9, 2.9), (-2.6, 2.6), (-2.6, 2.6)),
    )


def two_link_arm() -> PlanarArm:
    return PlanarArm(link_lengths=(200.0, 170.0), joint_limits=((-2.9, 2.9), (-2.7, 2.7)))


# Home pose, expressed as a point in the arm's own frame rather than as joint
# angles, so it stays meaningful if the link lengths change. It sits inside the
# table so the planner always has a legal start cell.
HOME_BASE_XY = (150.0, 90.0)


def home_configuration(arm: PlanarArm) -> np.ndarray:
    """Joint angles that park the tip at HOME_BASE_XY."""
    from .frames import Point2D
    from .kinematics.ik import solve

    return solve(arm, Point2D(*HOME_BASE_XY, "base")).q


SCENARIOS: dict[str, list[Block]] = {
    # A clear shot: nothing between the arm and the block.
    "clear": [
        Block(300.0, 210.0, 0.35, 46.0, "red"),
        Block(80.0, 60.0, -0.20, 44.0, "blue"),
    ],
    # An obstacle sitting exactly on the straight line from home to the block,
    # so a planner that ignores it would drive through it.
    "detour": [
        Block(330.0, 230.0, 0.55, 46.0, "red"),
        Block(220.0, 160.0, 0.00, 70.0, "obstacle"),
        Block(60.0, 240.0, -0.35, 44.0, "green"),
    ],
    # A wall with one gap in it. The straight line is blocked, and the only
    # route is up through the gap and back down, which is the case a greedy
    # best-first search gets wrong and A* gets right.
    "gap": [
        Block(330.0, 80.0, 0.10, 42.0, "red"),
        Block(220.0, 40.0, 0.0, 60.0, "obstacle"),
        Block(220.0, 100.0, 0.0, 60.0, "obstacle"),
        Block(220.0, 255.0, 0.0, 60.0, "obstacle"),
        Block(70.0, 250.0, 0.4, 44.0, "yellow"),
    ],
    # Clutter: several same-coloured candidates plus distractors.
    "clutter": [
        Block(310.0, 240.0, 0.6, 50.0, "red"),
        Block(120.0, 70.0, -0.4, 34.0, "red"),
        Block(230.0, 120.0, 0.2, 55.0, "obstacle"),
        Block(70.0, 220.0, 0.15, 44.0, "blue"),
        Block(340.0, 90.0, -0.6, 40.0, "green"),
        Block(200.0, 250.0, 0.3, 38.0, "yellow"),
    ],
    # The block is out past the arm's reach: the pipeline should say so, in the
    # transform stage, rather than handing the IK an impossible target.
    "unreachable": [
        Block(20.0, 285.0, 0.0, 40.0, "red"),
        Block(200.0, 150.0, 0.0, 50.0, "obstacle"),
    ],
}


def scene(name: str, **overrides) -> SceneSpec:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; have {sorted(SCENARIOS)}")
    return SceneSpec(blocks=list(SCENARIOS[name]), **overrides)
