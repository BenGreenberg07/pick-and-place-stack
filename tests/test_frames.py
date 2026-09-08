import math

import numpy as np
import pytest

from ppstack.frames import FrameMismatch, Point2D, Pose2D, RigidTransform2D


def test_frame_tag_is_enforced():
    p = Point2D(1.0, 2.0, "pixel")
    with pytest.raises(FrameMismatch):
        p.require("table")
    with pytest.raises(FrameMismatch):
        p.distance_to(Point2D(1.0, 2.0, "table"))
    with pytest.raises(FrameMismatch):
        Pose2D(0, 0, 0, "table").require("base")


def test_unknown_frame_rejected():
    with pytest.raises(ValueError):
        Point2D(0, 0, "camera_link")


@pytest.mark.parametrize("seed", range(20))
def test_rigid_transform_roundtrips(seed):
    rng = np.random.default_rng(seed)
    T = RigidTransform2D(*rng.uniform(-200, 200, 2), rng.uniform(-math.pi, math.pi),
                         "table", "base")
    p = Point2D(*rng.uniform(-300, 300, 2), "table")
    q = T.apply(p)
    assert q.frame == "base"
    for back in (T.apply_inverse(q), T.inverse().apply(q)):
        assert back.frame == "table"
        assert back.x == pytest.approx(p.x, abs=1e-9)
        assert back.y == pytest.approx(p.y, abs=1e-9)


def test_rigid_transform_preserves_distance():
    T = RigidTransform2D(37.0, -12.0, 0.9, "table", "base")
    a, b = Point2D(10, 20, "table"), Point2D(-40, 75, "table")
    assert T.apply(a).distance_to(T.apply(b)) == pytest.approx(a.distance_to(b))


def test_transform_rejects_wrong_input_frame():
    T = RigidTransform2D(0, 0, 0, "table", "base")
    with pytest.raises(FrameMismatch):
        T.apply(Point2D(0, 0, "base"))
