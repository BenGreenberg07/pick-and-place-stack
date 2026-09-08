import numpy as np
import pytest

from ppstack.frames import Point2D
from ppstack.kinematics.arm import JointLimitViolation, PlanarArm
from ppstack.kinematics.ik import IKUnreachable, solve, solve_2link, solve_dls
from ppstack.kinematics.trajectory import follow_path, interpolate_cartesian, quintic_scaling

ARM2 = PlanarArm((200.0, 170.0))
ARM3 = PlanarArm((170.0, 130.0, 70.0), ((-2.9, 2.9), (-2.6, 2.6), (-2.6, 2.6)))


def test_forward_kinematics_known_pose():
    tip = ARM2.forward([0.0, 0.0])
    assert (tip.x, tip.y) == pytest.approx((370.0, 0.0))
    tip = ARM2.forward([np.pi / 2, 0.0])
    assert (tip.x, tip.y) == pytest.approx((0.0, 370.0), abs=1e-9)


def test_reach_annulus():
    assert ARM2.max_reach == pytest.approx(370.0)
    assert ARM2.min_reach == pytest.approx(30.0)
    assert ARM3.min_reach == pytest.approx(0.0)


@pytest.mark.parametrize("seed", range(15))
def test_jacobian_matches_finite_differences(seed):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-2.0, 2.0, ARM3.n_joints)
    J = ARM3.jacobian(q)
    h = 1e-6
    for i in range(ARM3.n_joints):
        dq = np.zeros(ARM3.n_joints)
        dq[i] = h
        a, b = ARM3.forward(q + dq), ARM3.forward(q - dq)
        assert J[0, i] == pytest.approx((a.x - b.x) / (2 * h), abs=1e-4)
        assert J[1, i] == pytest.approx((a.y - b.y) / (2 * h), abs=1e-4)


def test_manipulability_vanishes_at_the_stretched_singularity():
    assert ARM2.manipulability([0.3, 0.0]) == pytest.approx(0.0, abs=1e-9)
    assert ARM2.manipulability([0.3, 1.0]) > 1000.0


def test_two_link_ik_roundtrips_over_the_workspace():
    rng = np.random.default_rng(0)
    for _ in range(400):
        q = rng.uniform(-np.pi, np.pi, 2)
        target = ARM2.fk_point(q)
        sols = solve_2link(ARM2, target)
        assert sols
        for s in sols:
            achieved = ARM2.forward(s.q)
            assert achieved.x == pytest.approx(target.x, abs=1e-8)
            assert achieved.y == pytest.approx(target.y, abs=1e-8)


def test_two_link_ik_returns_both_elbow_branches():
    sols = solve_2link(ARM2, Point2D(250.0, 90.0, "base"))
    assert {s.branch for s in sols} == {"elbow_up", "elbow_down"}
    assert np.sign(sols[0].q[1]) != np.sign(sols[1].q[1])


def test_two_link_ik_branches_coincide_when_stretched():
    # Exactly at the outer boundary the elbow angle is zero and there is one
    # solution, not two. Returning a duplicate would break branch selection.
    sols = solve_2link(ARM2, Point2D(370.0, 0.0, "base"))
    assert len(sols) == 1


@pytest.mark.parametrize("point", [(500.0, 0.0), (0.0, 400.0), (10.0, 5.0)])
def test_ik_rejects_unreachable_targets(point):
    with pytest.raises(IKUnreachable):
        solve_2link(ARM2, Point2D(*point, "base"))


def test_joint_limits_can_make_a_reachable_target_infeasible():
    narrow = PlanarArm((200.0, 170.0), ((0.0, 0.2), (0.0, 0.2)))
    target = Point2D(0.0, -300.0, "base")  # reachable geometrically, not with these limits
    assert narrow.min_reach <= 300.0 <= narrow.max_reach
    with pytest.raises(IKUnreachable):
        solve(narrow, target)


def test_limit_violation_reports_the_offending_joints():
    with pytest.raises(JointLimitViolation) as exc:
        ARM3.check_limits([0.0, 5.0, 0.0])
    assert "[1]" in str(exc.value)


def test_dls_solves_the_redundant_arm():
    rng = np.random.default_rng(2)
    errors = []
    for _ in range(200):
        q = rng.uniform(-2.4, 2.4, 3)
        target = ARM3.fk_point(q)
        errors.append(solve_dls(ARM3, target).position_error)
    assert max(errors) < 1e-2


def test_dls_nullspace_keeps_joints_away_from_their_limits():
    """The redundancy is spent on posture, so a solution found with the
    nullspace term should sit further inside the limits than one without."""
    target = Point2D(180.0, 140.0, "base")
    seed = np.array([2.5, -2.4, 2.4])
    with_ns = solve_dls(ARM3, target, q_seed=seed, nullspace_gain=0.4, restarts=0)
    without = solve_dls(ARM3, target, q_seed=seed, nullspace_gain=0.0, restarts=0)
    centers = ARM3.limit_centers
    assert np.abs(with_ns.q - centers).sum() < np.abs(without.q - centers).sum()


def test_seeding_prevents_elbow_branch_flips():
    """Independent per-waypoint solves can hop branches; seeding stops it."""
    path = interpolate_cartesian(Point2D(300, 40, "base"), Point2D(-60, 200, "base"), 80)
    traj = follow_path(ARM2, path, q_start=solve(ARM2, path[0]).q)
    assert traj.branch_flips == 0
    assert np.abs(np.diff(traj.q, axis=0)).max() < 0.3


def test_quintic_scaling_endpoints_are_smooth():
    s = quintic_scaling(400)
    assert s[0] == pytest.approx(0.0)
    assert s[-1] == pytest.approx(1.0)
    v = np.diff(s)
    a = np.diff(v)
    assert abs(v[0]) < 1e-4 and abs(v[-1]) < 1e-4      # zero velocity at the ends
    assert abs(a[0]) < 1e-5 and abs(a[-1]) < 1e-5      # zero acceleration too


def test_follow_path_reports_the_failing_waypoint():
    path = interpolate_cartesian(Point2D(300, 0, "base"), Point2D(900, 0, "base"), 10)
    with pytest.raises(IKUnreachable) as exc:
        follow_path(ARM2, path)
    assert "waypoint" in str(exc.value)
