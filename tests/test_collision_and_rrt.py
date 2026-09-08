import numpy as np
import pytest

from ppstack.frames import Point2D
from ppstack.kinematics.ik import enumerate_3link, enumerate_solutions, solve_collision_free
from ppstack.pipeline import default_calibration
from ppstack.planning.collision import ArmCollisionChecker, first_clear_solution
from ppstack.planning.occupancy import build_grid
from ppstack.planning.rrt import RRTFailed, RRTStar, path_cost, shortcut_configs
from ppstack.perception.scene import Block, SceneSpec
from ppstack.scenarios import BOUNDS, TABLE_TO_BASE, default_arm, two_link_arm


@pytest.fixture(scope="module")
def world():
    sc = SceneSpec(blocks=[Block(120.0, 150.0, 0.0, 56.0, "obstacle")])
    cal = default_calibration(sc, table_to_base=TABLE_TO_BASE, bounds=BOUNDS)
    arm = default_arm()
    grid = build_grid(
        BOUNDS, [(Point2D(120.0, 150.0, "table"), 40.0)],
        resolution_mm=5.0, clearance_mm=22.0,
    )
    return arm, cal, grid


# ------------------------------------------------------- the solution manifold


def test_enumerate_3link_covers_the_self_motion_manifold(world):
    arm, cal, _ = world
    target = cal.table_to_base.apply(Point2D(60.0, 250.0, "table"))
    sols = enumerate_3link(arm, target)
    assert len(sols) > 20
    # Every one reaches the target exactly, and they are genuinely different.
    for s in sols:
        tip = arm.forward(s.q)
        assert tip.x == pytest.approx(target.x, abs=1e-6)
        assert tip.y == pytest.approx(target.y, abs=1e-6)
    spread = np.vstack([s.q for s in sols]).std(axis=0)
    assert spread.min() > 0.05


def test_manifold_is_sorted_best_conditioned_first(world):
    arm, cal, _ = world
    target = cal.table_to_base.apply(Point2D(200.0, 200.0, "table"))
    sols = enumerate_3link(arm, target)
    m = [s.manipulability for s in sols]
    assert m == sorted(m, reverse=True)


def test_enumerate_solutions_dispatches_on_link_count():
    arm2 = two_link_arm()
    sols = enumerate_solutions(arm2, Point2D(250.0, 90.0, "base"))
    assert 1 <= len(sols) <= 2
    assert all(s.q.shape == (2,) for s in sols)


def test_enumerate_3link_rejects_the_wrong_arm():
    with pytest.raises(ValueError):
        enumerate_3link(two_link_arm(), Point2D(100.0, 0.0, "base"))


# ------------------------------------------------------------------ collision


def test_a_large_fraction_of_reaching_solutions_collide(world):
    """The headline for the whole module: reaching the target is not the same
    as being allowed to."""
    arm, cal, grid = world
    checker = ArmCollisionChecker(arm, grid, cal, link_radius_mm=10.0)
    target = cal.table_to_base.apply(Point2D(60.0, 250.0, "table"))
    sols = enumerate_3link(arm, target)
    hits = sum(1 for s in sols if checker.collides(s.q))
    assert 0 < hits < len(sols)          # some collide, some do not
    assert hits / len(sols) > 0.4        # and it is not a rare corner case


def test_collision_checker_uses_its_own_inflation_radius(world):
    arm, cal, grid = world
    thin = ArmCollisionChecker(arm, grid, cal, link_radius_mm=2.0)
    fat = ArmCollisionChecker(arm, grid, cal, link_radius_mm=45.0)
    assert fat.grid.blocked.sum() > thin.grid.blocked.sum()
    assert thin.grid.raw.sum() == fat.grid.raw.sum()


def test_inflate_grows_from_raw_not_from_the_already_inflated_grid(world):
    _, _, grid = world
    once = grid.inflate(10.0)
    twice = once.inflate(10.0)
    assert once.blocked.sum() == twice.blocked.sum()


def test_first_clear_solution_skips_the_colliding_branch(world):
    arm, cal, grid = world
    checker = ArmCollisionChecker(arm, grid, cal, link_radius_mm=10.0)
    target = cal.table_to_base.apply(Point2D(60.0, 250.0, "table"))
    sols = enumerate_3link(arm, target)
    chosen = first_clear_solution(sols, checker)
    assert chosen is not None
    assert not checker.collides(chosen.q)
    assert checker.collides(sols[0].q) or chosen is sols[0]


def test_solve_collision_free_returns_none_when_everything_collides(world):
    arm, cal, _ = world
    walled = build_grid(
        BOUNDS,
        [(Point2D(x, y, "table"), 60.0) for x in (80, 200, 320) for y in (80, 200)],
        resolution_mm=5.0, clearance_mm=25.0,
    )
    checker = ArmCollisionChecker(arm, walled, cal, link_radius_mm=14.0)
    target = cal.table_to_base.apply(Point2D(200.0, 140.0, "table"))
    assert solve_collision_free(arm, target, checker) is None


def test_solve_collision_free_prefers_staying_near_the_seed(world):
    arm, cal, grid = world
    checker = ArmCollisionChecker(arm, grid, cal, link_radius_mm=10.0)
    target = cal.table_to_base.apply(Point2D(320.0, 200.0, "table"))
    anchor = solve_collision_free(arm, target, checker)
    assert anchor is not None
    near = solve_collision_free(arm, target, checker, q_seed=anchor.q, travel_weight=5.0)
    far = solve_collision_free(arm, target, checker, q_seed=anchor.q, travel_weight=0.0)
    assert np.abs(near.q - anchor.q).sum() <= np.abs(far.q - anchor.q).sum() + 1e-9


def test_path_clear_rejects_an_edge_that_sweeps_through_an_obstacle(world):
    arm, cal, grid = world
    checker = ArmCollisionChecker(arm, grid, cal, link_radius_mm=10.0)
    left = cal.table_to_base.apply(Point2D(60.0, 250.0, "table"))
    right = cal.table_to_base.apply(Point2D(340.0, 80.0, "table"))
    a = solve_collision_free(arm, left, checker)
    b = solve_collision_free(arm, right, checker)
    assert a is not None and b is not None
    assert not checker.collides(a.q) and not checker.collides(b.q)
    # Both endpoints are legal; the straight joint-space line between them need
    # not be, which is exactly why edges get checked and not just nodes.
    straight = checker.path_clear(a.q, b.q)
    midpoint_ok = not checker.collides(0.5 * (a.q + b.q))
    assert straight == (straight and midpoint_ok)


# ------------------------------------------------------------------- planning


def _clear_configs(arm, cal, checker, xy, k=6):
    base = cal.table_to_base.apply(Point2D(float(xy[0]), float(xy[1]), "table"))
    return [s.q for s in enumerate_3link(arm, base) if not checker.collides(s.q)][:k]


def test_rrt_star_finds_a_collision_free_path(world):
    arm, cal, grid = world
    checker = ArmCollisionChecker(arm, grid, cal, link_radius_mm=10.0)
    starts = _clear_configs(arm, cal, checker, (60, 250))
    goals = _clear_configs(arm, cal, checker, (340, 80))
    assert starts and goals
    result = RRTStar(arm, checker, seed=3).plan(
        starts[0], np.vstack(goals), max_iterations=900, time_budget_s=20.0
    )
    assert len(result.path) >= 2
    for i in range(len(result.path) - 1):
        assert checker.path_clear(result.path[i], result.path[i + 1])
    assert result.cost > 0 and result.collision_checks > 0


def test_rrt_star_beats_plain_rrt_on_cost(world):
    arm, cal, grid = world
    checker = ArmCollisionChecker(arm, grid, cal, link_radius_mm=10.0)
    starts = _clear_configs(arm, cal, checker, (60, 250))
    goals = _clear_configs(arm, cal, checker, (340, 80))
    star = RRTStar(arm, checker, seed=5).plan(
        starts[0], np.vstack(goals), max_iterations=900, time_budget_s=20.0
    )
    plain = RRTStar(arm, checker, seed=5).plan(
        starts[0], np.vstack(goals), max_iterations=900, time_budget_s=20.0,
        stop_on_first=True,
    )
    assert star.cost <= plain.cost + 1e-9
    assert star.nodes >= plain.nodes


def test_rrt_refuses_a_start_that_is_already_in_collision(world):
    arm, cal, grid = world
    checker = ArmCollisionChecker(arm, grid, cal, link_radius_mm=10.0)
    target = cal.table_to_base.apply(Point2D(60.0, 250.0, "table"))
    colliding = next(s.q for s in enumerate_3link(arm, target) if checker.collides(s.q))
    with pytest.raises(RRTFailed, match="start"):
        RRTStar(arm, checker).plan(colliding, np.zeros((1, 3)))


def test_rrt_refuses_when_every_goal_collides(world):
    arm, cal, grid = world
    checker = ArmCollisionChecker(arm, grid, cal, link_radius_mm=10.0)
    starts = _clear_configs(arm, cal, checker, (340, 80))
    target = cal.table_to_base.apply(Point2D(60.0, 250.0, "table"))
    bad = [s.q for s in enumerate_3link(arm, target) if checker.collides(s.q)][:3]
    assert starts and bad
    with pytest.raises(RRTFailed, match="goal"):
        RRTStar(arm, checker).plan(starts[0], np.vstack(bad))


def test_shortcutting_lowers_cost_and_stays_collision_free(world):
    arm, cal, grid = world
    checker = ArmCollisionChecker(arm, grid, cal, link_radius_mm=10.0)
    starts = _clear_configs(arm, cal, checker, (60, 250))
    goals = _clear_configs(arm, cal, checker, (340, 80))
    result = RRTStar(arm, checker, seed=7).plan(
        starts[0], np.vstack(goals), max_iterations=800, time_budget_s=20.0
    )
    smoothed = shortcut_configs(result.path, checker)
    assert path_cost(smoothed) <= path_cost(result.path) + 1e-9
    for i in range(len(smoothed) - 1):
        assert checker.path_clear(smoothed[i], smoothed[i + 1])
