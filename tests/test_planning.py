import numpy as np
import pytest

from ppstack.frames import Point2D
from ppstack.planning.astar import (
    HEURISTICS, NoPathFound, astar, dijkstra_cost, h_manhattan, h_octile,
)
from ppstack.planning.occupancy import OccupancyGrid, OutsideGrid, build_grid
from ppstack.planning.smoothing import (
    line_of_sight, path_length_mm, resample, shortcut, supercover_line,
)
from ppstack.transforms.calibration import WorkspaceBounds

BOUNDS = WorkspaceBounds(0.0, 400.0, 0.0, 300.0)


def _empty(rows=40, cols=60):
    return OccupancyGrid(np.zeros((rows, cols), dtype=bool), BOUNDS, 5.0)


def test_cell_and_world_roundtrip():
    g = _empty()
    for cell in [(0, 0), (10, 25), (39, 59)]:
        assert g.to_cell(g.to_world(cell)) == cell


def test_points_outside_the_workspace_are_rejected():
    with pytest.raises(OutsideGrid):
        _empty().to_cell(Point2D(-10.0, 50.0, "table"))


def test_inflation_grows_obstacles_by_the_clearance():
    g = build_grid(BOUNDS, [(Point2D(200.0, 150.0, "table"), 10.0)],
                   resolution_mm=5.0, clearance_mm=0.0)
    g_inflated = build_grid(BOUNDS, [(Point2D(200.0, 150.0, "table"), 10.0)],
                            resolution_mm=5.0, clearance_mm=25.0)
    assert g_inflated.blocked.sum() > g.blocked.sum()
    # A point 20 mm from the centre is free without clearance and blocked with it.
    probe = Point2D(220.0, 150.0, "table")
    assert g.is_free(g.to_cell(probe))
    assert not g_inflated.is_free(g_inflated.to_cell(probe))


def test_footprints_rasterise_a_wall_as_a_wall():
    """A long thin obstacle modelled as a disc at its centroid is wrong in both
    directions; the footprint path has to keep its actual shape."""
    xs = np.linspace(180.0, 220.0, 200)
    wall = np.stack([xs, np.full_like(xs, 150.0)], axis=1)
    g = build_grid(BOUNDS, footprints=[wall], resolution_mm=5.0, clearance_mm=0.0)
    assert not g.is_free(g.to_cell(Point2D(200.0, 150.0, "table")))
    assert g.is_free(g.to_cell(Point2D(200.0, 200.0, "table")))  # not a fat disc
    assert not g.is_free(g.to_cell(Point2D(182.0, 150.0, "table")))  # still a wall


def test_astar_finds_the_optimal_cost():
    g = _empty()
    for name in HEURISTICS:
        if name == "manhattan":
            continue  # inadmissible on an 8-connected grid, tested separately
        r = astar(g, (2, 2), (35, 55), heuristic=name, tie_breaker=0.0)
        assert r.cost == pytest.approx(dijkstra_cost(g, (2, 2), (35, 55)))


def test_octile_beats_dijkstra_on_expansions():
    g = _empty()
    informed = astar(g, (2, 2), (35, 55), heuristic="octile")
    blind = astar(g, (2, 2), (35, 55), heuristic="zero")
    assert informed.expanded < blind.expanded


def test_manhattan_is_inadmissible_on_an_eight_connected_grid():
    """It charges 2 for a diagonal that costs sqrt(2), so it can overestimate."""
    a, b = (0, 0), (10, 10)
    assert h_manhattan(a, b) > h_octile(a, b)


def test_astar_escapes_a_u_shaped_trap():
    g = _empty(60, 80)
    g.blocked[10:50, 40] = True
    g.blocked[10, 40:70] = True
    g.blocked[49, 40:70] = True
    r = astar(g, (30, 20), (30, 60))
    assert r.path_cells[0] == (30, 20) and r.path_cells[-1] == (30, 60)
    assert all(g.is_free(c) for c in r.path_cells)


def test_walled_off_goal_raises():
    g = _empty(20, 20)
    g.blocked[:, 10] = True
    with pytest.raises(NoPathFound):
        astar(g, (5, 5), (5, 15))


def test_blocked_endpoints_are_rejected():
    g = _empty(20, 20)
    g.blocked[5, 5] = True
    with pytest.raises(OutsideGrid):
        astar(g, (5, 5), (10, 10))


def test_astar_will_not_cut_a_diagonal_corner():
    """Two blocked cells meeting at a corner leave a diagonal slit that a point
    robot could slip through and a real one could not."""
    g = _empty(9, 9)
    g.blocked[4, 5] = True
    g.blocked[5, 4] = True
    r = astar(g, (4, 4), (5, 5))
    assert len(r.path_cells) > 2


def test_nearest_free_nudges_a_goal_out_of_an_obstacle():
    g = build_grid(BOUNDS, [(Point2D(200.0, 150.0, "table"), 20.0)], clearance_mm=10.0)
    buried = g.to_cell(Point2D(200.0, 150.0, "table"))
    assert not g.is_free(buried)
    assert g.is_free(g.nearest_free(buried))


def test_supercover_line_includes_both_corner_cells():
    """Bresenham would skip them and let a line-of-sight test walk through a
    wall diagonally."""
    cells = set(supercover_line((0, 0), (2, 2)))
    assert {(0, 1), (1, 0)} & cells
    assert (1, 1) in cells


def test_line_of_sight_is_blocked_by_a_diagonal_wall():
    g = _empty(9, 9)
    g.blocked[4, 5] = True
    g.blocked[5, 4] = True
    assert not line_of_sight(g, (4, 4), (5, 5))


def test_shortcut_shortens_a_staircase_without_leaving_free_space():
    g = _empty(40, 40)
    raw = astar(g, (1, 1), (35, 35)).path_cells
    short = shortcut(g, raw)
    assert len(short) < len(raw)
    assert short[0] == raw[0] and short[-1] == raw[-1]
    assert all(line_of_sight(g, short[i], short[i + 1]) for i in range(len(short) - 1))


def test_resample_gives_even_spacing():
    pts = [Point2D(0, 0, "table"), Point2D(100, 0, "table"), Point2D(100, 100, "table")]
    out = resample(pts, 5.0)
    steps = [out[i].distance_to(out[i + 1]) for i in range(len(out) - 1)]
    assert max(steps) - min(steps) < 1e-6
    assert path_length_mm(out) == pytest.approx(200.0, abs=1e-6)
