import numpy as np
import pytest

from ppstack.benchmark import heuristic_comparison, localisation_accuracy
from ppstack.frames import Point2D
from ppstack.perception.detector import BlockDetector
from ppstack.perception.scene import Block, SceneSpec
from ppstack.pipeline import PickAndPlacePipeline, default_calibration
from ppstack.scenarios import (
    BOUNDS, SCENARIOS, TABLE_TO_BASE, default_arm, home_configuration, scene,
)

DETECTOR = BlockDetector(backend="numpy")


def _pipeline(sc, noise_px=0.5, **kwargs):
    arm = default_arm()
    cal = default_calibration(sc, table_to_base=TABLE_TO_BASE, bounds=BOUNDS, noise_px=noise_px)
    return arm, cal, PickAndPlacePipeline(arm, cal, DETECTOR, **kwargs)


@pytest.mark.parametrize("scenario", ["clear", "detour", "gap", "clutter"])
def test_pipeline_runs_end_to_end(scenario):
    sc = scene(scenario)
    arm, cal, pipe = _pipeline(sc)
    result = pipe.run(sc.render(), target_color="red", q_start=home_configuration(arm))
    assert result.ok, result.summary()
    assert [r.stage for r in result.reports] == ["perception", "transform", "planning", "kinematics"]
    assert all(r.ok for r in result.reports)
    assert result.trajectory is not None and len(result.trajectory) > 5


@pytest.mark.parametrize("scenario", ["clear", "detour", "gap", "clutter"])
def test_target_is_localised_to_within_two_millimetres(scenario):
    sc = scene(scenario)
    arm, cal, pipe = _pipeline(sc)
    result = pipe.run(sc.render(), target_color="red", q_start=home_configuration(arm))
    truth = min(
        (b for b in sc.targets if b.color == "red"),
        key=lambda b: (b.x - result.target.position.x) ** 2 + (b.y - result.target.position.y) ** 2,
    )
    err = np.hypot(truth.x - result.target.position.x, truth.y - result.target.position.y)
    assert err < 2.0, f"{scenario}: localisation error {err:.2f} mm"


def test_grasp_angle_matches_ground_truth_modulo_a_quarter_turn():
    sc = scene("clear")
    arm, cal, pipe = _pipeline(sc)
    result = pipe.run(sc.render(), target_color="red", q_start=home_configuration(arm))
    truth = [b for b in sc.targets if b.color == "red"][0]
    d = (result.target.grasp_angle_rad - truth.theta + np.pi / 4) % (np.pi / 2) - np.pi / 4
    assert abs(np.degrees(d)) < 3.0
    assert result.target.grasp_confidence > 0.3


def test_arm_tip_actually_reaches_the_block():
    """The point of the whole chain: the last joint solution, pushed back
    through forward kinematics and the calibration, lands on the block."""
    sc = scene("detour")
    arm, cal, pipe = _pipeline(sc)
    result = pipe.run(sc.render(), target_color="red", q_start=home_configuration(arm))
    landed = cal.base_to_table(arm.forward(result.trajectory.q[-1]).point)
    truth = [b for b in sc.targets if b.color == "red"][0]
    assert np.hypot(landed.x - truth.x, landed.y - truth.y) < 2.0


def test_planned_path_never_enters_an_obstacle():
    sc = scene("gap")
    arm, cal, pipe = _pipeline(sc)
    result = pipe.run(sc.render(), target_color="red", q_start=home_configuration(arm))
    grid = result.grid
    # The final waypoint is snapped onto the block itself, which the grid marks
    # as the goal rather than as free space, so it is excluded.
    for p in result.waypoints[:-1]:
        assert grid.is_free(grid.to_cell(p)), f"path enters an obstacle at {p}"


def test_obstacle_forces_a_detour_rather_than_a_straight_line():
    sc = scene("detour")
    arm, cal, pipe = _pipeline(sc)
    result = pipe.run(sc.render(), target_color="red", q_start=home_configuration(arm))
    start, goal = result.waypoints[0], result.waypoints[-1]
    from ppstack.planning.smoothing import path_length_mm

    assert path_length_mm(result.waypoints) > start.distance_to(goal) * 1.05


def test_trajectory_is_continuous_in_joint_space():
    sc = scene("clutter")
    arm, cal, pipe = _pipeline(sc)
    result = pipe.run(sc.render(), target_color="red", q_start=home_configuration(arm))
    assert result.trajectory.branch_flips == 0
    assert np.abs(np.diff(result.trajectory.q, axis=0)).max() < 0.5


# --------------------------------------------------------------- failure modes


def test_unreachable_target_fails_in_the_transform_stage():
    """Attribution matters: this is a geometry problem, not an IK problem, and
    the pipeline should say so before the solver ever sees the target."""
    sc = scene("unreachable")
    arm, cal, pipe = _pipeline(sc)
    result = pipe.run(sc.render(), target_color="red", q_start=home_configuration(arm))
    assert not result.ok
    assert result.failure_stage == "transform"
    assert "reachable annulus" in result.failure


def test_missing_colour_fails_in_perception_with_what_was_seen():
    sc = scene("clear")
    arm, cal, pipe = _pipeline(sc)
    result = pipe.run(sc.render(), target_color="yellow", q_start=home_configuration(arm))
    assert not result.ok and result.failure_stage == "transform"
    assert "no yellow object" in result.failure and "blue" in result.failure


def test_blank_frame_fails_in_perception():
    sc = scene("clear")
    arm, cal, pipe = _pipeline(sc)
    grey = np.full((480, 640, 3), 150, dtype=np.uint8)
    result = pipe.run(grey, q_start=home_configuration(arm))
    assert not result.ok and result.failure_stage == "perception"


def test_bad_calibration_is_refused_rather_than_acted_on():
    sc = scene("clear")
    arm, cal, pipe = _pipeline(sc, noise_px=12.0)
    result = pipe.run(sc.render(), target_color="red", q_start=home_configuration(arm))
    assert not result.ok
    assert "recalibrate" in result.failure


def test_walled_off_goal_fails_in_planning():
    wall = [Block(220.0, y, 0.0, 60.0, "obstacle") for y in (20, 80, 140, 200, 260)]
    sc = SceneSpec(blocks=[Block(340.0, 150.0, 0.2, 46.0, "red"), *wall])
    arm, cal, pipe = _pipeline(sc)
    result = pipe.run(sc.render(), target_color="red", q_start=home_configuration(arm))
    assert not result.ok and result.failure_stage == "planning"


def test_a_failed_run_still_reports_the_stages_that_worked():
    sc = scene("unreachable")
    arm, cal, pipe = _pipeline(sc)
    result = pipe.run(sc.render(), target_color="red", q_start=home_configuration(arm))
    assert result.reports[0].ok and result.reports[0].stage == "perception"
    assert not result.reports[-1].ok
    assert "transform" in result.summary()


# -------------------------------------------------------------- the benchmarks


def test_benchmark_accuracy_stays_under_a_millimetre_and_a_half():
    stats = localisation_accuracy(trials=8, seed=11)
    assert stats["position_mm"].n >= 6
    assert stats["position_mm"].mean < 1.5
    assert stats["angle_deg"].mean < 3.0


def test_benchmark_confirms_admissible_heuristics_are_optimal():
    solved, acc = heuristic_comparison(trials=8, seed=5)
    assert solved >= 4
    for name in ("zero", "euclidean", "octile"):
        assert max(acc[name]["cost_ratio"]) == pytest.approx(1.0, abs=1e-9)
    # And that the informed ones do less work for the same answer.
    assert np.mean(acc["octile"]["expanded"]) < np.mean(acc["zero"]["expanded"])
