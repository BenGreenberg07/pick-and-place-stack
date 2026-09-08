import numpy as np
import pytest

from ppstack.frames import Point2D
from ppstack.perception.detector import BlockDetector
from ppstack.pipeline import PickAndPlacePipeline, default_calibration
from ppstack.planning.collision import ArmCollisionChecker
from ppstack.scenarios import (
    BOUNDS, TABLE_TO_BASE, default_arm, default_bins, home_configuration, scene,
)
from ppstack.task import Bin, TaskExecutor, sequence_jobs, tour_cost

DETECTOR = BlockDetector(backend="numpy")


@pytest.fixture(scope="module")
def sorted_world():
    sc = scene("sorting")
    cal = default_calibration(sc, table_to_base=TABLE_TO_BASE, bounds=BOUNDS)
    arm = default_arm()
    result = PickAndPlacePipeline(arm, cal, DETECTOR).run(
        sc.render(), q_start=home_configuration(arm)
    )
    return sc, cal, arm, result


# ------------------------------------------------------------------ ordering


class _FakeObject:
    def __init__(self, x, y, color="red"):
        self.position = Point2D(x, y, "table")
        self.color = color
        self.is_obstacle = False
        self.size_mm = 40.0
        self.footprint_table = np.empty((0, 2))


def test_sequencing_never_makes_the_tour_longer():
    rng = np.random.default_rng(0)
    start = Point2D(200.0, 20.0, "table")
    for _ in range(20):
        jobs = [
            (_FakeObject(*rng.uniform(20, 380, 2)),
             Bin("b", *rng.uniform(20, 380, 2)))
            for _ in range(5)
        ]
        assert tour_cost(start, sequence_jobs(start, jobs)) <= tour_cost(start, jobs) + 1e-9


def test_small_instances_are_solved_exactly():
    """Up to six jobs the sequencer enumerates, so it must match brute force."""
    import itertools

    rng = np.random.default_rng(3)
    start = Point2D(200.0, 20.0, "table")
    jobs = [
        (_FakeObject(*rng.uniform(20, 380, 2)), Bin("b", *rng.uniform(20, 380, 2)))
        for _ in range(5)
    ]
    best = min(tour_cost(start, list(p)) for p in itertools.permutations(jobs))
    assert tour_cost(start, sequence_jobs(start, jobs)) == pytest.approx(best)


def test_two_opt_handles_instances_too_big_to_enumerate():
    rng = np.random.default_rng(1)
    start = Point2D(200.0, 20.0, "table")
    jobs = [
        (_FakeObject(*rng.uniform(20, 380, 2)), Bin("b", *rng.uniform(20, 380, 2)))
        for _ in range(9)
    ]
    ordered = sequence_jobs(start, jobs)
    assert len(ordered) == len(jobs)
    assert {id(o) for o, _ in ordered} == {id(o) for o, _ in jobs}
    assert tour_cost(start, ordered) < tour_cost(start, jobs)


def test_sequencing_is_a_no_op_for_one_job():
    start = Point2D(0.0, 0.0, "table")
    jobs = [(_FakeObject(10, 10), Bin("b", 20, 20))]
    assert sequence_jobs(start, jobs) == jobs


# ----------------------------------------------------------------- execution


def test_the_whole_cycle_places_every_block(sorted_world):
    sc, cal, arm, result = sorted_world
    ex = TaskExecutor(arm, cal).run(
        result.objects, default_bins(), q_start=home_configuration(arm)
    )
    assert ex.ok
    assert ex.picks == 6
    assert ex.skipped == []
    # Approach then carry, for each block.
    assert [s.kind for s in ex.segments] == ["approach", "carry"] * 6


def test_sequencing_beats_detection_order_on_the_sorting_scene(sorted_world):
    sc, cal, arm, result = sorted_world
    ex = TaskExecutor(arm, cal).run(
        result.objects, default_bins(), q_start=home_configuration(arm)
    )
    assert ex.tour_mm < ex.tour_mm_detection_order * 0.95


def test_every_solved_pose_is_collision_free(sorted_world):
    """The claim the collision-aware executor exists to make."""
    sc, cal, arm, result = sorted_world
    executor = TaskExecutor(arm, cal)
    ex = executor.run(result.objects, default_bins(), q_start=home_configuration(arm))
    tall = [o for o in result.objects if o.is_obstacle]
    audit = ArmCollisionChecker(
        arm, executor._grid(tall, 0.0, tall_only=True), cal, link_radius_mm=10.0
    )
    for seg in ex.segments:
        for q in seg.trajectory.q:
            assert not audit.collides(q)


def test_the_naive_executor_produces_colliding_poses(sorted_world):
    """The baseline, so the previous test is measuring something real."""
    sc, cal, arm, result = sorted_world
    executor = TaskExecutor(arm, cal, collision_aware=False)
    ex = executor.run(result.objects, default_bins(), q_start=home_configuration(arm))
    tall = [o for o in result.objects if o.is_obstacle]
    audit = ArmCollisionChecker(
        arm, executor._grid(tall, 0.0, tall_only=True), cal, link_radius_mm=10.0
    )
    bad = sum(int(audit.collides(q)) for seg in ex.segments for q in seg.trajectory.q)
    assert bad > 0


def test_flat_blocks_do_not_block_the_linkage(sorted_world):
    """The links pass above the block plane, so only tall obstacles count.

    Treating every block as an arm obstacle is what makes a naive version
    declare the whole workspace unreachable.
    """
    sc, cal, arm, result = sorted_world
    executor = TaskExecutor(arm, cal)
    everything = executor._grid(result.objects, 0.0, tall_only=False)
    tall_only = executor._grid(result.objects, 0.0, tall_only=True)
    assert tall_only.raw.sum() < everything.raw.sum()


def test_the_grid_shrinks_as_blocks_are_removed(sorted_world):
    sc, cal, arm, result = sorted_world
    ex = TaskExecutor(arm, cal).run(
        result.objects, default_bins(), q_start=home_configuration(arm)
    )
    first = ex.segments[0].grid.raw.sum()
    last = ex.segments[-1].grid.raw.sum()
    assert last < first


def test_carrying_a_block_widens_the_required_clearance(sorted_world):
    sc, cal, arm, result = sorted_world
    ex = TaskExecutor(arm, cal).run(
        result.objects, default_bins(), q_start=home_configuration(arm)
    )
    approach = ex.segments[0]
    carry = ex.segments[1]
    # Same obstacles, but the carry grid is inflated further for the cargo.
    assert carry.grid.raw.sum() <= approach.grid.raw.sum()
    assert carry.grid.blocked.sum() >= carry.grid.raw.sum()


def test_a_block_with_no_matching_bin_is_skipped_not_fatal(sorted_world):
    sc, cal, arm, result = sorted_world
    ex = TaskExecutor(arm, cal).run(
        result.objects, [Bin("only-red", 60.0, 35.0, ("red",))],
        q_start=home_configuration(arm),
    )
    assert ex.picks == 2
    assert {name for name, _ in ex.skipped} == {"blue", "green"}
    assert all("no bin accepts" in why for _, why in ex.skipped)


def test_no_pickable_objects_gives_an_empty_execution(sorted_world):
    sc, cal, arm, result = sorted_world
    ex = TaskExecutor(arm, cal).run(
        [o for o in result.objects if o.is_obstacle], default_bins(),
        q_start=home_configuration(arm),
    )
    assert not ex.ok and ex.picks == 0


def test_bin_colour_filter():
    b = Bin("warm", 10.0, 10.0, ("red", "yellow"))
    assert b.takes("red") and not b.takes("blue")
    assert Bin("any", 0.0, 0.0).takes("anything")
