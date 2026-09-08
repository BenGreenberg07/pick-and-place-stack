import itertools

import numpy as np
import pytest

from ppstack.perception.assignment import assignment_cost, hungarian
from ppstack.perception.detector import BlockDetector
from ppstack.perception.scene import Block, SceneSpec
from ppstack.perception.tracking import MultiObjectTracker, count_id_switches
from ppstack.pipeline import default_calibration
from ppstack.scenarios import BOUNDS, TABLE_TO_BASE


# ----------------------------------------------------------------- assignment


def _brute_force(cost):
    """Reference optimum by exhaustive search.

    When there are more rows than columns the free choice is *which* rows get
    used, not just which columns, so the matrix is transposed first. Getting
    this wrong makes the reference report a cost the real optimum beats.
    """
    n, m = cost.shape
    if n > m:
        return _brute_force(cost.T)
    return min(
        sum(cost[r, cols[r]] for r in range(n))
        for cols in itertools.permutations(range(m), n)
    )


@pytest.mark.parametrize("seed", range(25))
def test_hungarian_is_optimal(seed):
    rng = np.random.default_rng(seed)
    n, m = int(rng.integers(1, 6)), int(rng.integers(1, 6))
    cost = rng.uniform(0, 10, (n, m))
    pairs = hungarian(cost)
    assert len(pairs) == min(n, m)
    assert assignment_cost(cost, pairs) == pytest.approx(_brute_force(cost))


def test_hungarian_makes_no_duplicate_claims():
    rng = np.random.default_rng(0)
    cost = rng.uniform(0, 10, (5, 3))
    pairs = hungarian(cost)
    assert len({r for r, _ in pairs}) == len(pairs)
    assert len({c for _, c in pairs}) == len(pairs)


def test_hungarian_treats_infinity_as_forbidden():
    cost = np.array([[1.0, np.inf], [np.inf, 2.0]])
    assert hungarian(cost) == [(0, 0), (1, 1)]
    assert hungarian(np.array([[np.inf, np.inf]])) == []


def test_greedy_and_optimal_differ_on_the_classic_counterexample():
    """Greedy grabs the single cheapest cell and pays for it afterwards."""
    from ppstack.perception.tracking import _greedy

    cost = np.array([[1.0, 2.0], [1.1, 100.0]])
    assert assignment_cost(cost, hungarian(cost)) < assignment_cost(cost, _greedy(cost))


def test_empty_cost_matrix_is_handled():
    assert hungarian(np.zeros((0, 0))) == []


# -------------------------------------------------------------------- filter


def test_filter_recovers_a_constant_velocity():
    tracker = MultiObjectTracker(dt=0.1, measurement_sigma=1.0)
    for k in range(12):
        tracker.update([("red", (50.0 + 200.0 * 0.1 * k, 100.0 + 80.0 * 0.1 * k))])
    track = tracker.confirmed_tracks()[0]
    assert track.velocity[0] == pytest.approx(200.0, abs=15.0)
    assert track.velocity[1] == pytest.approx(80.0, abs=15.0)


def test_covariance_stays_symmetric_and_positive_definite():
    """The Joseph form update exists to guarantee this under round-off."""
    tracker = MultiObjectTracker(dt=0.05)
    for k in range(60):
        tracker.update([("red", (10.0 * k, 5.0 * k))])
    P = tracker.all_tracks()[0].P
    assert np.allclose(P, P.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(P) > 0)


def test_a_track_is_not_confirmed_by_a_single_detection():
    tracker = MultiObjectTracker(min_hits=3)
    assert tracker.update([("red", (100.0, 100.0))]) == []
    tracker.update([("red", (101.0, 100.0))])
    assert tracker.confirmed_tracks() == []
    tracker.update([("red", (102.0, 100.0))])
    assert len(tracker.confirmed_tracks()) == 1


def test_a_track_is_deleted_after_enough_misses():
    tracker = MultiObjectTracker(min_hits=1, max_misses=2)
    tracker.update([("red", (100.0, 100.0))])
    for _ in range(4):
        tracker.update([])
    assert tracker.all_tracks() == []


def test_a_coasting_track_keeps_its_identity_across_a_gap():
    tracker = MultiObjectTracker(dt=0.1, min_hits=2, max_misses=5)
    for k in range(5):
        tracker.update([("red", (100.0 + 200.0 * 0.1 * k, 100.0))])
    original = tracker.confirmed_tracks()[0].id
    for _ in range(3):
        tracker.update([])                       # the detector loses it
    assert tracker.confirmed_tracks()[0].id == original
    tracker.update([("red", (100.0 + 200.0 * 0.1 * 8, 100.0))])
    tracks = tracker.confirmed_tracks()
    assert len(tracks) == 1 and tracks[0].id == original


def test_colour_gating_refuses_to_match_across_colours():
    tracker = MultiObjectTracker(dt=0.1, min_hits=1, match_colors=True)
    tracker.update([("red", (100.0, 100.0))])
    tracker.update([("blue", (100.0, 100.0))])
    assert {t.color for t in tracker.all_tracks()} == {"red", "blue"}


def test_the_gate_widens_as_uncertainty_grows():
    """A confident track rejects a jump that a coasting one accepts."""
    confident = MultiObjectTracker(dt=0.1, min_hits=1, measurement_sigma=1.0)
    for _ in range(6):
        confident.update([("red", (100.0, 100.0))])
    tight = confident._gate_cost([("red", (140.0, 100.0))])

    coasting = MultiObjectTracker(dt=0.1, min_hits=1, measurement_sigma=1.0)
    coasting.update([("red", (100.0, 100.0))])
    for _ in range(4):
        coasting.update([])
    loose = coasting._gate_cost([("red", (140.0, 100.0))])
    assert np.isfinite(loose[0, 0])
    assert loose[0, 0] < tight[0, 0]


# ----------------------------------------------------------------- end to end


def test_tracking_a_real_rendered_sequence():
    sc = SceneSpec(
        blocks=[Block(60.0, 110.0, 0.2, 38.0, "red"), Block(340.0, 190.0, 0.2, 38.0, "red")],
        speckle_count=0,
    )
    cal = default_calibration(sc, table_to_base=TABLE_TO_BASE, bounds=BOUNDS)
    detector = BlockDetector(backend="numpy")
    tracker = MultiObjectTracker(dt=0.1)

    assignments, errors = [], []
    for frame, truth in sc.sequence({0: (280.0, 0.0), 1: (-280.0, 0.0)}, 10, dt=0.1,
                                    dropout={0: {4, 5}}):
        obs = []
        for d in detector.detect(frame):
            if d.is_obstacle:
                continue
            p = cal.pixel_to_table(d.u, d.v)
            obs.append((d.color, (p.x, p.y)))
        tracks = tracker.update(obs)
        frame_assign = {}
        for name, (tx, ty) in truth.items():
            if not tracks:
                continue
            best = min(tracks, key=lambda t: (t.position[0] - tx) ** 2 + (t.position[1] - ty) ** 2)
            frame_assign[name] = best.id
            errors.append(float(np.hypot(best.position[0] - tx, best.position[1] - ty)))
        assignments.append(frame_assign)

    assert count_id_switches(assignments) == 0
    assert np.mean(errors) < 3.0


def test_count_id_switches_counts_what_it_says():
    assert count_id_switches([{"a": 1}, {"a": 1}, {"a": 2}, {"a": 2}]) == 1
    assert count_id_switches([{"a": 1}, {"a": 2}, {"a": 1}]) == 2
    assert count_id_switches([{"a": 1}, {"b": 2}]) == 0
