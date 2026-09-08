"""Numbers, not vibes.

Three experiments, all runnable with `python -m ppstack bench`:

1. End-to-end localisation error. The synthetic scene knows exactly where each
   block is, so the full chain (threshold, moments, homography, rigid
   transform, IK) can be scored in millimetres rather than eyeballed.
2. Sensitivity to calibration quality. Error is swept against the noise on the
   calibration fiducials, which answers the question that actually matters on
   real hardware: how carefully do the corners have to be clicked?
3. Planner heuristics. Path cost and node expansions for each heuristic against
   a Dijkstra reference, on random obstacle fields. This is where the claim
   that Manhattan is inadmissible on an 8-connected grid gets checked rather
   than repeated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .frames import Point2D
from .perception.detector import BlockDetector
from .perception.scene import Block, SceneSpec
from .pipeline import PickAndPlacePipeline, default_calibration
from .planning.astar import HEURISTICS, astar
from .planning.occupancy import OccupancyGrid
from .scenarios import BOUNDS, TABLE_TO_BASE, default_arm, home_configuration
from .transforms.calibration import WorkspaceBounds


@dataclass
class ErrorStats:
    n: int
    mean: float
    median: float
    p95: float
    worst: float

    @classmethod
    def of(cls, values) -> "ErrorStats":
        v = np.asarray(values, dtype=float)
        if len(v) == 0:
            return cls(0, np.nan, np.nan, np.nan, np.nan)
        return cls(len(v), float(v.mean()), float(np.median(v)),
                   float(np.percentile(v, 95)), float(v.max()))

    def __str__(self) -> str:
        return (f"n={self.n:<4d} mean {self.mean:6.2f}  median {self.median:6.2f}  "
                f"p95 {self.p95:6.2f}  worst {self.worst:6.2f}")


def _random_scene(rng: np.random.Generator) -> tuple[SceneSpec, Block]:
    """A scene with one red target and a few distractors, all well separated."""
    placed: list[Block] = []

    def free_spot(size: float) -> tuple[float, float]:
        for _ in range(200):
            x = float(rng.uniform(60, 340))
            y = float(rng.uniform(60, 240))
            if all(np.hypot(x - b.x, y - b.y) > (size + b.size) * 0.95 for b in placed):
                return x, y
        raise RuntimeError("could not place a block")

    x, y = free_spot(48.0)
    target = Block(x, y, float(rng.uniform(-np.pi, np.pi)), 48.0, "red")
    placed.append(target)
    for colour in ("blue", "green", "yellow", "obstacle"):
        size = 52.0 if colour == "obstacle" else 42.0
        x, y = free_spot(size)
        placed.append(Block(x, y, float(rng.uniform(-np.pi, np.pi)), size, colour))
    return SceneSpec(blocks=placed, seed=int(rng.integers(1 << 30))), target


def localisation_accuracy(trials: int = 30, calib_noise_px: float = 0.6, seed: int = 0):
    """Score the perception-to-millimetres chain against ground truth."""
    rng = np.random.default_rng(seed)
    arm = default_arm()
    q_home = home_configuration(arm)
    detector = BlockDetector()

    position_err, angle_err, tip_err, timings, cal_rms = [], [], [], [], []
    failures: list[str] = []

    for _ in range(trials):
        sc, truth = _random_scene(rng)
        cal = default_calibration(sc, table_to_base=TABLE_TO_BASE, bounds=BOUNDS,
                                  noise_px=calib_noise_px)
        cal_rms.append(cal.rms_residual_mm)
        pipe = PickAndPlacePipeline(arm, cal, detector)
        t0 = time.perf_counter()
        res = pipe.run(sc.render(), target_color="red", q_start=q_home)
        timings.append((time.perf_counter() - t0) * 1e3)
        if not res.ok:
            failures.append(f"{res.failure_stage}: {res.failure}")
            continue

        est = res.target.position
        position_err.append(float(np.hypot(est.x - truth.x, est.y - truth.y)))

        # A square block's grasp is symmetric every 90 degrees, so the angle
        # residual is taken modulo a quarter turn. Comparing raw angles would
        # report a 90 degree "error" for a grasp that is exactly right.
        d = (res.target.grasp_angle_rad - truth.theta + np.pi / 4) % (np.pi / 2) - np.pi / 4
        angle_err.append(abs(np.degrees(d)))

        tip = cal.base_to_table(arm.forward(res.trajectory.q[-1]).point)
        tip_err.append(float(np.hypot(tip.x - truth.x, tip.y - truth.y)))

    return {
        "position_mm": ErrorStats.of(position_err),
        "angle_deg": ErrorStats.of(angle_err),
        "tip_mm": ErrorStats.of(tip_err),
        "runtime_ms": ErrorStats.of(timings),
        "calibration_rms_mm": ErrorStats.of(cal_rms),
        "failures": failures,
    }


def calibration_sensitivity(noises=(0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0), trials: int = 12):
    """How far the pick lands off when the fiducials are clicked sloppily.

    Two things happen as the noise grows, and they matter separately. The
    landing error grows, which is the accuracy story. And past some point the
    calibration's own residual crosses the acceptance threshold and the
    pipeline refuses to run at all, which is the safety story: it stops picking
    rather than picking somewhere wrong. Both columns are reported.
    """
    rows = []
    for noise in noises:
        stats = localisation_accuracy(trials=trials, calib_noise_px=noise, seed=7)
        refused = sum("recalibrate" in f for f in stats["failures"])
        other = len(stats["failures"]) - refused
        rows.append((noise, stats["position_mm"], stats["calibration_rms_mm"].mean,
                     refused, other, trials))
    return rows


def _random_grid(rng: np.random.Generator, size: int = 70, blobs: int = 14) -> OccupancyGrid:
    blocked = np.zeros((size, size), dtype=bool)
    for _ in range(blobs):
        r, c = rng.integers(5, size - 5, 2)
        h, w = rng.integers(3, 14, 2)
        blocked[max(0, r - h): r + h, max(0, c - w): c + w] = True
    blocked[:4, :4] = False
    blocked[-4:, -4:] = False
    return OccupancyGrid(blocked, WorkspaceBounds(0, size * 5.0, 0, size * 5.0), 5.0)


def heuristic_comparison(trials: int = 40, seed: int = 3):
    """Cost and expansions per heuristic, against the Dijkstra optimum."""
    rng = np.random.default_rng(seed)
    names = ["zero", "euclidean", "octile", "manhattan"]
    acc = {n: {"cost_ratio": [], "expanded": [], "ms": []} for n in names}
    solved = 0

    for _ in range(trials):
        grid = _random_grid(rng)
        start, goal = (1, 1), (grid.shape[0] - 2, grid.shape[1] - 2)
        try:
            optimal = astar(grid, start, goal, heuristic="zero", tie_breaker=0.0).cost
        except Exception:
            continue
        solved += 1
        for name in names:
            t0 = time.perf_counter()
            r = astar(grid, start, goal, heuristic=name, tie_breaker=0.0)
            acc[name]["ms"].append((time.perf_counter() - t0) * 1e3)
            acc[name]["cost_ratio"].append(r.cost / optimal)
            acc[name]["expanded"].append(r.expanded)
    return solved, acc


def run_all(trials: int = 30) -> None:
    print("=" * 74)
    print("1. end-to-end accuracy against synthetic ground truth")
    print("=" * 74)
    stats = localisation_accuracy(trials=trials)
    print(f"  block position error (mm)  {stats['position_mm']}")
    print(f"  grasp angle error (deg)    {stats['angle_deg']}")
    print(f"  arm tip landing error (mm) {stats['tip_mm']}")
    print(f"  full pipeline runtime (ms) {stats['runtime_ms']}")
    if stats["failures"]:
        print(f"  {len(stats['failures'])} run(s) stopped early:")
        for f in stats["failures"][:5]:
            print(f"    - {f}")

    print()
    print("=" * 74)
    print("2. sensitivity to calibration quality")
    print("=" * 74)
    print(f"  {'noise (px)':>11}{'cal RMS (mm)':>14}{'mean err (mm)':>15}"
          f"{'worst (mm)':>12}   outcome")
    for noise, s, rms, refused, other, trials in calibration_sensitivity():
        if refused == trials:
            outcome = f"all {trials} refused by the residual guard"
            errs = f"{'-':>15}{'-':>12}"
        else:
            outcome = "picked"
            if refused:
                outcome += f", {refused}/{trials} refused by the guard"
            if other:
                outcome += f", {other} other early stop(s)"
            errs = f"{s.mean:>15.2f}{s.worst:>12.2f}"
        print(f"  {noise:>11.1f}{rms:>14.2f}{errs}   {outcome}")

    print()
    print("=" * 74)
    print("3. A* heuristics on random obstacle fields (8-connected, no tie-break)")
    print("=" * 74)
    solved, acc = heuristic_comparison()
    print(f"  {solved} solvable grids")
    print(f"  {'heuristic':<12}{'cost / optimal':>16}{'expansions':>14}{'ms':>9}  admissible?")
    for name, d in acc.items():
        ratio = float(np.mean(d["cost_ratio"]))
        worst = float(np.max(d["cost_ratio"]))
        verdict = "yes" if worst <= 1.0 + 1e-9 else f"NO (worst {worst:.3f}x)"
        print(f"  {name:<12}{ratio:>16.4f}{np.mean(d['expanded']):>14.0f}"
              f"{np.mean(d['ms']):>9.1f}  {verdict}")
