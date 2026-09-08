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
from .scenarios import (
    BOUNDS, TABLE_TO_BASE, default_arm, default_bins, home_configuration, scene,
)
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


def orientation_estimator_comparison(trials: int = 30, seed: int = 0):
    """The moment-based principal axis against psi4, on identical data.

    This backs the repo's headline claim. Both estimators are run on the same
    detections from the same scenes, so the only thing that differs is the
    estimator. The moment axis is read from the detector's own region moments;
    psi4 is what the pipeline actually uses.
    """
    from .perception.detector import BlockDetector
    from .perception.orientation import square_orientation

    rng = np.random.default_rng(seed)
    detector = BlockDetector()
    moment_err, psi4_err = [], []

    for _ in range(trials):
        sc, truth = _random_scene(rng)
        cal = default_calibration(sc, table_to_base=TABLE_TO_BASE, bounds=BOUNDS)
        for d in detector.detect(sc.render()):
            if d.is_obstacle or d.color != "red":
                continue
            pos = cal.pixel_to_table(d.u, d.v)
            if np.hypot(pos.x - truth.x, pos.y - truth.y) > 5.0:
                continue

            # Estimator A: the second-moment principal axis, mapped into the
            # table frame the same way the good one is, so the comparison is
            # only about the estimator and not about the frame handling.
            step = 12.0
            du, dv = np.cos(d.angle_rad) * step, np.sin(d.angle_rad) * step
            p1 = cal.pixel_to_table(d.u - du, d.v - dv)
            p2 = cal.pixel_to_table(d.u + du, d.v + dv)
            moment_angle = float(np.arctan2(p2.y - p1.y, p2.x - p1.x))

            # Estimator B: psi4 on the footprint, in the table frame.
            footprint = cal.pixels_to_table(d.footprint_px)
            psi4_angle, _ = square_orientation(footprint, centroid=(pos.x, pos.y))

            for angle, bucket in ((moment_angle, moment_err), (psi4_angle, psi4_err)):
                delta = (angle - truth.theta + np.pi / 4) % (np.pi / 2) - np.pi / 4
                bucket.append(abs(np.degrees(delta)))
    return ErrorStats.of(moment_err), ErrorStats.of(psi4_err)


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


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def run_all(trials: int = 30, *, quick: bool = False) -> None:
    _rule("1. end-to-end accuracy against synthetic ground truth")
    stats = localisation_accuracy(trials=trials)
    print(f"  block position error (mm)  {stats['position_mm']}")
    print(f"  grasp angle error (deg)    {stats['angle_deg']}")
    print(f"  arm tip landing error (mm) {stats['tip_mm']}")
    print(f"  full pipeline runtime (ms) {stats['runtime_ms']}")
    if stats["failures"]:
        print(f"  {len(stats['failures'])} run(s) stopped early:")
        for f in stats["failures"][:3]:
            print(f"    - {f}")
    print()
    print("  grasp angle, same detections, two estimators:")
    moment, psi4 = orientation_estimator_comparison(trials=min(trials, 20))
    print(f"    second-moment principal axis  {moment}")
    print(f"    psi4 order parameter          {psi4}")
    print("    (a square's covariance is isotropic, so the moment axis is chance;")
    print("     the +-45 deg range makes uniform noise average about 22.5 deg)")

    _rule("2. sensitivity to calibration quality")
    print(f"  {'noise (px)':>11}{'cal RMS (mm)':>14}{'mean err (mm)':>15}"
          f"{'worst (mm)':>12}   outcome")
    for noise, s, rms, refused, other, n in calibration_sensitivity():
        if refused == n:
            outcome = f"all {n} refused by the residual guard"
            errs = f"{'-':>15}{'-':>12}"
        else:
            outcome = "picked"
            if refused:
                outcome += f", {refused}/{n} refused by the guard"
            if other:
                outcome += f", {other} other early stop(s)"
            errs = f"{s.mean:>15.2f}{s.worst:>12.2f}"
        print(f"  {noise:>11.1f}{rms:>14.2f}{errs}   {outcome}")

    _rule("3. A* heuristics on random obstacle fields (8-connected, no tie-break)")
    solved, acc = heuristic_comparison()
    print(f"  {solved} solvable grids")
    print(f"  {'heuristic':<12}{'cost / optimal':>16}{'expansions':>14}{'ms':>9}  admissible?")
    for name, d in acc.items():
        ratio = float(np.mean(d["cost_ratio"]))
        worst = float(np.max(d["cost_ratio"]))
        verdict = "yes" if worst <= 1.0 + 1e-9 else f"NO (worst {worst:.3f}x)"
        print(f"  {name:<12}{ratio:>16.4f}{np.mean(d['expanded']):>14.0f}"
              f"{np.mean(d['ms']):>9.1f}  {verdict}")

    _rule("4. whole-arm collision: what a tip-only pipeline never looks at")
    print("  Every IK solution below reaches the target exactly. The question is")
    print("  what the rest of the arm is doing while it does.")
    print(f"  {'target (mm)':>14}{'solutions':>11}{'in collision':>14}{'fraction':>10}")
    for tx, ty, total, hits in elbow_collision_rate():
        print(f"  {f'({tx}, {ty})':>14}{total:>11}{hits:>14}{hits / total:>9.0%}")
    print()
    print(f"  {'scenario':<10}{'collision-aware':>32}{'tip-only':>28}")
    print(f"  {'':<10}{'picks  colliding poses / total':>32}{'picks  colliding / total':>28}")
    for row in collision_aware_vs_tip_only():
        a, n = row.get("aware"), row.get("naive")
        if a is None or n is None:
            continue
        print(f"  {row['scenario']:<10}{a[0]:>13}{a[1]:>10} / {a[2]:<8}"
              f"{n[0]:>11}{n[1]:>8} / {n[2]:<8}")

    _rule("5. task-space A* versus configuration-space RRT*")
    cmp = planner_comparison(trials=3 if quick else 6)
    print(f"  {'planner':<12}{'raw cost':>10}{'shortcut':>10}{'nodes':>8}"
          f"{'coll. checks':>14}{'seconds':>9}")
    for key, label in (("rrt_star", "RRT*"), ("rrt", "RRT")):
        rows = cmp[key]
        if not rows:
            continue
        a = np.mean(np.array(rows), axis=0)
        print(f"  {label:<12}{a[0]:>10.2f}{a[1]:>10.2f}{a[2]:>8.0f}{a[3]:>14.0f}{a[4]:>9.2f}")
    print("  (cost is joint-space path length in radians, lower is better)")

    _rule("6. data association: optimal versus greedy")
    print("  Total identity switches over 30 seeded runs, 4 crossing objects.")
    print(f"  {'meas. sigma (mm)':>18}{'hungarian':>12}{'greedy':>10}{'penalty':>10}")
    for sigma, h, g in association_benchmark(seeds=10 if quick else 30):
        ratio = f"{g / h:.1f}x" if h else "-"
        print(f"  {sigma:>18}{h:>12}{g:>10}{ratio:>10}")
    print()
    print(f"  {'occlusion (frames)':>20}{'id switches':>14}{'mean err (mm)':>16}")
    for gap, switches, err in occlusion_benchmark():
        print(f"  {gap:>20}{switches:>14}{err:>16.2f}")

    _rule("7. robust calibration: mis-clicked fiducials and lens distortion")
    print(f"  {'outliers':>9}{'least squares (mm)':>20}{'RANSAC (mm)':>14}{'outliers excluded':>20}")
    for k, ls, rc, caught, total in robust_calibration_benchmark(trials=4 if quick else 8):
        excl = f"{caught}/{total}" if total else "-"
        print(f"  {k:>9}{ls:>20.2f}{rc:>14.2f}{excl:>20}")
    print()
    print(f"  {'true k1':>9}{'naive rms':>12}{'naive err':>12}"
          f"{'aware rms':>12}{'aware err':>12}{'k1 recovered':>14}")
    for k1, rms_n, err_n, _, rms_a, err_a, k1_hat in distortion_benchmark():
        print(f"  {k1:>+9.2f}{rms_n:>12.2f}{err_n:>12.2f}{rms_a:>12.2f}"
              f"{err_a:>12.2f}{k1_hat:>+14.3f}")

    _rule("8. pick ordering")
    print(f"  {'scenario':<10}{'picks':>7}{'detection order (mm)':>22}"
          f"{'sequenced (mm)':>17}{'saved':>8}")
    for name, picks, unordered, ordered, executed in sequencing_benchmark():
        saved = 1 - ordered / max(unordered, 1e-9)
        print(f"  {name:<10}{picks:>7}{unordered:>22.0f}{ordered:>17.0f}{saved:>8.0%}")


# ============================================================================
# 4. Whole-arm collision: what a tip-only pipeline is not looking at
# ============================================================================


def _collision_setup(obstacle_xy=(120.0, 150.0), radius=40.0):
    from .planning.collision import ArmCollisionChecker
    from .planning.occupancy import build_grid

    sc = SceneSpec(blocks=[Block(*obstacle_xy, 0.0, radius * 1.4, "obstacle")])
    cal = default_calibration(sc, table_to_base=TABLE_TO_BASE, bounds=BOUNDS)
    arm = default_arm()
    grid = build_grid(
        BOUNDS, [(Point2D(*obstacle_xy, "table"), radius)],
        resolution_mm=5.0, clearance_mm=22.0,
    )
    return arm, cal, ArmCollisionChecker(arm, grid, cal, link_radius_mm=10.0)


def elbow_collision_rate(targets=None):
    """How much of the reachable solution set is quietly in collision.

    For each target, the whole self-motion manifold is enumerated and every
    solution is checked for whole-arm collision. The fraction that collides is
    the probability that a tip-only pipeline, which picks a solution without
    looking, hands the controller a posture that drives a link through an
    obstacle.
    """
    from .kinematics.ik import enumerate_3link

    arm, cal, checker = _collision_setup()
    targets = targets or [(60, 250), (200, 260), (60, 80), (340, 250), (340, 80)]
    rows = []
    for tx, ty in targets:
        base = cal.table_to_base.apply(Point2D(float(tx), float(ty), "table"))
        sols = enumerate_3link(arm, base)
        if not sols:
            continue
        hits = sum(1 for s in sols if checker.collides(s.q))
        rows.append((tx, ty, len(sols), hits))
    return rows


def collision_aware_vs_tip_only(scenarios=("detour", "gap", "clutter", "sorting")):
    """Run the task layer with and without collision awareness, and count how
    many solved poses put a link inside an obstacle."""
    from .perception.detector import BlockDetector
    from .planning.collision import ArmCollisionChecker
    from .task import TaskExecutor

    detector = BlockDetector()
    arm = default_arm()
    rows = []
    for name in scenarios:
        sc = scene(name)
        cal = default_calibration(sc, table_to_base=TABLE_TO_BASE, bounds=BOUNDS)
        result = PickAndPlacePipeline(arm, cal, detector).run(
            sc.render(), q_start=home_configuration(arm)
        )
        if not result.ok and not result.objects:
            continue
        bins = default_bins()
        row = {"scenario": name}
        for aware in (False, True):
            ex = TaskExecutor(arm, cal, collision_aware=aware).run(
                result.objects, bins, q_start=home_configuration(arm)
            )
            tall = [o for o in result.objects if o.is_obstacle]
            audit = ArmCollisionChecker(
                arm,
                TaskExecutor(arm, cal)._grid(tall, 0.0, tall_only=True),
                cal,
                link_radius_mm=10.0,
            )
            bad = sum(
                int(audit.collides(q)) for seg in ex.segments for q in seg.trajectory.q
            )
            total = sum(len(seg.trajectory) for seg in ex.segments)
            row["aware" if aware else "naive"] = (ex.picks, bad, total)
        rows.append(row)
    return rows


# ============================================================================
# 5. Task-space A* versus configuration-space RRT*
# ============================================================================


def planner_comparison(trials: int = 6, seed: int = 0):
    """Grid A* on the tip against RRT* on the joints, on the same problems."""
    import time as _time

    from .kinematics.ik import enumerate_3link
    from .planning.rrt import RRTFailed, RRTStar, path_cost, shortcut_configs

    rng = np.random.default_rng(seed)
    arm, cal, checker = _collision_setup()
    out = {"rrt_star": [], "rrt": []}
    pairs = [((60, 250), (340, 80)), ((60, 80), (340, 250)), ((200, 260), (330, 60))]

    for trial in range(trials):
        start_xy, goal_xy = pairs[trial % len(pairs)]
        def clear(xy, k=8):
            base = cal.table_to_base.apply(Point2D(float(xy[0]), float(xy[1]), "table"))
            return [s.q for s in enumerate_3link(arm, base) if not checker.collides(s.q)][:k]

        qs, qg = clear(start_xy), clear(goal_xy)
        if not qs or not qg:
            continue
        for key, star in (("rrt_star", True), ("rrt", False)):
            planner = RRTStar(arm, checker, seed=int(rng.integers(1 << 20)))
            t0 = _time.perf_counter()
            try:
                r = planner.plan(
                    qs[0], np.vstack(qg), max_iterations=900,
                    time_budget_s=10.0, stop_on_first=not star,
                )
            except RRTFailed:
                continue
            smoothed = path_cost(shortcut_configs(r.path, checker))
            out[key].append(
                (r.cost, smoothed, r.nodes, r.collision_checks,
                 _time.perf_counter() - t0)
            )
    return out


# ============================================================================
# 6. Data association: optimal versus greedy
# ============================================================================


def association_benchmark(sigmas=(2, 10, 25, 40, 60), seeds: int = 30):
    """Identity switches for both association strategies, against noise.

    Run on synthetic observations rather than through the detector on purpose.
    The detector localises to well under a millimetre, so the two strategies
    never disagree on real pipeline output; forcing them apart takes injected
    noise, and saying that plainly is more useful than a rigged scene.
    """
    from .perception.tracking import MultiObjectTracker, count_id_switches

    def run(assoc, sigma, seed, n_obj=4, frames=25, dt=0.1):
        rng = np.random.default_rng(seed)
        p0 = rng.uniform(50, 350, (n_obj, 2))
        v = rng.uniform(-260, 260, (n_obj, 2))
        tracker = MultiObjectTracker(
            dt=dt, associate=assoc, measurement_sigma=max(sigma, 1e-3), match_colors=False
        )
        assignments = []
        for k in range(frames):
            truth = p0 + v * dt * k
            obs = [("red", tuple(truth[i] + rng.normal(0, sigma, 2))) for i in range(n_obj)]
            rng.shuffle(obs)
            tracks = tracker.update(obs)
            frame = {}
            for i in range(n_obj):
                if not tracks:
                    continue
                best = min(
                    tracks,
                    key=lambda t: (t.position[0] - truth[i, 0]) ** 2
                    + (t.position[1] - truth[i, 1]) ** 2,
                )
                frame[f"o{i}"] = best.id
            assignments.append(frame)
        return count_id_switches(assignments)

    return [
        (s, sum(run("hungarian", s, k) for k in range(seeds)),
         sum(run("greedy", s, k) for k in range(seeds)))
        for s in sigmas
    ]


def occlusion_benchmark(gap_frames: int = 3):
    """Does a track survive an occlusion with its identity intact?"""
    from .perception.detector import BlockDetector
    from .perception.tracking import MultiObjectTracker, count_id_switches

    sc = SceneSpec(
        blocks=[Block(60.0, 110.0, 0.2, 38.0, "red"), Block(340.0, 190.0, 0.2, 38.0, "red")],
        speckle_count=0,
    )
    cal = default_calibration(sc, table_to_base=TABLE_TO_BASE, bounds=BOUNDS)
    detector = BlockDetector()
    hidden = set(range(4, 4 + gap_frames))
    out = []
    for dropout in (None, {0: hidden}):
        tracker = MultiObjectTracker(dt=0.1)
        assignments, errors = [], []
        for frame, truth in sc.sequence({0: (280.0, 0.0), 1: (-280.0, 0.0)}, 12,
                                        dt=0.1, dropout=dropout):
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
        out.append((gap_frames if dropout else 0, count_id_switches(assignments),
                    float(np.mean(errors))))
    return out


# ============================================================================
# 7. Robust calibration: outliers and lens distortion
# ============================================================================


def robust_calibration_benchmark(n_outliers=(0, 1, 2, 3), trials: int = 8):
    """Least squares against RANSAC as mis-clicked fiducials are added."""
    from .transforms.calibration import CameraCalibration
    from .transforms.homography import DegenerateCorrespondences

    rows = []
    for k in n_outliers:
        ls_err, rc_err, caught = [], [], 0
        for t in range(trials):
            rng = np.random.default_rng(100 + t)
            sc = SceneSpec(blocks=[Block(200.0, 150.0, 0.0, 44.0, "red")])
            px, tab = sc.calibration_correspondences(noise_px=0.5, grid=4, seed=t)
            px = px.copy()
            bad = rng.choice(len(px), k, replace=False) if k else []
            for b in bad:
                px[b] += rng.normal(0, 1, 2) * 30 + np.sign(rng.normal(size=2)) * 25
            good = np.ones(len(px), bool)
            good[list(bad)] = False
            truth_px, truth_tab = px[good], tab[good]
            for robust, bucket in ((False, ls_err), (True, rc_err)):
                try:
                    cal = CameraCalibration.from_correspondences(
                        px, tab, table_to_base=TABLE_TO_BASE, bounds=BOUNDS,
                        max_residual_mm=1e9, robust=robust,
                    )
                except DegenerateCorrespondences:
                    continue
                pred = np.array([cal.pixel_to_table(u, v).as_tuple() for u, v in truth_px])
                bucket.append(float(np.sqrt(((pred - truth_tab) ** 2).sum(axis=1)).mean()))
                if robust and cal.inliers is not None:
                    caught += int((~cal.inliers[list(bad)]).sum()) if k else 0
        rows.append((k, float(np.mean(ls_err)) if ls_err else np.nan,
                     float(np.mean(rc_err)) if rc_err else np.nan,
                     caught, k * trials))
    return rows


def distortion_benchmark(k1s=(0.0, 0.04, 0.08, 0.12)):
    """What ignoring a barrel lens costs, and whether one scalar recovers it."""
    from .perception.detector import BlockDetector
    from .transforms.calibration import CameraCalibration

    arm = default_arm()
    q_home = home_configuration(arm)
    detector = BlockDetector()
    rows = []
    for k1 in k1s:
        sc = scene("clear", distortion_k1=k1)
        px, tab = sc.calibration_correspondences(noise_px=0.4, grid=4)
        truth = [b for b in sc.targets if b.color == "red"][0]
        frame = sc.render()
        entry = [k1]
        for aware in (False, True):
            cal = CameraCalibration.from_correspondences(
                px, tab, table_to_base=TABLE_TO_BASE, bounds=BOUNDS,
                max_residual_mm=1e9, estimate_distortion=aware, image_size=sc.image_size,
            )
            res = PickAndPlacePipeline(arm, cal, detector).run(frame, q_start=q_home)
            err = (
                float(np.hypot(res.target.position.x - truth.x, res.target.position.y - truth.y))
                if res.ok else np.nan
            )
            entry += [cal.rms_residual_mm, err,
                      cal.distortion.k1 if cal.distortion else 0.0]
        rows.append(tuple(entry))
    return rows


# ============================================================================
# 8. Task sequencing
# ============================================================================


def sequencing_benchmark(scenarios=("sorting", "clutter")):
    from .perception.detector import BlockDetector
    from .task import TaskExecutor

    detector = BlockDetector()
    arm = default_arm()
    rows = []
    for name in scenarios:
        sc = scene(name)
        cal = default_calibration(sc, table_to_base=TABLE_TO_BASE, bounds=BOUNDS)
        res = PickAndPlacePipeline(arm, cal, detector).run(
            sc.render(), q_start=home_configuration(arm)
        )
        ex = TaskExecutor(arm, cal).run(res.objects, default_bins(), q_start=home_configuration(arm))
        rows.append((name, ex.picks, ex.tour_mm_detection_order, ex.tour_mm, ex.executed_mm))
    return rows
