"""Command line entry point.

    python -m ppstack demo --scenario detour
    python -m ppstack demo --scenario gap --gif assets/gap.gif
    python -m ppstack webcam --target red
    python -m ppstack bench
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .perception.detector import BlockDetector
from .pipeline import PickAndPlacePipeline, default_calibration
from .scenarios import BOUNDS, SCENARIOS, TABLE_TO_BASE, default_arm, home_configuration, scene


def _build(scenario: str, backend: str, seed: int, noise: float):
    sc = scene(scenario, seed=seed)
    cal = default_calibration(sc, table_to_base=TABLE_TO_BASE, bounds=BOUNDS, noise_px=noise)
    arm = default_arm()
    pipe = PickAndPlacePipeline(arm, cal, BlockDetector(backend=backend))
    return sc, cal, arm, pipe


def cmd_demo(args: argparse.Namespace) -> int:
    sc, cal, arm, pipe = _build(args.scenario, args.backend, args.seed, args.calib_noise)
    frame = sc.render()
    result = pipe.run(frame, target_color=args.target, q_start=home_configuration(arm))

    print(f"scenario: {args.scenario}   calibration RMS: {cal.rms_residual_mm:.2f} mm")
    print(result.summary())

    if result.ok:
        truth = _nearest_truth(sc, result.target)
        if truth is not None:
            err = np.hypot(truth.x - result.target.position.x, truth.y - result.target.position.y)
            print(f"\nend-to-end target error vs ground truth: {err:.2f} mm")
            tip = arm.forward(result.trajectory.q[-1])
            landed = cal.base_to_table(tip.point)
            print(
                f"arm tip lands at ({landed.x:.1f}, {landed.y:.1f}) mm, "
                f"{np.hypot(landed.x - truth.x, landed.y - truth.y):.2f} mm from the block centre"
            )

    if args.plot or args.save or args.gif:
        import matplotlib

        if not args.plot:
            matplotlib.use("Agg")
        from .viz.render import draw, save_gif

        if args.gif:
            if not result.ok:
                print("cannot animate: the pipeline did not finish", file=sys.stderr)
                return 1
            print("wrote", save_gif(args.gif, frame, result, arm, cal, sc))
        fig, _ = draw(frame, result, arm, cal, sc, title=f"scenario: {args.scenario}")
        if args.save:
            fig.savefig(args.save, dpi=130)
            print("wrote", args.save)
        if args.plot:
            import matplotlib.pyplot as plt

            plt.show()
    return 0 if result.ok else 1


def _nearest_truth(sc, target):
    """The ground-truth block closest to the estimate, for scoring."""
    if target is None:
        return None
    same = [b for b in sc.targets if b.color == target.color]
    if not same:
        return None
    return min(same, key=lambda b: (b.x - target.position.x) ** 2 + (b.y - target.position.y) ** 2)


def cmd_webcam(args: argparse.Namespace) -> int:
    """Run the perception stage on a live camera.

    The rest of the stack needs a calibration, and a real calibration needs the
    four table corners clicked in a real image, so this subcommand stops after
    detection and prints pixel-space results. It exists to show the detector
    working on real, noisy input; the millimetre numbers come from the
    synthetic scene, where there is ground truth to compare against.
    """
    try:
        import cv2
    except ImportError:
        print("webcam mode needs opencv-python: pip install opencv-python", file=sys.stderr)
        return 1

    detector = BlockDetector(backend="opencv")
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"could not open camera {args.device}", file=sys.stderr)
        return 1
    print("press q to quit")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for d in detector.detect(rgb):
                u0, v0, u1, v1 = d.bbox
                cv2.rectangle(bgr, (u0, v0), (u1, v1), (0, 255, 0), 2)
                cv2.putText(
                    bgr,
                    f"{d.color} {np.degrees(d.angle_rad):+.0f}deg",
                    (u0, max(14, v0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
            cv2.imshow("ppstack perception", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .benchmark import run_all

    run_all(trials=args.trials)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ppstack", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run the full stack on a synthetic scene")
    d.add_argument("--scenario", default="detour", choices=sorted(SCENARIOS))
    d.add_argument("--target", default="red")
    d.add_argument("--backend", default="auto", choices=["auto", "numpy", "opencv"])
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--calib-noise", type=float, default=0.6,
                   help="pixel noise on the calibration fiducials")
    d.add_argument("--plot", action="store_true", help="show the figure")
    d.add_argument("--save", help="write the figure to this path")
    d.add_argument("--gif", help="write an animated GIF to this path")
    d.set_defaults(func=cmd_demo)

    w = sub.add_parser("webcam", help="run the detector on a live camera")
    w.add_argument("--device", type=int, default=0)
    w.add_argument("--target", default="red")
    w.set_defaults(func=cmd_webcam)

    b = sub.add_parser("bench", help="accuracy and planner benchmarks")
    b.add_argument("--trials", type=int, default=30)
    b.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
