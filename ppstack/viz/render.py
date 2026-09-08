"""Three-panel figure: what the camera sees, what the planner thinks, what the
arm does. Keeping all three in one frame is the fastest way to tell which stage
is at fault when the result looks wrong."""

from __future__ import annotations

import numpy as np

from ..frames import Point2D
from ..kinematics.arm import PlanarArm
from ..perception.scene import SceneSpec
from ..pipeline import PipelineResult
from ..transforms.calibration import CameraCalibration

_COLORS = {
    "red": "#c42a26",
    "green": "#2ca044",
    "blue": "#284ebe",
    "yellow": "#deba2c",
    "obstacle": "#2e2e34",
}


def draw(
    frame: np.ndarray,
    result: PipelineResult,
    arm: PlanarArm,
    calibration: CameraCalibration,
    scene: SceneSpec | None = None,
    *,
    step: int | None = None,
    title: str = "",
):
    """Build the figure. Returns (fig, axes)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    _panel_camera(axes[0], frame, result)
    _panel_world(axes[1], result, calibration, scene)
    _panel_arm(axes[2], result, arm, calibration, step)
    header = title or "pixels -> millimetres -> joint angles"
    ok = "ok" if result.ok else f"FAILED in {result.failure_stage}"
    fig.suptitle(f"{header}    [{ok}, {result.total_ms:.0f} ms]", fontsize=12)
    fig.tight_layout()
    return fig, axes


def _panel_camera(ax, frame, result: PipelineResult) -> None:
    from matplotlib.patches import Rectangle

    ax.imshow(frame)
    ax.set_title("1. camera: HSV threshold and moments")
    for obj in result.objects:
        d = obj.source
        u0, v0, u1, v1 = d.bbox
        colour = _COLORS.get(d.color, "white")
        ax.add_patch(
            Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False, edgecolor=colour, lw=1.6)
        )
        ax.plot(*d.centroid_px, marker="+", color="white", ms=9, mew=1.6)
        L = 18
        ax.plot(
            [d.u - L * np.cos(d.angle_rad), d.u + L * np.cos(d.angle_rad)],
            [d.v - L * np.sin(d.angle_rad), d.v + L * np.sin(d.angle_rad)],
            color="white", lw=1.2, alpha=0.9,
        )
    if result.target is not None:
        ax.plot(*result.target.source.centroid_px, marker="o", mfc="none",
                mec="white", ms=22, mew=2.0)
    ax.set_xticks([]); ax.set_yticks([])


def _panel_world(ax, result: PipelineResult, calibration, scene) -> None:
    b = calibration.bounds
    ax.set_title("2. table frame: occupancy, A*, shortcut")
    if result.grid is not None:
        ax.imshow(
            result.grid.blocked,
            origin="lower",
            extent=(b.x_min, b.x_max, b.y_min, b.y_max),
            cmap="Greys",
            alpha=0.35,
            interpolation="nearest",
        )
    if scene is not None:
        for blk in scene.blocks:
            c = blk.corners()
            ax.fill(*np.vstack([c, c[:1]]).T, color=_COLORS.get(blk.color, "grey"),
                    alpha=0.30, lw=0)
    for obj in result.objects:
        ax.plot(obj.position.x, obj.position.y, "o", color=_COLORS.get(obj.color, "grey"), ms=7)
    if result.plan is not None:
        raw = result.plan.world_path()
        ax.plot([p.x for p in raw], [p.y for p in raw], "-", color="#9aa0a6", lw=1.0,
                label=f"A* ({result.plan.expanded} expansions)")
    if result.waypoints:
        ax.plot([p.x for p in result.waypoints], [p.y for p in result.waypoints],
                "-", color="#0b6fd1", lw=2.2, label="shortcut path")
    if result.target is not None:
        ax.plot(result.target.position.x, result.target.position.y, "*",
                color="#111", ms=17, label="target")
    base = calibration.base_to_table(Point2D(0.0, 0.0, "base"))
    ax.plot(base.x, base.y, "s", color="#111", ms=8)
    ax.annotate("arm base", (base.x, base.y), textcoords="offset points",
                xytext=(8, -12), fontsize=8)
    ax.set_xlim(b.x_min - 20, b.x_max + 20)
    ax.set_ylim(b.y_min - 20, b.y_max + 20)
    ax.set_aspect("equal")
    ax.set_xlabel("table x (mm)"); ax.set_ylabel("table y (mm)")
    if result.plan is not None:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)


def _panel_arm(ax, result: PipelineResult, arm: PlanarArm, calibration, step) -> None:
    ax.set_title("3. base frame: joint solution")
    R = arm.max_reach
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R * np.cos(theta), R * np.sin(theta), ":", color="#bbb", lw=1)
    if arm.min_reach > 1e-6:
        ax.plot(arm.min_reach * np.cos(theta), arm.min_reach * np.sin(theta), ":",
                color="#bbb", lw=1)

    b = calibration.bounds
    table_outline = [
        Point2D(b.x_min, b.y_min, "table"), Point2D(b.x_max, b.y_min, "table"),
        Point2D(b.x_max, b.y_max, "table"), Point2D(b.x_min, b.y_max, "table"),
    ]
    pts = [calibration.table_to_base.apply(p) for p in table_outline]
    ax.plot([p.x for p in pts] + [pts[0].x], [p.y for p in pts] + [pts[0].y],
            "-", color="#ccc", lw=1)

    traj = result.trajectory
    if traj is not None:
        ax.plot(traj.tip[:, 0], traj.tip[:, 1], "-", color="#0b6fd1", lw=1.6, alpha=0.85)
        k = len(traj) - 1 if step is None else int(np.clip(step, 0, len(traj) - 1))
        for j in range(0, len(traj), max(1, len(traj) // 6)):
            gp = arm.joint_positions(traj.q[j])
            ax.plot(gp[:, 0], gp[:, 1], "-", color="#c9c9c9", lw=1.2, zorder=1)
        pts_arm = arm.joint_positions(traj.q[k])
        ax.plot(pts_arm[:, 0], pts_arm[:, 1], "-o", color="#111", lw=3.0, ms=6, zorder=3)
        ax.plot(pts_arm[-1, 0], pts_arm[-1, 1], "o", color="#c42a26", ms=9, zorder=4)
    ax.plot(0, 0, "s", color="#111", ms=9)
    ax.set_xlim(-R * 1.1, R * 1.1); ax.set_ylim(-R * 1.1, R * 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("base x (mm)"); ax.set_ylabel("base y (mm)")


def save_gif(path, frame, result, arm, calibration, scene=None, fps: int = 20) -> str:
    """Animate the arm along the solved trajectory and write a GIF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if result.trajectory is None:
        raise ValueError("nothing to animate: the pipeline did not reach the arm")
    fig, axes = draw(frame, result, arm, calibration, scene, step=0)

    def update(i):
        axes[2].clear()
        _panel_arm(axes[2], result, arm, calibration, i)
        return axes[2].get_children()

    anim = FuncAnimation(fig, update, frames=len(result.trajectory), interval=1000 / fps)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return str(path)
