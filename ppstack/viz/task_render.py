"""Animating a whole pick-and-place cycle.

One figure, three panels, one shared clock: what the camera sees, what the
planner is doing about it right now, and where the arm actually is. The point
of animating the whole task rather than a single reach is that the interesting
failures are all temporal. A block that stops being an obstacle once it is
picked up, a gripper that has to fit through a narrower gap while carrying
something, an elbow that has to swing the long way round: none of it is visible
in a still frame of the final pose.
"""

from __future__ import annotations

import numpy as np

from ..frames import Point2D
from ..kinematics.arm import PlanarArm
from ..perception.scene import SceneSpec
from ..task import TaskExecution
from ..transforms.calibration import CameraCalibration

_COLORS = {
    "red": "#c42a26",
    "green": "#2ca044",
    "blue": "#284ebe",
    "yellow": "#deba2c",
    "obstacle": "#2e2e34",
}


def _timeline(execution: TaskExecution, hold: int = 3, stride: int = 1):
    """Flatten every segment into one list of (segment index, pose index).

    A few repeated frames are inserted at each hand-over so the grasp and the
    release read as events rather than as the animation briefly stalling.
    """
    frames = []
    for si, seg in enumerate(execution.segments):
        n = len(seg.trajectory)
        for k in range(0, n, stride):
            frames.append((si, k))
        frames.extend([(si, n - 1)] * hold)
    return frames


def _draw_camera(ax, frame, result, execution, seg_index):
    from matplotlib.patches import Rectangle

    ax.imshow(frame)
    ax.set_title("1. camera (one frame, t=0)", fontsize=11)
    current = execution.segments[seg_index].obj if execution.segments else None
    for obj in result.objects:
        d = obj.source
        u0, v0, u1, v1 = d.bbox
        colour = _COLORS.get(d.color, "white")
        ax.add_patch(
            Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False, edgecolor=colour, lw=1.4)
        )
        ax.plot(*d.centroid_px, marker="+", color="white", ms=7, mew=1.3)
    if current is not None:
        ax.plot(*current.source.centroid_px, marker="o", mfc="none", mec="white",
                ms=20, mew=2.0)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_table(ax, execution, calibration, bins, seg_index, pose_index, placed):
    from matplotlib.patches import Circle, Rectangle

    b = calibration.bounds
    seg = execution.segments[seg_index]
    ax.set_title(
        f"2. table frame  |  {seg.kind}  {seg.obj.color} -> {seg.bin.name}", fontsize=11
    )
    ax.imshow(
        seg.grid.blocked, origin="lower",
        extent=(b.x_min, b.x_max, b.y_min, b.y_max),
        cmap="Greys", alpha=0.30, interpolation="nearest",
    )
    for dest in bins:
        ax.add_patch(
            Rectangle((dest.x - 26, dest.y - 26), 52, 52, fill=False,
                      edgecolor=_COLORS.get(dest.accepts[0] if dest.accepts else "obstacle",
                                            "#555"),
                      lw=1.6, ls="--", alpha=0.8)
        )
        ax.annotate(dest.name, (dest.x, dest.y - 34), ha="center", fontsize=7, alpha=0.8)

    # Blocks still on the table, plus the ones already delivered, drawn at their bins.
    for obj in execution.segments[seg_index].grid_objects:
        ax.add_patch(Circle((obj.position.x, obj.position.y), max(10.0, obj.size_mm / 2),
                            color=_COLORS.get(obj.color, "grey"), alpha=0.45))
    for obj, dest in placed:
        ax.add_patch(Circle((dest.x, dest.y), max(9.0, obj.size_mm / 2.5),
                            color=_COLORS.get(obj.color, "grey"), alpha=0.85))

    xs = [p.x for p in seg.waypoints]
    ys = [p.y for p in seg.waypoints]
    ax.plot(xs, ys, "-", color="#9aa0a6", lw=1.2, alpha=0.8)
    ax.plot(xs[: pose_index + 1], ys[: pose_index + 1], "-",
            color="#0b6fd1" if seg.holding else "#111", lw=2.4)

    tip = seg.waypoints[min(pose_index, len(seg.waypoints) - 1)]
    ax.plot(tip.x, tip.y, "o", color="#111", ms=7)
    if seg.holding:
        ax.add_patch(Circle((tip.x, tip.y), max(10.0, seg.obj.size_mm / 2),
                            color=_COLORS.get(seg.obj.color, "grey"), alpha=0.9))

    base = calibration.base_to_table(Point2D(0.0, 0.0, "base"))
    ax.plot(base.x, base.y, "s", color="#111", ms=7)
    ax.set_xlim(b.x_min - 25, b.x_max + 25)
    ax.set_ylim(b.y_min - 25, b.y_max + 25)
    ax.set_aspect("equal")
    ax.set_xlabel("table x (mm)")
    ax.set_ylabel("table y (mm)")


def _draw_arm(ax, execution, arm, calibration, seg_index, pose_index):
    seg = execution.segments[seg_index]
    ax.set_title("3. base frame: joint solution", fontsize=11)
    R = arm.max_reach
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R * np.cos(th), R * np.sin(th), ":", color="#ccc", lw=1)

    outline = [
        Point2D(calibration.bounds.x_min, calibration.bounds.y_min, "table"),
        Point2D(calibration.bounds.x_max, calibration.bounds.y_min, "table"),
        Point2D(calibration.bounds.x_max, calibration.bounds.y_max, "table"),
        Point2D(calibration.bounds.x_min, calibration.bounds.y_max, "table"),
    ]
    pts = [calibration.table_to_base.apply(p) for p in outline]
    ax.plot([p.x for p in pts] + [pts[0].x], [p.y for p in pts] + [pts[0].y],
            "-", color="#ddd", lw=1)

    traj = seg.trajectory
    k = int(np.clip(pose_index, 0, len(traj) - 1))
    ax.plot(traj.tip[: k + 1, 0], traj.tip[: k + 1, 1], "-",
            color="#0b6fd1" if seg.holding else "#111", lw=1.5, alpha=0.8)
    joints = arm.joint_positions(traj.q[k])
    ax.plot(joints[:, 0], joints[:, 1], "-o", color="#111", lw=3.2, ms=5.5, zorder=3)
    ax.plot(joints[-1, 0], joints[-1, 1], "o",
            color=_COLORS.get(seg.obj.color, "#c42a26") if seg.holding else "#666",
            ms=10, zorder=4)
    ax.plot(0, 0, "s", color="#111", ms=8)
    ax.set_xlim(-R * 1.08, R * 1.08)
    ax.set_ylim(-R * 1.08, R * 1.08)
    ax.set_aspect("equal")
    ax.set_xlabel("base x (mm)")
    ax.set_ylabel("base y (mm)")


def save_task_gif(
    path,
    frame: np.ndarray,
    result,
    execution: TaskExecution,
    arm: PlanarArm,
    calibration: CameraCalibration,
    bins,
    scene: SceneSpec | None = None,
    *,
    fps: int = 14,
    title: str = "",
    stride: int = 3,
    dpi: int = 62,
    figsize: tuple[float, float] = (14.0, 4.9),
    colors: int = 64,
) -> str:
    """Animate a whole multi-object pick-and-place cycle to a GIF.

    `stride`, `dpi` and `colors` exist because a README hero image has a real
    budget: GitHub will not display an image much over 10 MB, and a full-rate,
    full-resolution, full-palette render of this animation lands around 13 MB.
    Halving the frame rate, dropping to 70 dpi and quantising to a 96-colour
    palette costs nothing visible on a line-art figure and brings it under 4 MB.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if not execution.segments:
        raise ValueError("nothing to animate: the task produced no segments")

    timeline = _timeline(execution, stride=stride)
    fig, axes = plt.subplots(
        1, 3, figsize=figsize, layout="constrained",
        gridspec_kw={"width_ratios": [1.15, 1.0, 0.95]},
    )
    header = title or "pick-and-place-stack: pixels -> millimetres -> joint angles"

    def render_frame(i):
        seg_index, pose_index = timeline[i]
        placed = [
            (s.obj, s.bin)
            for s in execution.segments[:seg_index]
            if s.kind == "carry"
        ]
        for ax in axes:
            ax.clear()
        _draw_camera(axes[0], frame, result, execution, seg_index)
        _draw_table(axes[1], execution, calibration, bins, seg_index, pose_index, placed)
        _draw_arm(axes[2], execution, arm, calibration, seg_index, pose_index)
        # Count only carries that have already finished, so the counter does
        # not claim a block is placed while the arm is still carrying it.
        done = sum(1 for s in execution.segments[:seg_index] if s.kind == "carry")
        fig.suptitle(
            f"{header}    |    {done}/{execution.picks} placed", fontsize=12
        )
        return []

    def update(i):
        return render_frame(i)

    fig.render_frame = render_frame          # exposed for still-frame inspection
    anim = FuncAnimation(fig, update, frames=len(timeline), interval=1000 / fps)
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    _quantise(path, colors)
    return str(path)


def _quantise(path, colors: int) -> None:
    """Re-encode the GIF with a smaller palette, in place.

    Matplotlib writes a full-colour GIF. These figures are line art on white
    and use maybe thirty distinct colours, so most of that palette is paid for
    and never used.
    """
    try:
        from PIL import Image, ImageSequence
    except ImportError:
        return
    with Image.open(path) as im:
        frames = [
            f.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT)
            for f in ImageSequence.Iterator(im)
        ]
        duration = im.info.get("duration", 50)
    frames[0].save(
        path, save_all=True, append_images=frames[1:], loop=0,
        duration=duration, optimize=True, disposal=2,
    )
