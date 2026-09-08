"""A visual pick-and-place stack: camera pixels to joint angles.

The public surface is deliberately small. Everything interesting is one level
down, in `perception`, `transforms`, `planning` and `kinematics`.
"""

from .frames import FrameMismatch, Point2D, Pose2D, RigidTransform2D
from .kinematics.arm import JointLimitViolation, PlanarArm
from .kinematics.ik import (
    IKUnreachable, enumerate_3link, enumerate_solutions, solve, solve_2link,
    solve_collision_free, solve_dls,
)
from .perception.detector import BlockDetector, Detection, DetectorConfig
from .perception.scene import Block, SceneSpec
from .pipeline import PickAndPlacePipeline, PipelineResult, default_calibration
from .perception.assignment import hungarian
from .perception.tracking import MultiObjectTracker, Track
from .planning.astar import NoPathFound, astar
from .planning.collision import ArmCollisionChecker
from .planning.occupancy import OccupancyGrid, build_grid
from .planning.rrt import RRTFailed, RRTStar
from .task import Bin, TaskExecutor, TaskExecution, sequence_jobs
from .transforms.calibration import CalibrationError, CameraCalibration, WorkspaceBounds
from .transforms.distortion import RadialDistortion, estimate_k1
from .transforms.homography import fit_homography, fit_homography_ransac

__version__ = "0.1.0"

__all__ = [
    "ArmCollisionChecker", "Bin", "Block", "BlockDetector", "CalibrationError",
    "CameraCalibration", "Detection", "DetectorConfig", "FrameMismatch",
    "IKUnreachable", "JointLimitViolation", "MultiObjectTracker", "NoPathFound",
    "OccupancyGrid", "PickAndPlacePipeline", "PipelineResult", "PlanarArm",
    "Point2D", "Pose2D", "RRTFailed", "RRTStar", "RadialDistortion",
    "RigidTransform2D", "SceneSpec", "TaskExecution", "TaskExecutor", "Track",
    "WorkspaceBounds", "astar", "build_grid", "default_calibration",
    "enumerate_3link", "enumerate_solutions", "estimate_k1", "fit_homography",
    "fit_homography_ransac", "hungarian", "sequence_jobs", "solve", "solve_2link",
    "solve_collision_free", "solve_dls", "__version__",
]
