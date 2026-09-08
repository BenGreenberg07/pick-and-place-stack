"""A visual pick-and-place stack: camera pixels to joint angles.

The public surface is deliberately small. Everything interesting is one level
down, in `perception`, `transforms`, `planning` and `kinematics`.
"""

from .frames import FrameMismatch, Point2D, Pose2D, RigidTransform2D
from .kinematics.arm import JointLimitViolation, PlanarArm
from .kinematics.ik import IKUnreachable, solve, solve_2link, solve_dls
from .perception.detector import BlockDetector, Detection, DetectorConfig
from .perception.scene import Block, SceneSpec
from .pipeline import PickAndPlacePipeline, PipelineResult, default_calibration
from .planning.astar import NoPathFound, astar
from .planning.occupancy import OccupancyGrid, build_grid
from .transforms.calibration import CalibrationError, CameraCalibration, WorkspaceBounds

__version__ = "0.1.0"

__all__ = [
    "Block", "BlockDetector", "CalibrationError", "CameraCalibration", "Detection",
    "DetectorConfig", "FrameMismatch", "IKUnreachable", "JointLimitViolation",
    "NoPathFound", "OccupancyGrid", "PickAndPlacePipeline", "PipelineResult",
    "PlanarArm", "Point2D", "Pose2D", "RigidTransform2D", "SceneSpec",
    "WorkspaceBounds", "astar", "build_grid", "default_calibration", "solve",
    "solve_2link", "solve_dls", "__version__",
]
