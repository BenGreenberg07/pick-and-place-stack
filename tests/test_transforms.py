import numpy as np
import pytest

from ppstack.frames import Point2D, RigidTransform2D
from ppstack.transforms.calibration import CalibrationError, CameraCalibration, WorkspaceBounds
from ppstack.transforms.homography import (
    DegenerateCorrespondences, apply_homography, fit_homography, reprojection_error,
)
from ppstack.scenarios import BOUNDS, TABLE_TO_BASE
from ppstack.perception.scene import Block, SceneSpec


def _random_homography(rng):
    H = np.eye(3) + rng.normal(0, 0.08, (3, 3))
    H[2, :2] *= 1e-3
    return H / H[2, 2]


@pytest.mark.parametrize("seed", range(20))
def test_four_points_recover_the_homography_exactly(seed):
    rng = np.random.default_rng(seed)
    H = _random_homography(rng)
    src = rng.uniform(0, 640, (10, 2))
    dst = apply_homography(H, src)
    H_fit = fit_homography(src[:4], dst[:4])
    assert reprojection_error(H_fit, src, dst).max() < 1e-6


def test_overdetermined_fit_averages_down_the_noise():
    rng = np.random.default_rng(0)
    H = _random_homography(rng)
    src = rng.uniform(50, 600, (30, 2))
    dst = apply_homography(H, src) + rng.normal(0, 0.5, (30, 2))
    assert np.median(reprojection_error(fit_homography(src, dst), src, dst)) < 1.0


def test_collinear_points_are_rejected():
    line = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    with pytest.raises(DegenerateCorrespondences):
        fit_homography(line, np.array([[0.0, 0.0], [1, 0], [2, 0], [3, 0.0]]))


def test_too_few_correspondences_rejected():
    with pytest.raises(DegenerateCorrespondences):
        fit_homography(np.zeros((3, 2)), np.zeros((3, 2)))


def _calibration(noise_px=0.0):
    sc = SceneSpec(blocks=[Block(200.0, 150.0, 0.0, 40.0, "red")])
    px, tab = sc.calibration_correspondences(noise_px=noise_px)
    return sc, CameraCalibration.from_correspondences(
        px, tab, table_to_base=TABLE_TO_BASE, bounds=BOUNDS
    )


def test_pixel_to_table_recovers_the_scene_geometry():
    sc, cal = _calibration()
    for x, y in [(0, 0), (400, 0), (400, 300), (0, 300), (137.0, 211.0)]:
        u, v = cal.table_to_pixel(Point2D(x, y, "table"))
        back = cal.pixel_to_table(u, v)
        assert (back.x, back.y) == pytest.approx((x, y), abs=1e-6)


def test_full_chain_lands_in_the_base_frame():
    _, cal = _calibration()
    u, v = cal.table_to_pixel(Point2D(200.0, 150.0, "table"))
    base = cal.pixel_to_base(u, v)
    assert base.frame == "base"
    # Table (200, 150) is 210 mm in front of a base sitting at table (200, -60)
    # and rotated a quarter turn, so it lands on the base frame's +x axis.
    assert (base.x, base.y) == pytest.approx((210.0, 0.0), abs=1e-6)
    assert cal.base_to_table(base).as_tuple() == pytest.approx((200.0, 150.0), abs=1e-6)


def test_pixel_scale_varies_across_a_perspective_frame():
    """mm-per-pixel is not constant under perspective, which is why an area
    threshold in pixels cannot be converted with a single scale factor."""
    sc, cal = _calibration()
    near = cal.pixel_scale_mm(*cal.table_to_pixel(Point2D(200.0, 20.0, "table")))
    far = cal.pixel_scale_mm(*cal.table_to_pixel(Point2D(200.0, 280.0, "table")))
    assert far > near * 1.15


def test_calibration_validate_rejects_a_bad_fit():
    _, cal = _calibration(noise_px=40.0)
    with pytest.raises(CalibrationError):
        cal.validate()


def test_clean_calibration_passes_validation():
    _, cal = _calibration(noise_px=0.5)
    cal.validate()
    assert cal.rms_residual_mm < 3.0


def test_calibration_survives_a_json_roundtrip(tmp_path):
    _, cal = _calibration(noise_px=0.5)
    path = tmp_path / "cal.json"
    cal.to_json(path)
    other = CameraCalibration.from_json(path)
    assert np.allclose(other.H_pixel_to_table, cal.H_pixel_to_table)
    p = Point2D(150.0, 90.0, "table")
    assert other.table_to_base.apply(p).as_tuple() == pytest.approx(
        cal.table_to_base.apply(p).as_tuple()
    )


def test_workspace_bounds_reject_off_table_points():
    b = WorkspaceBounds(10, 390, 10, 290)
    assert b.contains(Point2D(200, 150, "table"))
    assert not b.contains(Point2D(-5, 150, "table"))


def test_minimal_four_point_calibration_warns():
    """Four points fit exactly, so the residual is meaningless. The API should
    say so rather than let a convincing-looking zero residual through."""
    sc = SceneSpec(blocks=[Block(200.0, 150.0, 0.0, 40.0, "red")])
    px, tab = sc.calibration_correspondences(grid=2)
    with pytest.warns(UserWarning, match="4 correspondences"):
        cal = CameraCalibration.from_correspondences(
            px, tab, table_to_base=TABLE_TO_BASE, bounds=BOUNDS
        )
    assert cal.rms_residual_mm == pytest.approx(0.0, abs=1e-6)
