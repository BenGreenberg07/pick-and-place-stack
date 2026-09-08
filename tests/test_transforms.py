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


# ------------------------------------------------------- robust and distorted


def test_ransac_excludes_a_mis_clicked_fiducial():
    rng = np.random.default_rng(0)
    sc = SceneSpec(blocks=[Block(200.0, 150.0, 0.0, 44.0, "red")])
    px, tab = sc.calibration_correspondences(noise_px=0.4, grid=4)
    px = px.copy()
    px[5] += [40.0, -35.0]                    # one badly placed click

    from ppstack.transforms.homography import fit_homography_ransac

    H_ls = fit_homography(px, tab)
    H_rc, inliers = fit_homography_ransac(px, tab, threshold=2.0)
    good = np.ones(len(px), bool)
    good[5] = False
    assert not inliers[5]                     # the outlier is named, not smeared
    assert inliers[good].all()
    ls = np.median(reprojection_error(H_ls, px[good], tab[good]))
    rc = np.median(reprojection_error(H_rc, px[good], tab[good]))
    assert rc < ls / 3


def test_ransac_needs_a_consensus_set():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 600, (10, 2))
    dst = rng.uniform(0, 400, (10, 2))        # unrelated, no consistent homography
    from ppstack.transforms.homography import fit_homography_ransac

    H, inliers = fit_homography_ransac(src, dst, threshold=0.5)
    assert inliers.sum() < len(src)


def test_ransac_with_exactly_four_points_is_the_plain_fit():
    sc = SceneSpec(blocks=[Block(200.0, 150.0, 0.0, 44.0, "red")])
    px, tab = sc.calibration_correspondences(grid=2)
    from ppstack.transforms.homography import fit_homography_ransac

    H, inliers = fit_homography_ransac(px, tab)
    assert inliers.all()
    assert np.allclose(H, fit_homography(px, tab))


def test_distortion_roundtrips():
    from ppstack.transforms.distortion import RadialDistortion

    rng = np.random.default_rng(2)
    pts = np.stack([rng.uniform(0, 640, 200), rng.uniform(0, 480, 200)], axis=1)
    for k1 in (0.0, 0.05, 0.12, -0.06):
        lens = RadialDistortion.for_image((640, 480), k1)
        assert np.abs(lens.distort(lens.undistort(pts)) - pts).max() < 1e-6


def test_distortion_raises_instead_of_silently_diverging():
    """Past the fold radius the inverse map has no fixed point."""
    from ppstack.transforms.distortion import DistortionDiverged, RadialDistortion

    lens = RadialDistortion.for_image((640, 480), -0.9)
    with pytest.raises(DistortionDiverged):
        lens.distort(np.array([[2000.0, 2000.0]]))


@pytest.mark.parametrize("true_k1", [0.0, 0.05, 0.10])
def test_k1_is_recovered_from_the_fiducials_alone(true_k1):
    from ppstack.transforms.distortion import estimate_k1

    sc = SceneSpec(blocks=[Block(200.0, 150.0, 0.0, 44.0, "red")], distortion_k1=true_k1)
    px, tab = sc.calibration_correspondences(noise_px=0.3, grid=4)
    lens, rms = estimate_k1(px, tab, sc.image_size)
    assert lens.k1 == pytest.approx(true_k1, abs=0.02)
    assert rms < 1.0


def test_estimating_distortion_needs_redundant_fiducials():
    from ppstack.transforms.distortion import estimate_k1

    sc = SceneSpec(blocks=[Block(200.0, 150.0, 0.0, 44.0, "red")])
    px, tab = sc.calibration_correspondences(grid=2)
    with pytest.raises(ValueError, match="more than 4"):
        estimate_k1(px, tab, sc.image_size)


def test_distortion_aware_calibration_beats_the_naive_one():
    sc = SceneSpec(blocks=[Block(200.0, 150.0, 0.0, 44.0, "red")], distortion_k1=0.10)
    px, tab = sc.calibration_correspondences(noise_px=0.3, grid=4)
    naive = CameraCalibration.from_correspondences(
        px, tab, table_to_base=TABLE_TO_BASE, bounds=BOUNDS, max_residual_mm=1e9
    )
    aware = CameraCalibration.from_correspondences(
        px, tab, table_to_base=TABLE_TO_BASE, bounds=BOUNDS, max_residual_mm=1e9,
        estimate_distortion=True, image_size=sc.image_size,
    )
    assert aware.rms_residual_mm < naive.rms_residual_mm / 2


def test_distortion_survives_a_json_roundtrip(tmp_path):
    sc = SceneSpec(blocks=[Block(200.0, 150.0, 0.0, 44.0, "red")], distortion_k1=0.08)
    px, tab = sc.calibration_correspondences(noise_px=0.3, grid=4)
    cal = CameraCalibration.from_correspondences(
        px, tab, table_to_base=TABLE_TO_BASE, bounds=BOUNDS, max_residual_mm=1e9,
        estimate_distortion=True, image_size=sc.image_size,
    )
    path = tmp_path / "cal.json"
    cal.to_json(path)
    other = CameraCalibration.from_json(path)
    assert other.distortion is not None
    assert other.distortion.k1 == pytest.approx(cal.distortion.k1)
    assert other.pixel_to_table(320, 240).as_tuple() == pytest.approx(
        cal.pixel_to_table(320, 240).as_tuple()
    )


def test_scene_projection_roundtrips_through_the_lens():
    for k1 in (0.0, 0.09, -0.07):
        sc = SceneSpec(blocks=[Block(200.0, 150.0, 0.0, 44.0, "red")], distortion_k1=k1)
        tab = np.array([[x, y] for y in (20.0, 150.0, 280.0) for x in (20.0, 200.0, 380.0)])
        assert np.abs(sc.unproject(sc.project(tab)) - tab).max() < 1e-9
