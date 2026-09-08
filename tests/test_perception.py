import numpy as np
import pytest

from ppstack.perception.color import ColorBand, rgb_to_hsv
from ppstack.perception.detector import BlockDetector, DetectorConfig
from ppstack.perception.morphology import (
    closing, connected_components, dilate, erode, opening, region_properties,
)
from ppstack.perception.orientation import square_orientation
from ppstack.perception.scene import PALETTE, Block, SceneSpec
from ppstack.scenarios import scene


def test_rgb_to_hsv_matches_known_values():
    img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]],
                   dtype=np.uint8)
    hsv = rgb_to_hsv(img)[0]
    assert hsv[0] == pytest.approx([0.0, 1.0, 1.0])
    assert hsv[1] == pytest.approx([120.0, 1.0, 1.0])
    assert hsv[2] == pytest.approx([240.0, 1.0, 1.0])
    assert hsv[3][1] == pytest.approx(0.0)          # white: no saturation
    assert hsv[4][1] == pytest.approx(0.0)          # black: saturation is defined as 0


def test_hue_band_wraps_through_zero():
    """Red straddles the 0/360 seam, which a naive lo <= h <= hi test misses."""
    band = ColorBand("red", 345.0, 15.0, sat_min=0.0, val_min=0.0)
    hsv = np.array([[[350.0, 1, 1], [5.0, 1, 1], [180.0, 1, 1]]])
    assert band.mask(hsv).tolist() == [[True, True, False]]


def test_value_threshold_would_fail_where_hue_survives():
    """The lighting gradient dims a block without moving its hue, which is the
    whole reason the detector thresholds in HSV rather than RGB."""
    bright = np.array([[PALETTE["red"]]], dtype=np.uint8)
    dim = (bright * 0.55).astype(np.uint8)
    h_bright, h_dim = rgb_to_hsv(bright)[0, 0], rgb_to_hsv(dim)[0, 0]
    assert abs(h_bright[0] - h_dim[0]) < 3.0        # hue barely moves
    assert h_bright[2] - h_dim[2] > 0.3             # value moves a lot


def test_erode_and_dilate_are_dual():
    rng = np.random.default_rng(0)
    m = rng.random((40, 40)) > 0.6
    inner = np.s_[2:-2, 2:-2]
    assert np.array_equal(erode(m, 1)[inner], (~dilate(~m, 1))[inner])


def test_morphology_does_not_wrap_at_the_border():
    m = np.zeros((10, 10), dtype=bool)
    m[0, :] = True
    assert not dilate(m, 1)[-1].any()


def test_opening_removes_specks_and_closing_fills_holes():
    m = np.zeros((30, 30), dtype=bool)
    m[5:15, 5:15] = True
    m[25, 25] = True                      # a speck
    m[9:11, 9:11] = False                 # a hole
    assert opening(m, 1).sum() == 100 - 4  # speck gone, hole still there
    assert closing(m, 1)[5:15, 5:15].all()  # hole filled


def test_connected_components_counts_and_separates():
    m = np.zeros((20, 20), dtype=bool)
    m[2:6, 2:6] = True
    m[12:16, 12:16] = True
    labels, n = connected_components(m)
    assert n == 2
    assert sorted(r.area_px for r in region_properties(labels, n)) == [16, 16]


def test_eight_connectivity_joins_a_diagonal_staircase():
    m = np.zeros((10, 10), dtype=bool)
    for i in range(8):
        m[i, i] = True
    assert connected_components(m, connectivity=8)[1] == 1
    assert connected_components(m, connectivity=4)[1] == 8


def test_region_moments_recover_a_bar_angle():
    yy, xx = np.mgrid[0:101, 0:101]
    a = np.deg2rad(30.0)
    lx = (xx - 50) * np.cos(-a) - (yy - 50) * np.sin(-a)
    ly = (xx - 50) * np.sin(-a) + (yy - 50) * np.cos(-a)
    labels, n = connected_components((np.abs(lx) < 30) & (np.abs(ly) < 6))
    r = region_properties(labels, n)[0]
    assert np.degrees(r.angle_rad) == pytest.approx(30.0, abs=1.0)
    assert r.eccentricity > 0.9


def test_square_orientation_is_accurate_where_moments_are_degenerate():
    """The second-moment axis of a square is undefined; psi4 is not."""
    rng = np.random.default_rng(0)
    worst_psi4 = 0.0
    for deg in np.arange(-45.0, 45.0, 2.5):
        a = np.deg2rad(deg)
        pts = rng.uniform(-1, 1, (3000, 2))
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        angle, conf = square_orientation(pts @ R.T)
        # Both are wrapped into a quarter turn, so compare modulo one.
        d = (angle - a + np.pi / 4) % (np.pi / 2) - np.pi / 4
        worst_psi4 = max(worst_psi4, abs(np.degrees(d)))
        assert conf > 0.3
    assert worst_psi4 < 3.0


def test_square_orientation_reports_low_confidence_for_a_disc():
    rng = np.random.default_rng(1)
    p = rng.normal(0, 1, (6000, 2))
    assert square_orientation(p[np.linalg.norm(p, axis=1) < 2])[1] < 0.1


def test_detector_finds_every_block_in_a_clean_scene():
    sc = SceneSpec(blocks=[
        Block(90.0, 80.0, 0.4, 46.0, "red"),
        Block(300.0, 200.0, -0.3, 46.0, "blue"),
        Block(200.0, 60.0, 0.1, 44.0, "green"),
    ], speckle_count=0)
    found = {d.color for d in BlockDetector(backend="numpy").detect(sc.render())}
    assert {"red", "green", "blue"} <= found


def test_detector_rejects_specks_below_the_area_threshold():
    sc = SceneSpec(blocks=[Block(200.0, 150.0, 0.0, 46.0, "red")], speckle_count=200)
    dets = BlockDetector(backend="numpy").detect(sc.render())
    assert len([d for d in dets if d.color == "red"]) == 1


def test_detector_survives_the_lighting_gradient():
    sc = SceneSpec(blocks=[Block(60.0, 250.0, 0.2, 46.0, "red"),
                           Block(350.0, 40.0, 0.2, 46.0, "red")],
                   lighting_gradient=0.6)
    reds = [d for d in BlockDetector(backend="numpy").detect(sc.render()) if d.color == "red"]
    assert len(reds) == 2


def test_obstacles_are_allowed_to_be_elongated():
    """A wall is a legitimate obstacle; the squareness filter is for grasp
    candidates only, and must not throw the wall away."""
    sc = scene("gap")
    dets = BlockDetector(backend="numpy").detect(sc.render())
    walls = [d for d in dets if d.is_obstacle]
    assert walls and max(d.eccentricity for d in walls) > 0.9


def test_detector_rejects_a_malformed_image():
    with pytest.raises(ValueError):
        BlockDetector().detect(np.zeros((10, 10), dtype=np.uint8))


def test_featureless_frame_yields_no_detections():
    """A flat mid-grey has no hue band and is too bright to be an obstacle."""
    grey = np.full((60, 60, 3), 128, dtype=np.uint8)
    assert BlockDetector(backend="numpy").detect(grey) == []


def test_an_all_black_frame_reads_as_one_obstacle():
    """Not a bug: the obstacle band is defined by darkness, so a blacked-out
    camera is correctly reported as the whole field being blocked."""
    dets = BlockDetector(backend="numpy").detect(np.zeros((60, 60, 3), np.uint8))
    assert len(dets) == 1 and dets[0].is_obstacle


@pytest.mark.parametrize("scenario", ["clear", "detour", "gap", "clutter"])
def test_numpy_and_opencv_backends_agree(scenario):
    cv2 = pytest.importorskip("cv2")
    frame = scene(scenario).render()
    cfg = DetectorConfig()
    a = BlockDetector(config=cfg, backend="numpy").detect(frame)
    b = BlockDetector(config=cfg, backend="opencv").detect(frame)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.color == y.color
        assert x.area_px == y.area_px
        assert x.centroid_px == pytest.approx(y.centroid_px, abs=1e-6)
        assert x.angle_rad == pytest.approx(y.angle_rad, abs=1e-9)
