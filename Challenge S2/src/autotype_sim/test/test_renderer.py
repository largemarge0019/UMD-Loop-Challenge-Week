"""Tests for core/renderer.py against the design spec section 9.

All randomness comes from seeded numpy Generators; the only timing test
asserts a loose bound so slow CI machines do not flake. Geometry and key map
come from the session fixtures (conftest.py): the board_geometry.yaml in
private/ when available, else ``DEFAULT_GEOMETRY`` -- nothing here depends on
a particular keyboard placement.

Pixel conventions asserted here: texture pixel ``i`` spans board
``[i, i+1) / px_per_m``; OpenCV (ArUco, warpPerspective, project_points with
``cx = (width-1)/2``) puts pixel centres on integers, so a physical texture
coordinate ``u`` is OpenCV coordinate ``u - 0.5``.
"""

import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from autotype_sim.core.board import (
    MARKER_IDS,
    R_WB_NOMINAL,
    BoardGeometry,
    BoardPose,
    SampleRanges,
    board_pose_from_center,
    marker_corners_board,
    marker_corners_world,
    sample_board_pose,
)
from autotype_sim.core.config import ArmConfig, CameraConfig
from autotype_sim.core.keymap import TKL_HEIGHT_U, TKL_WIDTH_U
from autotype_sim.core.kinematics import forward_kinematics, project_points
from autotype_sim.core.renderer import (
    CAP_BGR,
    DEFAULT_PX_PER_M,
    LABEL_BGR,
    MARKER_QUIET_MODULES,
    PLATE_BGR,
    REFINE_METHODS,
    aruco_dictionary,
    board_to_texture,
    build_texture,
    detect_markers,
    key_area_box_px,
    render,
    texture_corners_cv,
    texture_size,
    texture_to_board,
)

REPO = Path(__file__).resolve().parents[3]
GRID_PHOTO = REPO / "assets" / "keyboard_grid.png"
ARM = ArmConfig()
CAM = CameraConfig()
PPM = DEFAULT_PX_PER_M
IDS = set(MARKER_IDS)
CORNER_TOL_PX = 2.0  # the design spec s.9 requirement
# The strict 2 px tests run with the utility's default refinement and with
# OpenCV's stock default ("none"), so the design's plain-ArucoDetector claim
# is verified as stated.
STRICT_REFINES = ("contour", "none")

HOME = forward_kinematics(ARM, ARM.q_home)


@pytest.fixture(scope="module")
def texture(geometry, keymap) -> np.ndarray:
    return build_texture(geometry, keymap, PPM)


def _nominal_pose(geometry: BoardGeometry) -> BoardPose:
    """Unperturbed board (R_WB_nominal) at the centre of the sampling box."""
    ranges = SampleRanges()
    centre = 0.5 * (ranges.box_min + ranges.box_max)
    return board_pose_from_center(geometry, centre, 0.0, 0.0, 0.0)


def _truth_px(geometry: BoardGeometry, pose: BoardPose) -> dict[int, np.ndarray]:
    """Projected marker corners at q_home, {id: uv[4,2]}, ArUco order."""
    out = {}
    for mid, corners in marker_corners_world(geometry, pose).items():
        uv, z = project_points(CAM, HOME.R_cam, HOME.t_cam, corners)
        assert np.all(z > 0.0)
        out[mid] = uv
    return out


def _corner_errors(detected: dict[int, np.ndarray], truth: dict[int, np.ndarray]) -> np.ndarray:
    """Per-corner pixel error for all four ids, in matching corner order."""
    assert set(detected) == IDS, f"detected ids {sorted(detected)}"
    return np.concatenate(
        [np.linalg.norm(detected[mid] - truth[mid], axis=1) for mid in MARKER_IDS]
    )


def _flat_marker_corners_cv(geometry: BoardGeometry, mid: int) -> np.ndarray:
    """Where the printed corners of marker `mid` sit in the flat texture (OpenCV px)."""
    return board_to_texture(marker_corners_board(geometry)[mid][:, :2], PPM) - 0.5


# --------------------------------------------------------------------------- #
# Scale helpers
# --------------------------------------------------------------------------- #


def test_scale_helpers_are_pure_scale(geometry):
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.0, 0.4, size=(10, 2))
    np.testing.assert_allclose(texture_to_board(board_to_texture(xy, PPM), PPM), xy, rtol=0, atol=1e-15)
    np.testing.assert_allclose(board_to_texture([geometry.panel_w, geometry.panel_h], PPM), [1600.0, 700.0])
    np.testing.assert_allclose(texture_to_board([1600.0, 700.0], PPM), [geometry.panel_w, geometry.panel_h])
    assert board_to_texture(xy, PPM).dtype == np.float64


def test_texture_size_rounds_panel_extent(geometry):
    assert texture_size(geometry, PPM) == (round(geometry.panel_w * PPM), round(geometry.panel_h * PPM))
    assert texture_size(geometry, 3333.0) == (round(0.4 * 3333.0), round(0.175 * 3333.0))


def test_texture_corners_cv_are_half_pixel_outside():
    tex = np.zeros((7, 11, 3), np.uint8)
    np.testing.assert_array_equal(
        texture_corners_cv(tex), [[-0.5, -0.5], [10.5, -0.5], [10.5, 6.5], [-0.5, 6.5]]
    )
    assert texture_corners_cv(tex).dtype == np.float32


# --------------------------------------------------------------------------- #
# 1. build_texture
# --------------------------------------------------------------------------- #


def test_texture_size_dtype_and_black_background(geometry, texture):
    w, h = texture_size(geometry, PPM)
    assert texture.shape == (h, w, 3) == (700, 1600, 3)
    assert texture.dtype == np.uint8
    # Outside every quiet zone and the plate the panel is black.
    assert texture[0, 0].tolist() == [0, 0, 0]
    assert texture[0, w // 2].tolist() == [0, 0, 0]
    assert texture[h - 1, w // 2].tolist() == [0, 0, 0]
    assert not np.any(texture[:, w // 2 - 20 : w // 2 + 20][:8])  # top strip, mid-panel


@pytest.mark.parametrize("ppm", [2500.0, 3000.0, 5000.0])
def test_texture_size_follows_px_per_m(geometry, keymap, ppm):
    tex = build_texture(geometry, keymap, ppm)
    assert tex.shape == (round(geometry.panel_h * ppm), round(geometry.panel_w * ppm), 3)
    assert tex.dtype == np.uint8


@pytest.mark.parametrize("refine", sorted(REFINE_METHODS))
def test_flat_texture_markers_detect_at_their_centres(geometry, texture, refine):
    det = detect_markers(texture, refine=refine)
    assert set(det) == IDS
    side_px = geometry.marker_size * PPM
    for mid in MARKER_IDS:
        c = det[mid]
        assert c.shape == (4, 2)
        expected_centre = board_to_texture(geometry.marker_centers[mid], PPM) - 0.5
        assert np.linalg.norm(c.mean(axis=0) - expected_centre) <= 1.5
        edges = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
        assert np.all(np.abs(edges - side_px) <= 0.02 * side_px), edges


def test_flat_texture_markers_are_upright(geometry, texture):
    """ArUco's TL,TR,BR,BL of the printed marker must coincide with the
    board-frame corner order of marker_corners_board (marker y along -Y_B)."""
    det = detect_markers(texture)
    for mid in MARKER_IDS:
        err = np.linalg.norm(det[mid] - _flat_marker_corners_cv(geometry, mid), axis=1)
        assert np.all(err <= 1.5), (mid, err)


def test_marker_pixels_match_generateImageMarker(geometry, texture):
    """Each marker is the cv2.aruco bitmap, pasted verbatim and upright."""
    dictionary = aruco_dictionary(geometry.marker_dict)
    side = int(round(geometry.marker_size * PPM))
    for mid in MARKER_IDS:
        ref = cv2.aruco.generateImageMarker(dictionary, mid, side)
        x0, y0 = (int(round(v)) for v in board_to_texture(geometry.marker_centers[mid], PPM) - side / 2)
        patch = texture[y0 : y0 + side, x0 : x0 + side]
        assert patch.shape == (side, side, 3)
        for ch in range(3):
            np.testing.assert_array_equal(patch[:, :, ch], ref)


def test_quiet_zone_is_white_for_one_module(geometry, texture):
    side = int(round(geometry.marker_size * PPM))
    module = side // 6  # 4x4 bits + 1-module border each side
    assert module >= 1
    for mid in MARKER_IDS:
        x0, y0 = (int(round(v)) for v in board_to_texture(geometry.marker_centers[mid], PPM) - side / 2)
        ring = texture[y0 - module : y0 + side + module, x0 - module : x0 + side + module].copy()
        ring[module : module + side, module : module + side] = 255  # ignore the marker itself
        assert np.all(ring == 255), mid


def test_synthetic_keyboard_is_drawn_from_keymap(geometry, texture, keymap):
    """Plate is dark grey, caps lighter, labels red-ish, and the cap edges sit
    exactly on the KeyMap rectangles (texture == truth by construction)."""
    (ox, oy), w, h = geometry.kb_origin, geometry.kb_w, geometry.kb_h
    px0, py0 = (int(round(v)) for v in board_to_texture((ox, oy), PPM))
    px1, py1 = (int(round(v)) for v in board_to_texture((ox + w, oy + h), PPM))
    plate = texture[py0:py1, px0:px1]
    assert plate.mean() > 30.0

    checked = 0
    for key in list(keymap.alnum()) + [keymap.by_name["SPACE"]]:
        x0, y0 = (int(round(v)) for v in board_to_texture((key.x0, key.y0), PPM))
        x1, y1 = (int(round(v)) for v in board_to_texture((key.x1, key.y1), PPM))
        yc, xc = (y0 + y1) // 2, (x0 + x1) // 2
        # Just inside each edge at mid-span: cap colour (labels use <= 80% of the width).
        assert texture[yc, x0].tolist() == list(CAP_BGR), key.name
        assert texture[yc, x1 - 1].tolist() == list(CAP_BGR), key.name
        # Just outside each edge: the plate shows through the inset gap.
        assert texture[yc, x0 - 1].tolist() == list(PLATE_BGR), key.name
        assert texture[yc, x1].tolist() == list(PLATE_BGR), key.name
        assert texture[y0 - 1, xc].tolist() == list(PLATE_BGR), key.name
        assert texture[y1, xc].tolist() == list(PLATE_BGR), key.name
        # A red-ish legend exists inside the cap.
        cap = texture[y0:y1, x0:x1].astype(int)
        reddish = (cap[:, :, 2] > cap[:, :, 0] + 60) & (cap[:, :, 2] > cap[:, :, 1] + 60)
        assert reddish.any(), key.name
        checked += 1
    assert checked == 37
    assert LABEL_BGR[2] > LABEL_BGR[0] and LABEL_BGR[2] > LABEL_BGR[1]


def test_build_texture_rejects_bad_inputs(geometry, keymap):
    with pytest.raises(ValueError):
        build_texture(geometry, keymap, 0.0)
    with pytest.raises(ValueError):
        build_texture(geometry, keymap, 200.0)  # 4 px marker: fewer pixels than modules
    with pytest.raises(ValueError):
        aruco_dictionary("DICT_DOES_NOT_EXIST")
    with pytest.raises(ValueError):
        detect_markers(np.zeros((10, 10), np.uint8), refine="magic")


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan"), -1.0, 0.0])
def test_build_texture_rejects_non_finite_px_per_m(geometry, keymap, bad):
    """Every bad px_per_m is a ValueError (inf used to escape as OverflowError
    from texture_size's int(round(...)))."""
    with pytest.raises(ValueError):
        build_texture(geometry, keymap, bad)


def _geometry_with_marker_clearance(geometry: BoardGeometry, d: float) -> BoardGeometry:
    """``geometry`` with all four marker centres exactly ``d`` metres in from their panel corner."""
    w, h = geometry.panel_w, geometry.panel_h
    return BoardGeometry(
        panel_w=w,
        panel_h=h,
        marker_size=geometry.marker_size,
        marker_dict=geometry.marker_dict,
        marker_centers={0: [d, d], 1: [w - d, d], 2: [w - d, h - d], 3: [d, h - d]},
        kb_origin=geometry.kb_origin,
        kb_w=geometry.kb_w,
        kb_h=geometry.kb_h,
        key_pitch=geometry.key_pitch,
        key_area_origin=geometry.key_area_origin,
        key_inset=geometry.key_inset,
        pose_salt=geometry.pose_salt,
    )


@pytest.mark.parametrize("clearance_mm", [10.0, 11.0, 12.0])
def test_build_texture_rejects_marker_whose_quiet_zone_is_clipped(geometry, keymap, clearance_mm):
    """The design spec s.9: every marker gets a white quiet border >= 1 module.
    BoardGeometry accepts centres down to marker_size/2 (10 mm) from the edge,
    where the quiet zone would be clipped by the texture edge -- at 10 mm the
    black marker border touches the black background and ArUco finds nothing,
    at 11 mm roughly half the sampled poses drop a marker. build_texture must
    refuse such a geometry instead of silently shipping a broken panel."""
    geom = _geometry_with_marker_clearance(geometry, clearance_mm / 1e3)  # legal for BoardGeometry
    with pytest.raises(ValueError, match="quiet zone"):
        build_texture(geom, keymap, PPM)


def test_build_texture_accepts_marker_with_one_module_clearance(geometry, keymap):
    """Just past marker_size/2 + 1 module (10 + 3.33 mm) the quiet zone fits:
    the white ring is complete and the markers detect flat and rendered."""
    geom = _geometry_with_marker_clearance(geometry, 0.0135)
    tex = build_texture(geom, keymap, PPM)
    side = int(round(geom.marker_size * PPM))
    module = side // 6
    for mid in MARKER_IDS:
        x0, y0 = (int(round(v)) for v in board_to_texture(geom.marker_centers[mid], PPM) - side / 2)
        ring = tex[y0 - module : y0 + side + module, x0 - module : x0 + side + module].copy()
        assert ring.shape == (side + 2 * module, side + 2 * module, 3), mid
        ring[module : module + side, module : module + side] = 255
        assert np.all(ring == 255), mid
    assert set(detect_markers(tex)) == IDS
    for seed in range(5):
        pose = sample_board_pose(seed, geom, ARM, CAM)
        img = render(CAM, HOME.R_cam, HOME.t_cam, pose, tex, geom)
        assert set(detect_markers(img)) == IDS, seed


# --------------------------------------------------------------------------- #
# 2/3. render + detect against projected truth
# --------------------------------------------------------------------------- #


def test_render_shape_dtype_and_black_outside_panel(geometry, texture):
    pose = _nominal_pose(geometry)
    img = render(CAM, HOME.R_cam, HOME.t_cam, pose, texture, geometry)
    assert img.shape == (CAM.height, CAM.width, 3)
    assert img.dtype == np.uint8
    uv, z = project_points(CAM, HOME.R_cam, HOME.t_cam, pose.corners_world(geometry))
    assert np.all(z > 0.0)
    mask = np.zeros((CAM.height, CAM.width), np.uint8)
    cv2.fillPoly(mask, [np.round(uv).astype(np.int32)], 255)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    assert not np.any(img[mask == 0])
    inside = img[mask > 0]
    assert inside.size > 0 and np.mean(inside.max(axis=1) > 0) > 0.5


def test_render_is_bit_exact_at_unit_scale(geometry, keymap):
    """Fronto-parallel view at 1 texture px per image px: the warp must
    reproduce the texture pixel-for-pixel. This pins the pixel-centre
    convention (texture corners at -0.5); an off-by-half homography blends
    neighbours and fails by hundreds of grey levels."""
    ppm = 2000.0
    tex = build_texture(geometry, keymap, ppm)  # 800 x 350
    tw, th = texture_size(geometry, ppm)
    depth = CAM.fx / ppm  # fx * panel_w / depth == tw  <=>  1 px per px
    R_cam = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # cam z = +X_W
    t_cam = np.zeros(3)
    half = np.array([geometry.panel_w / 2.0, geometry.panel_h / 2.0, 0.0])
    pose = BoardPose(R_WB_NOMINAL, np.array([depth, 0.0, 0.0]) - R_WB_NOMINAL @ half)

    img = render(CAM, R_cam, t_cam, pose, tex, geometry)
    u0 = int(round(CAM.cx - tw / 2.0 + 0.5))
    v0 = int(round(CAM.cy - th / 2.0 + 0.5))
    np.testing.assert_array_equal(img[v0 : v0 + th, u0 : u0 + tw], tex)
    outside = img.copy()
    outside[v0 : v0 + th, u0 : u0 + tw] = 0
    assert not np.any(outside)


@pytest.mark.parametrize("refine", STRICT_REFINES)
def test_nominal_pose_markers_within_2px(geometry, texture, refine):
    pose = _nominal_pose(geometry)
    img = render(CAM, HOME.R_cam, HOME.t_cam, pose, texture, geometry)
    det = detect_markers(img, refine=refine)
    err = _corner_errors(det, _truth_px(geometry, pose))
    assert err.max() < CORNER_TOL_PX, err.max()


@pytest.mark.parametrize("refine", STRICT_REFINES)
def test_twenty_sampled_poses_markers_within_2px(geometry, texture, refine):
    worst = 0.0
    worst_seed = -1
    for seed in range(20):
        pose = sample_board_pose(seed, geometry, ARM, CAM)
        img = render(CAM, HOME.R_cam, HOME.t_cam, pose, texture, geometry)
        det = detect_markers(img, refine=refine)
        err = _corner_errors(det, _truth_px(geometry, pose))
        if err.max() > worst:
            worst, worst_seed = float(err.max()), seed
    print(f"[renderer] refine={refine}: worst corner error over 20 poses = {worst:.3f} px (seed {worst_seed})")
    assert worst < CORNER_TOL_PX, (worst, worst_seed)


def test_sampled_pose_is_reproducible_from_seed(geometry, texture):
    a = sample_board_pose(7, geometry, ARM, CAM)
    b = sample_board_pose(7, geometry, ARM, CAM)
    np.testing.assert_array_equal(
        render(CAM, HOME.R_cam, HOME.t_cam, a, texture, geometry),
        render(CAM, HOME.R_cam, HOME.t_cam, b, texture, geometry),
    )


# --------------------------------------------------------------------------- #
# 4. behind the camera
# --------------------------------------------------------------------------- #


def test_pose_behind_camera_is_all_black(geometry, texture):
    behind = board_pose_from_center(geometry, np.array([-1.0, 0.0, 0.36]), 0.0, 0.0, 0.0)
    _, z = project_points(CAM, HOME.R_cam, HOME.t_cam, behind.corners_world(geometry))
    assert np.all(z < 0.0)
    img = render(CAM, HOME.R_cam, HOME.t_cam, behind, texture, geometry)
    assert img.shape == (CAM.height, CAM.width, 3) and img.dtype == np.uint8
    assert not np.any(img)


def test_pose_straddling_camera_plane_is_all_black(geometry, texture):
    """Any single corner with P_C.z <= 0 blanks the frame (no wrap-around)."""
    half = np.array([geometry.panel_w / 2.0, geometry.panel_h / 2.0, 0.0])
    straddle = BoardPose(R_WB_NOMINAL, HOME.t_cam - R_WB_NOMINAL @ half)  # centre at the camera
    _, z = project_points(CAM, HOME.R_cam, HOME.t_cam, straddle.corners_world(geometry))
    assert np.any(z <= 0.0) and np.any(z > 0.0)
    img = render(CAM, HOME.R_cam, HOME.t_cam, straddle, texture, geometry)
    assert not np.any(img)


def test_out_buffer_is_reused(geometry, texture):
    buf = np.empty((CAM.height, CAM.width, 3), np.uint8)
    pose = _nominal_pose(geometry)
    res = render(CAM, HOME.R_cam, HOME.t_cam, pose, texture, geometry, out=buf)
    assert res is buf
    np.testing.assert_array_equal(buf, render(CAM, HOME.R_cam, HOME.t_cam, pose, texture, geometry))
    behind = board_pose_from_center(geometry, np.array([-1.0, 0.0, 0.36]), 0.0, 0.0, 0.0)
    res = render(CAM, HOME.R_cam, HOME.t_cam, behind, texture, geometry, out=buf)
    assert res is buf and not np.any(buf)


def test_render_rejects_bad_out_buffer(geometry, texture):
    """The design spec s.9: render returns (H, W, 3) uint8. A mis-shaped, mis-typed or
    non-packed ``out`` must raise for visible AND behind-camera poses -- it used
    to be silently ignored on visible frames (fresh array returned, caller's
    buffer stale) and zero-filled-and-returned on black frames (wrong shape),
    so the returned shape depended on the pose. The caller's buffer is left
    untouched."""
    pose = _nominal_pose(geometry)
    behind = board_pose_from_center(geometry, np.array([-1.0, 0.0, 0.36]), 0.0, 0.0, 0.0)
    bad = [
        np.full((CAM.width, CAM.height, 3), 7, np.uint8),  # swapped dims
        np.full((CAM.height, CAM.width), 7, np.uint8),  # no channel axis
        np.full((480, 640, 3), 7, np.uint8),  # wrong size
        np.full((CAM.height, CAM.width, 3), 7, np.float32),  # wrong dtype
        np.asfortranarray(np.full((CAM.height, CAM.width, 3), 7, np.uint8)),  # column-major
        np.full((CAM.height, CAM.width * 2, 3), 7, np.uint8)[:, ::2],  # column-strided view
    ]
    for buf in bad:
        for p in (pose, behind):
            snapshot = buf.copy()
            with pytest.raises(ValueError):
                render(CAM, HOME.R_cam, HOME.t_cam, p, texture, geometry, out=buf)
            np.testing.assert_array_equal(buf, snapshot)
    with pytest.raises(ValueError):
        render(CAM, HOME.R_cam, HOME.t_cam, pose, texture, geometry, out=[[0]])


def test_render_out_accepts_packed_views(geometry, texture):
    """A correctly shaped view whose pixels are packed within rows (a
    sub-window, or every other row of a taller buffer) is written in place:
    OpenCV supports a row step, so the guard must not demand C-contiguity."""
    pose = _nominal_pose(geometry)
    ref = render(CAM, HOME.R_cam, HOME.t_cam, pose, texture, geometry)
    sub = np.zeros((CAM.height + 10, CAM.width + 10, 3), np.uint8)[5:-5, 5:-5]
    assert render(CAM, HOME.R_cam, HOME.t_cam, pose, texture, geometry, out=sub) is sub
    np.testing.assert_array_equal(sub, ref)
    rows = np.zeros((CAM.height * 2, CAM.width, 3), np.uint8)[::2]
    assert render(CAM, HOME.R_cam, HOME.t_cam, pose, texture, geometry, out=rows) is rows
    np.testing.assert_array_equal(rows, ref)


# --------------------------------------------------------------------------- #
# 5. timing
# --------------------------------------------------------------------------- #


def test_render_time_at_1280x720(geometry, texture):
    pose = _nominal_pose(geometry)
    for _ in range(5):  # warm up
        render(CAM, HOME.R_cam, HOME.t_cam, pose, texture, geometry)
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        render(CAM, HOME.R_cam, HOME.t_cam, pose, texture, geometry)
    mean_ms = (time.perf_counter() - t0) / n * 1e3
    print(f"[renderer] mean render time over {n} calls at {CAM.width}x{CAM.height}: {mean_ms:.3f} ms (target < 5 ms)")
    assert mean_ms < 20.0, mean_ms


# --------------------------------------------------------------------------- #
# 6. photo path
# --------------------------------------------------------------------------- #


def _plate_box_px(geometry: BoardGeometry) -> tuple[int, int, int, int]:
    (ox, oy), w, h = geometry.kb_origin, geometry.kb_w, geometry.kb_h
    px0, py0 = (int(round(v)) for v in board_to_texture((ox, oy), PPM))
    px1, py1 = (int(round(v)) for v in board_to_texture((ox + w, oy + h), PPM))
    return px0, py0, px1, py1


def _grid_box_px(geometry: BoardGeometry) -> tuple[int, int, int, int]:
    return key_area_box_px(geometry, PPM)


def test_photo_fills_key_area_and_markers_still_detect(geometry, keymap):
    rng = np.random.default_rng(123)
    photo = rng.integers(0, 256, size=(300, 900, 3), dtype=np.uint8)
    tex = build_texture(geometry, keymap, PPM, photo=photo)
    assert tex.shape == (700, 1600, 3) and tex.dtype == np.uint8

    gx0, gy0, gx1, gy1 = _grid_box_px(geometry)
    grid = tex[gy0:gy1, gx0:gx1]
    assert grid.mean() > 100.0  # noise, not black
    assert np.mean(grid.max(axis=2) > 0) > 0.95

    # The bezel around the grid is the flat plate colour, never the photo.
    px0, py0, px1, py1 = _plate_box_px(geometry)
    assert px0 < gx0 and py0 < gy0 and gx1 < px1 and gy1 < py1
    xc = (gx0 + gx1) // 2
    bezel = tex[py0:gy0, xc - 20 : xc + 20]
    assert bezel.size and np.all(bezel == np.array(PLATE_BGR, np.uint8))

    # Outside the plate and the quiet zones the panel stays black.
    assert not np.any(tex[:8, 300:1300])
    assert not np.any(tex[-8:, 300:1300])

    det = detect_markers(tex)
    assert set(det) == IDS
    for mid in MARKER_IDS:
        expected_centre = board_to_texture(geometry.marker_centers[mid], PPM) - 0.5
        assert np.linalg.norm(det[mid].mean(axis=0) - expected_centre) <= 1.5


def _marker_intrusion_px(
    geometry: BoardGeometry, box: tuple[float, float, float, float] | None = None
) -> int:
    """How far (px, plus one) a marker's white quiet zone can reach into
    ``box`` (board metres ``x0, y0, x1, y1``, default the plate) from its
    corners -- depends on where the box sits, so derived."""
    half = geometry.marker_size / 2.0 + MARKER_QUIET_MODULES * geometry.marker_size / 6.0
    if box is None:
        (kx, ky), w, h = geometry.kb_origin, geometry.kb_w, geometry.kb_h
        box = (kx, ky, kx + w, ky + h)
    bx0, by0, bx1, by1 = box
    reach = 0.0
    for cx, cy in geometry.marker_centers.values():
        dx = min(cx + half, bx1) - max(cx - half, bx0)
        dy = min(cy + half, by1) - max(cy - half, by0)
        if dx > 0.0 and dy > 0.0:  # the quiet square overlaps the box
            reach = max(reach, dx, dy)
    return int(math.ceil(reach * PPM)) + 1


def test_photo_is_resized_into_key_area_rectangle(geometry, keymap):
    """Photo already at the key grid's pixel size goes in verbatim (no resample).

    The grid rectangle is what the photo covers, and it is derived from
    ``key_area_origin`` and the public 18.25u x 6.25u extent -- the plate
    around it never scales the image.
    """
    gx0, gy0, gx1, gy1 = _grid_box_px(geometry)
    rng = np.random.default_rng(5)
    photo = rng.integers(0, 256, size=(gy1 - gy0, gx1 - gx0, 3), dtype=np.uint8)
    tex = build_texture(geometry, keymap, PPM, photo=photo)
    # Compare away from the four corners, where the marker quiet zones can
    # still overwrite the grid.
    ax, ay = geometry.key_area_origin
    m = _marker_intrusion_px(
        geometry,
        (ax, ay, ax + TKL_WIDTH_U * geometry.key_pitch, ay + TKL_HEIGHT_U * geometry.key_pitch),
    )
    assert 0 < m < min(gx1 - gx0, gy1 - gy0) // 4
    np.testing.assert_array_equal(tex[gy0 + m : gy1 - m, gx0:gx1], photo[m:-m])
    np.testing.assert_array_equal(tex[gy0:gy1, gx0 + m : gx1 - m], photo[:, m:-m])


def test_photo_accepts_grey_and_rejects_bad_shapes(geometry, keymap):
    rng = np.random.default_rng(9)
    grey = rng.integers(0, 256, size=(100, 300), dtype=np.uint8)
    tex = build_texture(geometry, keymap, PPM, photo=grey)
    gx0, gy0, gx1, gy1 = _grid_box_px(geometry)
    assert tex[gy0:gy1, gx0:gx1].mean() > 100.0
    with pytest.raises(ValueError):
        build_texture(geometry, keymap, PPM, photo=np.zeros((10, 10, 2), np.uint8))


def _nan_photo() -> np.ndarray:
    p = np.full((20, 40, 3), 100.0)
    p[3, 4, 1] = np.nan
    return p


def _inf_photo() -> np.ndarray:
    p = np.full((20, 40, 3), 100.0)
    p[3, 4, 1] = np.inf
    return p


@pytest.mark.parametrize(
    "make",
    [
        lambda: np.zeros((0, 0, 3), np.uint8),  # empty (was raw cv2.error from resize)
        lambda: np.zeros((0, 5, 3), np.uint8),
        lambda: np.zeros((5, 0), np.uint8),
        lambda: np.ones((20, 40, 3), bool),  # was silently accepted as an all-1 (black) plate
        lambda: np.ones((20, 40), bool),  # was raw cv2.error from cvtColor
        lambda: np.full((5, 5, 3), None, dtype=object),  # was TypeError from np.clip
        lambda: np.zeros(10, np.uint8),
        lambda: np.zeros((10, 10, 1), np.uint8),
        lambda: np.zeros((2, 10, 10, 3), np.uint8),
        _nan_photo,  # was a RuntimeWarning and a black plate
        _inf_photo,
    ],
)
def test_photo_rejects_degenerate_arrays_with_value_error(geometry, keymap, make):
    """Empty, bool, object, non-finite or mis-shaped photos raise ValueError --
    never a raw cv2.error, and never a silently black plate."""
    with pytest.raises(ValueError):
        build_texture(geometry, keymap, PPM, photo=make())


@pytest.mark.parametrize(
    "make",
    [
        lambda: np.full((20, 40), 100.0),  # float64 grey (cvtColor rejected float64 before)
        lambda: np.full((20, 40, 3), 100),  # int64 BGR
        lambda: np.full((20, 40, 3), 100, np.uint16),  # uint16 BGR
        lambda: np.full((20, 40, 4), 100.0, np.float32),  # float32 BGRA
        lambda: np.full((20, 40, 3), 100.0)[::2, ::2],  # non-contiguous float view
    ],
)
def test_photo_accepts_finite_numeric_dtypes(geometry, keymap, make):
    """Any finite integer / floating grey, BGR or BGRA array is clipped to uint8 and used."""
    tex = build_texture(geometry, keymap, PPM, photo=make())
    gx0, gy0, gx1, gy1 = _grid_box_px(geometry)
    yc, xc = (gy0 + gy1) // 2, (gx0 + gx1) // 2
    assert np.all(tex[yc - 5 : yc + 5, xc - 5 : xc + 5] == 100)


@pytest.mark.skipif(not GRID_PHOTO.exists(), reason="assets/keyboard_grid.png not present")
def test_registered_grid_photo_renders_and_detects(geometry, keymap):
    photo = cv2.imread(str(GRID_PHOTO), cv2.IMREAD_COLOR)
    assert photo is not None
    tex = build_texture(geometry, keymap, PPM, photo=photo)
    assert set(detect_markers(tex)) == IDS
    pose = _nominal_pose(geometry)
    img = render(CAM, HOME.R_cam, HOME.t_cam, pose, tex, geometry)
    err = _corner_errors(detect_markers(img), _truth_px(geometry, pose))
    assert err.max() < CORNER_TOL_PX, err.max()
