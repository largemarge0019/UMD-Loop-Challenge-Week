"""Tests for core/board.py against the design spec sections 1, 4 and 6.

All randomness comes from seeded numpy Generators. Geometry comes from the
session ``geometry`` / ``geometry_dict`` fixtures (conftest.py): the
board_geometry.yaml in private/ when it is available, otherwise
``DEFAULT_GEOMETRY`` from ``autotype_sim.testing``. Nothing here pins the
keyboard placement to specific coordinates -- only that whatever placement is
in use is self-consistent (grid inside plate inside panel) -- so the suite
passes from a bare checkout without private/.
"""

import itertools

import numpy as np
import pytest
import yaml

from autotype_sim.core.board import (
    MARKER_IDS,
    PUBLIC_FIELDS,
    R_WB_NOMINAL,
    BoardGeometry,
    BoardPose,
    SampleRanges,
    board_pose_from_center,
    marker_corners_board,
    marker_corners_world,
    pose_rng,
    sample_board_pose,
)
from autotype_sim.core.config import DEG, ArmConfig, CameraConfig
from autotype_sim.core.keymap import TKL_HEIGHT_U, TKL_WIDTH_U, generate_tkl
from autotype_sim.core.kinematics import (
    forward_kinematics,
    in_frame,
    project_points,
    rot_x,
    rot_y,
    rot_z,
)
from autotype_sim.testing import DEFAULT_GEOMETRY, PLACEMENT_FIELDS

ARM = ArmConfig()
CAM = CameraConfig()
EX, EY, EZ = np.eye(3)
MARGIN_PX = 40.0
SHOULDER_DIST = (0.55, 1.2)  # guard on panel-centre distance from the shoulder
MARKER_EDGE_M = 0.016  # the design spec s.4 (public): marker centres 16 mm in from each panel edge


def _assert_rotation(R: np.ndarray, tol: float = 1e-12) -> None:
    assert R.shape == (3, 3)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=tol, rtol=0)
    assert abs(np.linalg.det(R) - 1.0) < tol


def _random_pose(geometry: BoardGeometry, rng: np.random.Generator) -> BoardPose:
    """Arbitrary proper pose (any orientation), not necessarily in frame."""
    center = rng.uniform([0.5, -0.5, 0.0], [1.5, 0.5, 1.0])
    yaw, pitch, roll = rng.uniform(-np.pi, np.pi, size=3)
    return board_pose_from_center(geometry, center, yaw, pitch, roll)


def _edge_margin(uv: np.ndarray) -> float:
    """Smallest distance (px) of any projected point to any image edge."""
    u, v = uv[:, 0], uv[:, 1]
    return float(
        min(u.min(), CAM.width - 1 - u.max(), v.min(), CAM.height - 1 - v.max())
    )


def _yaw_of(pose: BoardPose) -> float:
    """Signed yaw of a pose built as Rz(yaw) Ry(pitch) Rx(roll) R_WB_nominal.

    M = R @ R_nominal^T = Rz(yaw) Ry(pitch) Rx(roll); its ZYX Euler yaw is
    atan2(M[1,0], M[0,0]) = atan2(sin yaw cos pitch, cos yaw cos pitch), exact
    for |pitch| < 90 deg. For pitch = roll = 0 this is atan2(R[0,0], -R[1,0]).
    """
    M = pose.R @ R_WB_NOMINAL.T
    return float(np.arctan2(M[1, 0], M[0, 0]))


def _yaw_from_first_draw(seed: int, salt: int, ranges: SampleRanges) -> float:
    """The yaw the sampler draws first for `seed` (the design spec s.6 draw order).

    Uses the SALTED stream -- the seed alone reaches a different one.
    """
    g = pose_rng(seed, salt)
    g.uniform(ranges.box_min, ranges.box_max)
    return float(g.uniform(-ranges.yaw, ranges.yaw))


# --------------------------------------------------------------------------- #
# 1. Nominal orientation
# --------------------------------------------------------------------------- #


def test_nominal_rotation_columns_exact_and_right_handed():
    np.testing.assert_array_equal(R_WB_NOMINAL[:, 0], [0.0, -1.0, 0.0])
    np.testing.assert_array_equal(R_WB_NOMINAL[:, 1], [0.0, 0.0, -1.0])
    np.testing.assert_array_equal(R_WB_NOMINAL[:, 2], [1.0, 0.0, 0.0])
    assert R_WB_NOMINAL.dtype == np.float64
    assert abs(np.linalg.det(R_WB_NOMINAL) - 1.0) < 1e-15
    np.testing.assert_array_equal(
        np.cross(R_WB_NOMINAL[:, 0], R_WB_NOMINAL[:, 1]), R_WB_NOMINAL[:, 2]
    )
    _assert_rotation(R_WB_NOMINAL, tol=1e-15)


# --------------------------------------------------------------------------- #
# 2. Transform round trips
# --------------------------------------------------------------------------- #


def test_to_world_to_board_are_inverses_for_random_poses(geometry):
    rng = np.random.default_rng(2)
    for _ in range(100):
        pose = _random_pose(geometry, rng)
        _assert_rotation(pose.R)
        b = rng.normal(size=(7, 5, 3))
        P = pose.to_world(b)
        assert P.shape == b.shape
        np.testing.assert_allclose(pose.to_board(P), b, atol=1e-12, rtol=0)
        W = rng.normal(size=(11, 3))
        np.testing.assert_allclose(pose.to_world(pose.to_board(W)), W, atol=1e-12, rtol=0)
        # Explicit formula on a single (3,) point.
        p = rng.normal(size=3)
        np.testing.assert_allclose(pose.to_world(p), pose.R @ p + pose.t, atol=1e-13)
        np.testing.assert_allclose(pose.to_board(p), pose.R.T @ (p - pose.t), atol=1e-13)


def test_to_world_matches_formula_for_nominal_pose():
    t = np.array([1.0, 0.2, 0.5])
    pose = BoardPose(R=R_WB_NOMINAL, t=t)
    # board +x (right, as seen from the arm) is world -Y; board +y (down) is world -Z.
    np.testing.assert_allclose(pose.to_world([0.1, 0.0, 0.0]), t + 0.1 * -EY, atol=1e-15)
    np.testing.assert_allclose(pose.to_world([0.0, 0.1, 0.0]), t + 0.1 * -EZ, atol=1e-15)
    np.testing.assert_allclose(pose.to_world([0.0, 0.0, 0.1]), t + 0.1 * EX, atol=1e-15)
    np.testing.assert_array_equal(pose.to_world(np.zeros(3)), t)


def test_board_pose_validates_inputs():
    with pytest.raises(ValueError):
        BoardPose(R=np.eye(3) * 2.0, t=np.zeros(3))
    with pytest.raises(ValueError):
        BoardPose(R=np.diag([1.0, 1.0, -1.0]), t=np.zeros(3))  # improper (det -1)
    with pytest.raises(ValueError):
        BoardPose(R=np.eye(3), t=np.zeros(2))
    pose = BoardPose(R=R_WB_NOMINAL, t=[1, 0, 0])
    assert pose.R.dtype == np.float64 and pose.t.dtype == np.float64
    with pytest.raises(ValueError):
        pose.to_world(np.zeros((4, 2)))


def test_board_pose_rejects_non_finite_R_and_t():
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            BoardPose(R=R_WB_NOMINAL, t=[bad, 0.0, 0.0])
        with pytest.raises(ValueError, match="finite"):
            BoardPose(R=R_WB_NOMINAL, t=[0.0, 0.0, bad])
        R = R_WB_NOMINAL.copy()
        R[1, 2] = bad
        with pytest.raises(ValueError, match="finite"):
            BoardPose(R=R, t=np.zeros(3))
    with pytest.raises(ValueError):
        BoardPose(R=np.full((3, 3), np.nan), t=np.zeros(3))
    # The module constant is read-only; poses built from it own writable copies.
    assert not R_WB_NOMINAL.flags.writeable
    with pytest.raises(ValueError):
        R_WB_NOMINAL[0, 0] = 5.0
    pose = BoardPose(R=R_WB_NOMINAL, t=np.zeros(3))
    assert not np.shares_memory(pose.R, R_WB_NOMINAL)


# --------------------------------------------------------------------------- #
# 3. Normal toward the arm
# --------------------------------------------------------------------------- #


def test_normal_toward_arm_nominal_points_back_at_the_arm(geometry):
    pose = BoardPose(R=R_WB_NOMINAL, t=[1.0, 0.0, 0.4])
    np.testing.assert_array_equal(pose.normal_toward_arm, [-1.0, 0.0, 0.0])
    # The arm base is at the world origin, on the -Z_B side of the panel.
    to_arm = np.zeros(3) - pose.center_world(geometry)
    assert np.dot(pose.normal_toward_arm, to_arm) > 0.0
    assert pose.to_board(np.zeros(3))[2] < 0.0


def test_normal_toward_arm_is_minus_z_column_and_unit(geometry):
    rng = np.random.default_rng(3)
    for _ in range(50):
        pose = _random_pose(geometry, rng)
        n = pose.normal_toward_arm
        np.testing.assert_array_equal(n, -pose.R[:, 2])
        assert abs(np.linalg.norm(n) - 1.0) < 1e-12
        # In-plane axes are orthogonal to it.
        assert abs(np.dot(n, pose.R[:, 0])) < 1e-12
        assert abs(np.dot(n, pose.R[:, 1])) < 1e-12


# --------------------------------------------------------------------------- #
# 4. Panel corners
# --------------------------------------------------------------------------- #


def test_corners_world_nominal_pose_known_t(geometry):
    t = np.array([1.0, 0.2, 0.5])
    pose = BoardPose(R=R_WB_NOMINAL, t=t)
    corners = pose.corners_world(geometry)
    assert corners.shape == (4, 3)
    TL, TR, BR, BL = corners
    np.testing.assert_array_equal(TL, t)
    np.testing.assert_allclose(TR - TL, [0.0, -0.400, 0.0], atol=1e-15)  # 0.400 m along -Y_W
    np.testing.assert_allclose(BL - TL, [0.0, 0.0, -0.175], atol=1e-15)  # 0.175 m along -Z_W
    np.testing.assert_allclose(BR, TL + (TR - TL) + (BL - TL), atol=1e-15)
    assert BL[2] < TL[2]  # "bottom" is lower in the world
    np.testing.assert_allclose(
        pose.center_world(geometry), t + [0.0, -0.200, -0.0875], atol=1e-15
    )
    np.testing.assert_allclose(pose.center_world(geometry), corners.mean(axis=0), atol=1e-15)


def test_corners_world_is_a_rigid_rectangle_for_random_poses(geometry):
    rng = np.random.default_rng(4)
    for _ in range(50):
        pose = _random_pose(geometry, rng)
        TL, TR, BR, BL = pose.corners_world(geometry)
        assert abs(np.linalg.norm(TR - TL) - geometry.panel_w) < 1e-12
        assert abs(np.linalg.norm(BL - TL) - geometry.panel_h) < 1e-12
        assert abs(np.dot(TR - TL, BL - TL)) < 1e-12
        # (right) x (down) == into the panel == -normal_toward_arm
        into = np.cross(TR - TL, BL - TL)
        into /= np.linalg.norm(into)
        np.testing.assert_allclose(into, -pose.normal_toward_arm, atol=1e-12)


# --------------------------------------------------------------------------- #
# 5. Marker corners
# --------------------------------------------------------------------------- #


def test_marker_corners_world_are_centred_squares_coplanar_with_panel(geometry):
    rng = np.random.default_rng(5)
    poses = [BoardPose(R=R_WB_NOMINAL, t=[1.0, 0.0, 0.4])] + [_random_pose(geometry, rng) for _ in range(30)]
    s = geometry.marker_size
    for pose in poses:
        mc = marker_corners_world(geometry, pose)
        assert tuple(sorted(mc)) == MARKER_IDS
        for mid, c in mc.items():
            assert c.shape == (4, 3)
            TL, TR, BR, BL = c
            for a, b in ((TL, TR), (TR, BR), (BR, BL), (BL, TL)):
                assert abs(np.linalg.norm(b - a) - s) < 1e-12
            assert abs(np.linalg.norm(BR - TL) - s * np.sqrt(2.0)) < 1e-12
            assert abs(np.linalg.norm(BL - TR) - s * np.sqrt(2.0)) < 1e-12
            # Centred where board_geometry says.
            np.testing.assert_allclose(
                c.mean(axis=0), pose.to_world(geometry.marker_center(mid)), atol=1e-12
            )
            # Coplanar with the panel: board z == 0 for every corner.
            np.testing.assert_allclose(pose.to_board(c)[:, 2], 0.0, atol=1e-12)
            # ArUco order TL,TR,BR,BL as seen from the arm: (TR-TL) x (BL-TL)
            # points into the panel, i.e. away from the arm.
            assert np.dot(np.cross(TR - TL, BL - TL), pose.normal_toward_arm) < 0.0
            # Upright marker: marker x along +X_B, marker "up" along -Y_B.
            np.testing.assert_allclose((TR - TL) / s, pose.R[:, 0], atol=1e-12)
            np.testing.assert_allclose((TL - BL) / s, -pose.R[:, 1], atol=1e-12)


def test_marker_ids_map_to_panel_corners_clockwise_from_tl(geometry):
    pose = BoardPose(R=R_WB_NOMINAL, t=[1.0, 0.0, 0.4])
    panel = pose.corners_world(geometry)  # TL, TR, BR, BL
    mc = marker_corners_world(geometry, pose)
    centres = {mid: c.mean(axis=0) for mid, c in mc.items()}
    for mid, corner in zip(MARKER_IDS, panel):
        d = {k: np.linalg.norm(v - corner) for k, v in centres.items()}
        assert min(d, key=d.get) == mid, f"marker nearest panel corner {mid} is {min(d, key=d.get)}"
    # Specifically: id 0 nearest TL, id 2 nearest BR.
    assert min(centres, key=lambda k: np.linalg.norm(centres[k] - panel[0])) == 0
    assert min(centres, key=lambda k: np.linalg.norm(centres[k] - panel[2])) == 2
    # Board-frame corners: TL has the smallest x and y of each marker.
    for mid, c in marker_corners_board(geometry).items():
        assert np.all(c[0, :2] <= c[:, :2].min(axis=0) + 1e-15)
        assert np.all(c[2, :2] >= c[:, :2].max(axis=0) - 1e-15)
        np.testing.assert_array_equal(c[:, 2], 0.0)


# --------------------------------------------------------------------------- #
# 6. Randomisation
# --------------------------------------------------------------------------- #


def test_board_pose_from_center_composition(geometry):
    rng = np.random.default_rng(6)
    for _ in range(50):
        center = rng.uniform([0.5, -0.5, 0.0], [1.5, 0.5, 1.0])
        yaw, pitch, roll = rng.uniform(-np.pi, np.pi, size=3)
        pose = board_pose_from_center(geometry, center, yaw, pitch, roll)
        R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll) @ R_WB_NOMINAL
        np.testing.assert_allclose(pose.R, R, atol=1e-15)
        np.testing.assert_allclose(pose.center_world(geometry), center, atol=1e-12)
        np.testing.assert_allclose(
            pose.t, center - R @ [geometry.panel_w / 2, geometry.panel_h / 2, 0.0], atol=1e-15
        )
    # Positive yaw is about world +Z (right-handed): the board's right edge
    # swings toward world +X (toward the arm's forward direction).
    pose = board_pose_from_center(geometry, [1.0, 0.0, 0.4], 10 * DEG, 0.0, 0.0)
    assert pose.R[0, 0] > 0.0 and pose.R[2, 0] == 0.0
    # For a pure yaw, X_B = Rz(yaw) @ (0, -1, 0) = (sin yaw, -cos yaw, 0), so
    # the yaw is recovered *with sign* by atan2(R[0,0], -R[1,0]).
    for yaw in (10 * DEG, -20 * DEG, 25 * DEG):
        p = board_pose_from_center(geometry, [1.0, 0.0, 0.4], yaw, 0.0, 0.0)
        np.testing.assert_allclose(p.R[:, 0], [np.sin(yaw), -np.cos(yaw), 0.0], atol=1e-15)
        assert abs(_yaw_of(p) - yaw) < 1e-12
        assert abs(np.arctan2(-p.R[0, 0], -p.R[1, 0]) + yaw) < 1e-12  # the inverted form


def test_sample_board_pose_seeds_0_to_199_in_frame_and_reproducible(geometry):
    home = forward_kinematics(ARM, ARM.q_home)
    ranges = SampleRanges()
    poses = []
    for seed in range(200):
        pose = sample_board_pose(seed, geometry, ARM, CAM)
        _assert_rotation(pose.R)

        # All 16 marker corners in frame at q_home (independent re-check).
        corners = np.concatenate(list(marker_corners_world(geometry, pose).values()), axis=0)
        assert corners.shape == (16, 3)
        uv, z = project_points(CAM, home.R_cam, home.t_cam, corners)
        assert np.all(z > 0.0)
        assert np.all(in_frame(CAM, uv, z))

        # The whole panel sits >= MARGIN_PX from every image edge.
        uv_p, z_p = project_points(CAM, home.R_cam, home.t_cam, pose.corners_world(geometry))
        assert np.all(z_p > 0.0)
        assert _edge_margin(uv_p) >= MARGIN_PX, f"seed {seed}: margin {_edge_margin(uv_p):.1f} px"

        # Reproducible per seed (bit-exact).
        again = sample_board_pose(seed, geometry, ARM, CAM, ranges)
        np.testing.assert_array_equal(again.R, pose.R)
        np.testing.assert_array_equal(again.t, pose.t)

        # With the tuned box no draw is rejected, so the pose is exactly the
        # first draw in the design spec order: centre xyz, yaw, pitch, roll.
        g = pose_rng(seed, geometry.pose_salt)
        c = g.uniform(ranges.box_min, ranges.box_max)
        yaw = g.uniform(-ranges.yaw, ranges.yaw)
        pitch = g.uniform(-ranges.pitch, ranges.pitch)
        roll = g.uniform(-ranges.roll, ranges.roll)
        first = board_pose_from_center(geometry, c, yaw, pitch, roll)
        np.testing.assert_array_equal(first.R, pose.R)
        np.testing.assert_array_equal(first.t, pose.t)
        poses.append(pose)

    # Different seeds give different poses.
    assert not np.allclose(poses[0].t, poses[1].t)
    # Sampled orientations really do vary across the full angle ranges. The
    # yaw is read back from each sampled pose and compared WITH SIGN to the
    # yaw the sampler drew for that seed, so an estimator or a sampler that
    # flipped the sign of yaw (Rz(-yaw)) cannot pass.
    drawn = np.array(
        [_yaw_from_first_draw(seed, geometry.pose_salt, ranges) for seed in range(200)]
    )
    yaws = np.array([_yaw_of(p) for p in poses])
    np.testing.assert_allclose(yaws, drawn, atol=1e-12, rtol=0)
    assert np.abs(drawn).max() <= ranges.yaw
    assert yaws.min() < -20 * DEG and yaws.max() > 20 * DEG
    assert np.any(yaws > 0.0) and np.any(yaws < 0.0)


def test_sample_box_extremes_keep_panel_in_frame_with_margin(geometry):
    """Deterministic worst case: every box corner x every angle extreme."""
    home = forward_kinematics(ARM, ARM.q_home)
    r = SampleRanges()
    worst = np.inf
    for center in itertools.product(*zip(r.box_min, r.box_max)):
        for yaw, pitch, roll in itertools.product((-r.yaw, r.yaw), (-r.pitch, r.pitch), (-r.roll, r.roll)):
            pose = board_pose_from_center(geometry, np.array(center), yaw, pitch, roll)
            uv, z = project_points(CAM, home.R_cam, home.t_cam, pose.corners_world(geometry))
            assert np.all(z > 0.0)
            worst = min(worst, _edge_margin(uv))
    assert worst >= MARGIN_PX, f"worst-case panel margin {worst:.1f} px"


def test_sample_box_keeps_panel_centre_at_sane_reach(geometry):
    """Guard against absurd boxes: centre 0.55-1.2 m from the shoulder."""
    shoulder = forward_kinematics(ARM, ARM.q_home).p_shoulder
    r = SampleRanges()
    for center in itertools.product(*zip(r.box_min, r.box_max)):
        d = np.linalg.norm(np.array(center) - shoulder)
        assert SHOULDER_DIST[0] <= d <= SHOULDER_DIST[1], f"box corner {center}: {d:.3f} m"
    for seed in range(50):
        pose = sample_board_pose(seed, geometry, ARM, CAM)
        d = np.linalg.norm(pose.center_world(geometry) - shoulder)
        assert SHOULDER_DIST[0] <= d <= SHOULDER_DIST[1]
    # And q_home is strictly inside the joint limits (not parked on a stop).
    assert np.all(ARM.q_home > ARM.q_min) and np.all(ARM.q_home < ARM.q_max)


def test_pose_is_not_derivable_from_the_public_seed(geometry):
    """The episode seed is public (it is a ROS parameter and an /rosout line);
    the pose it produces is not derivable from it. Running the sampler on a
    plain ``default_rng(seed)`` -- no salt -- must land somewhere else
    entirely, and so must any other salt."""
    ranges = SampleRanges()
    for seed in range(25):
        real = sample_board_pose(seed, geometry, ARM, CAM, ranges)

        # (a) the unsalted stream, i.e. everything a member can compute.
        guess = sample_board_pose(
            None, geometry, ARM, CAM, ranges, rng=np.random.default_rng(seed)
        )
        gap = np.linalg.norm(real.corners_world(geometry) - guess.corners_world(geometry), axis=1)
        assert gap.min() > 0.01, f"seed {seed}: closest corner only {gap.min() * 1e3:.1f} mm off"

        # (b) a different salt is a different stream again.
        other_salt = int(DEFAULT_GEOMETRY["pose_salt"])
        if other_salt != geometry.pose_salt:
            alt = sample_board_pose(
                None, geometry, ARM, CAM, ranges, rng=pose_rng(seed, other_salt)
            )
            gap2 = np.linalg.norm(
                real.corners_world(geometry) - alt.corners_world(geometry), axis=1
            )
            assert gap2.min() > 0.01, f"seed {seed}: other salt lands {gap2.min() * 1e3:.1f} mm off"

        # (c) but the same (seed, salt) pair IS bit-exactly reproducible.
        again = sample_board_pose(
            None, geometry, ARM, CAM, ranges, rng=pose_rng(seed, geometry.pose_salt)
        )
        np.testing.assert_array_equal(again.R, real.R)
        np.testing.assert_array_equal(again.t, real.t)

    # A different salt on the same seed is a different pose, too.
    other = sample_board_pose(0, geometry, ARM, CAM, ranges, rng=pose_rng(0, 1))
    base = sample_board_pose(0, geometry, ARM, CAM, ranges)
    assert not np.allclose(other.t, base.t)


def test_pose_salt_is_required_and_never_public(geometry, geometry_dict):
    """``pose_salt`` is required on load and absent from ``public_dict``."""
    assert "pose_salt" not in geometry.public_dict()
    assert "pose_salt" not in PUBLIC_FIELDS
    with pytest.raises(ValueError, match=r"missing keys \['pose_salt'\]"):
        BoardGeometry.from_dict({k: v for k, v in geometry_dict.items() if k != "pose_salt"})
    for bad in (1.5, "7", None, True):
        with pytest.raises(ValueError, match="pose_salt must be an integer"):
            BoardGeometry(**{**_geom_kwargs(), "pose_salt": bad})
    for bad_seed in (-1, 1.5, "3", True):
        with pytest.raises(ValueError, match="seed must be a non-negative integer"):
            pose_rng(bad_seed, geometry.pose_salt)
    with pytest.raises(TypeError, match="must be a numpy.random.Generator"):
        sample_board_pose(None, geometry, ARM, CAM, rng=7)
    with pytest.raises(ValueError, match="either a seed or an explicit rng"):
        sample_board_pose(None, geometry, ARM, CAM)


def test_sample_board_pose_raises_naming_the_failing_constraint(geometry):
    behind = SampleRanges(box_min=[-1.0, 0.0, 0.4], box_max=[-1.0, 0.0, 0.4])
    with pytest.raises(RuntimeError, match="behind the camera"):
        sample_board_pose(0, geometry, ARM, CAM, behind)
    far_left = SampleRanges(box_min=[1.0, 2.0, 0.4], box_max=[1.0, 2.0, 0.4])
    with pytest.raises(RuntimeError, match="left of the image"):
        sample_board_pose(0, geometry, ARM, CAM, far_left, max_draws=5)
    too_low = SampleRanges(box_min=[1.0, 0.0, -0.5], box_max=[1.0, 0.0, -0.5])
    with pytest.raises(RuntimeError, match="below the image"):
        sample_board_pose(0, geometry, ARM, CAM, too_low, max_draws=5)
    with pytest.raises(RuntimeError, match="1000 consecutive draws"):
        sample_board_pose(0, geometry, ARM, CAM, behind)


def test_sample_board_pose_rejects_nonpositive_max_draws(geometry):
    """max_draws <= 0 is an argument error, not a bare max() ValueError."""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_draws must be >= 1"):
            sample_board_pose(0, geometry, ARM, CAM, max_draws=bad)
    # The smallest legal budget still works and still raises the diagnostic.
    pose = sample_board_pose(0, geometry, ARM, CAM, max_draws=1)
    _assert_rotation(pose.R)
    behind = SampleRanges(box_min=[-1.0, 0.0, 0.4], box_max=[-1.0, 0.0, 0.4])
    with pytest.raises(RuntimeError, match="1 consecutive draws"):
        sample_board_pose(0, geometry, ARM, CAM, behind, max_draws=1)


def test_sample_ranges_validation():
    with pytest.raises(ValueError):
        SampleRanges(box_min=[1.0, 0.0, 0.5], box_max=[1.0, 0.0, 0.4])
    with pytest.raises(ValueError):
        SampleRanges(yaw=-1.0)
    r = SampleRanges()
    assert r.box_min.dtype == np.float64 and r.box_max.shape == (3,)
    # Non-finite ranges are a configuration bug (the design spec s.6) and are named
    # at construction instead of surfacing as numpy OverflowError in uniform().
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="box_min must be finite"):
            SampleRanges(box_min=[bad, -0.1, 0.3], box_max=[1.1, 0.1, 0.4])
        with pytest.raises(ValueError, match="box_max must be finite"):
            SampleRanges(box_min=[1.0, -0.1, 0.3], box_max=[1.1, 0.1, bad])
        for name in ("yaw", "pitch", "roll"):
            with pytest.raises(ValueError, match=f"{name} half-range must be a finite"):
                SampleRanges(**{name: bad})


def test_sample_ranges_copies_inputs_and_is_read_only(geometry):
    bm = np.array([1.0, -0.1, 0.3])
    bx = np.array([1.1, 0.1, 0.4])
    r = SampleRanges(box_min=bm, box_max=bx)
    assert not np.shares_memory(r.box_min, bm) and not np.shares_memory(r.box_max, bx)
    bm[0] = 99.0
    bx[2] = -99.0
    np.testing.assert_array_equal(r.box_min, [1.0, -0.1, 0.3])
    np.testing.assert_array_equal(r.box_max, [1.1, 0.1, 0.4])
    # Stored arrays (including the shared default instance in the
    # sample_board_pose signature) cannot be edited in place.
    import inspect

    default = inspect.signature(sample_board_pose).parameters["ranges"].default
    for ranges in (r, default, SampleRanges()):
        assert not ranges.box_min.flags.writeable and not ranges.box_max.flags.writeable
        with pytest.raises(ValueError):
            ranges.box_min[0] = 0.0
        with pytest.raises(ValueError):
            ranges.box_max[:] = 0.0
    np.testing.assert_array_equal(default.box_min, SampleRanges().box_min)
    np.testing.assert_array_equal(default.box_max, SampleRanges().box_max)
    # Read-only ranges still drive the sampler.
    _assert_rotation(sample_board_pose(0, geometry, ARM, CAM, r).R)


# --------------------------------------------------------------------------- #
# Geometry loading and the public subset
# --------------------------------------------------------------------------- #


def _write_yaml(path, geom: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(geom, fh, sort_keys=False)


def test_geometry_load_matches_yaml(tmp_path, geometry_dict, geometry):
    """load() is a thin wrapper over from_dict(): the same mapping through a
    YAML file and through the dict gives field-for-field the same object."""
    path = tmp_path / "board_geometry.yaml"
    _write_yaml(path, geometry_dict)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    g = BoardGeometry.load(path)
    assert g == geometry == BoardGeometry.from_dict(raw)
    assert g.panel_w == raw["panel_w"] == 0.400
    assert g.panel_h == raw["panel_h"] == 0.175
    assert g.marker_size == raw["marker_size"] == 0.020
    assert g.marker_dict == "DICT_4X4_50"
    for mid in MARKER_IDS:
        assert g.marker_centers[mid] == tuple(float(v) for v in raw["marker_centers"][mid])
    assert g.kb_origin == tuple(raw["kb_origin"])
    assert g.key_area_origin == tuple(raw["key_area_origin"])
    assert g.key_pitch == raw["key_pitch"]
    assert g.key_inset == raw["key_inset"]
    # Extra keys are ignored, missing keys are named, non-mappings rejected.
    assert BoardGeometry.from_dict({**geometry_dict, "comment": "x"}) == g
    with pytest.raises(ValueError, match=r"missing keys \['kb_w'\]"):
        BoardGeometry.from_dict({k: v for k, v in geometry_dict.items() if k != "kb_w"})
    with pytest.raises(ValueError, match="expected a mapping"):
        BoardGeometry.from_dict([1, 2, 3])
    path.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a mapping at top level"):
        BoardGeometry.load(path)


@pytest.mark.private
def test_private_yaml_is_what_the_fixtures_serve(private_dir, geometry, geometry_is_private):
    """With private/ present the fixtures serve it, not ``DEFAULT_GEOMETRY``."""
    assert geometry_is_private
    real = BoardGeometry.load(private_dir / "board_geometry.yaml")
    assert real == geometry
    assert real.kb_origin != tuple(DEFAULT_GEOMETRY["kb_origin"])


def test_public_dict_exposes_only_the_urc_public_subset(geometry):
    pub = geometry.public_dict()
    assert set(pub) == set(PUBLIC_FIELDS) == {
        "panel_w", "panel_h", "marker_size", "marker_dict", "marker_centers"
    }
    for key in pub:
        assert not key.startswith("kb") and not key.startswith("key")
    assert pub["panel_w"] == geometry.panel_w and pub["panel_h"] == geometry.panel_h
    assert pub["marker_size"] == geometry.marker_size
    assert pub["marker_dict"] == geometry.marker_dict
    assert set(pub["marker_centers"]) == set(MARKER_IDS)
    for mid in MARKER_IDS:
        assert pub["marker_centers"][mid] == list(geometry.marker_centers[mid])
        assert all(isinstance(v, float) for v in pub["marker_centers"][mid])
    # Serialisable as-is, and nothing about the keyboard leaks through.
    text = yaml.safe_dump(pub)
    assert "kb_" not in text and "key_" not in text
    assert yaml.safe_load(text) == pub


GOOD_MARKERS = {0: [0.016, 0.016], 1: [0.384, 0.016], 2: [0.384, 0.159], 3: [0.016, 0.159]}

# The default placement, used as the base for the validation tests below
# (any valid placement would do).
KX, KY = DEFAULT_GEOMETRY["kb_origin"]
GX, GY = DEFAULT_GEOMETRY["key_area_origin"]
PITCH = DEFAULT_GEOMETRY["key_pitch"]
GRID_W, GRID_H = TKL_WIDTH_U * PITCH, TKL_HEIGHT_U * PITCH


def _geom_kwargs(**over) -> dict:
    """The default geometry as kwargs, with overrides."""
    base = {**DEFAULT_GEOMETRY, "marker_centers": GOOD_MARKERS}
    base.update(over)
    return base


def _assert_placement_consistent(g: BoardGeometry) -> None:
    """The design spec s.4/s.5 containment: plate inside panel, grid inside plate,
    every generated key inside the plate."""
    kx, ky = g.kb_origin
    gx, gy = g.key_area_origin
    assert 0.0 <= kx and 0.0 <= ky
    assert kx + g.kb_w <= g.panel_w and ky + g.kb_h <= g.panel_h
    grid_w, grid_h = TKL_WIDTH_U * g.key_pitch, TKL_HEIGHT_U * g.key_pitch
    assert kx <= gx and ky <= gy
    assert gx + grid_w <= kx + g.kb_w + 1e-12 and gy + grid_h <= ky + g.kb_h + 1e-12
    km = generate_tkl(g)
    assert len(km) == 87
    for k in km:
        assert kx <= k.x0 < k.x1 <= kx + g.kb_w + 1e-12, k.name
        assert ky <= k.y0 < k.y1 <= ky + g.kb_h + 1e-12, k.name


def test_geometry_fixture_is_self_consistent(geometry):
    """Whatever placement is in use is physically consistent; the suite asserts
    that consistency instead of pinning coordinates."""
    _assert_placement_consistent(geometry)
    # Public marker layout per the design spec s.4: 16 mm in from each panel edge.
    expected = {
        0: (MARKER_EDGE_M, MARKER_EDGE_M),
        1: (geometry.panel_w - MARKER_EDGE_M, MARKER_EDGE_M),
        2: (geometry.panel_w - MARKER_EDGE_M, geometry.panel_h - MARKER_EDGE_M),
        3: (MARKER_EDGE_M, geometry.panel_h - MARKER_EDGE_M),
    }
    assert sorted(geometry.marker_centers) == list(MARKER_IDS)
    for mid in MARKER_IDS:
        np.testing.assert_allclose(geometry.marker_centers[mid], expected[mid], atol=1e-12, rtol=0)


def test_default_geometry_is_valid_public_and_distinct_from_the_configured_one(
    geometry_dict, geometry_is_private
):
    """DEFAULT_GEOMETRY must (a) be a legal BoardGeometry with the grid inside
    the plate inside the panel, (b) carry exactly the URC-public values, and
    (c) differ from the configured placement in every placement field.
    (c) is checked whenever a configured YAML is available."""
    default_geom = BoardGeometry.from_dict(DEFAULT_GEOMETRY)
    _assert_placement_consistent(default_geom)
    assert default_geom.public_dict() == BoardGeometry.from_dict(geometry_dict).public_dict()
    assert set(DEFAULT_GEOMETRY) == set(BoardGeometry.__dataclass_fields__)
    if geometry_is_private:
        configured = BoardGeometry.from_dict(geometry_dict)
        assert default_geom.kb_origin != configured.kb_origin
        assert default_geom.key_area_origin != configured.key_area_origin
        assert any(
            getattr(default_geom, f) != getattr(configured, f) for f in PLACEMENT_FIELDS
        )
        # Every placement field, not just the plate origin.
        for f in ("kb_w", "kb_h", "key_inset", "pose_salt"):
            assert getattr(default_geom, f) != getattr(configured, f), f
        # ... including the bezel offset (key_area_origin - kb_origin).
        bezel = lambda g: (  # noqa: E731
            round(g.key_area_origin[0] - g.kb_origin[0], 9),
            round(g.key_area_origin[1] - g.kb_origin[1], 9),
        )
        assert bezel(default_geom) != bezel(configured)


def test_geometry_accepts_marker_flush_with_panel_edge():
    """Containment has a float tolerance: 0.175 - 0.01 rounds to 0.164999...,
    so a marker exactly flush with the edge must not be rejected by rounding."""
    flush = {0: [0.01, 0.01], 1: [0.39, 0.01], 2: [0.39, 0.165], 3: [0.01, 0.165]}
    g = BoardGeometry(**_geom_kwargs(marker_centers=flush))
    assert g.marker_centers[2] == (0.39, 0.165)
    # Literal decimals for (panel_h, flush cy) where panel_h - 0.01 rounds
    # below cy in binary (0.15 - 0.01 = 0.13999..., 0.18 - 0.01 = 0.16999...).
    for ph, cy in ((0.15, 0.14), (0.175, 0.165), (0.18, 0.17)):
        assert ph - 0.01 < cy  # the rounding this test guards against
        # A plate that fits the shortest panel tried here: this test is about
        # marker rounding, not about the plate size.
        g = BoardGeometry(
            **_geom_kwargs(
                panel_h=ph,
                marker_centers={**flush, 2: [0.39, cy], 3: [0.01, cy]},
                kb_origin=[0.024, 0.002],
                kb_h=0.120,
                key_area_origin=[0.025, 0.0025],
            )
        )
        assert g.marker_centers[2] == (0.39, cy)
    # Anything physically off the panel is still rejected.
    for bad in ({**flush, 2: [0.39, 0.1651]}, {**flush, 0: [0.005, 0.01]}, {**flush, 1: [0.3901, 0.01]}):
        with pytest.raises(ValueError, match="does not fit inside the panel"):
            BoardGeometry(**_geom_kwargs(marker_centers=bad))


def test_geometry_rejects_bad_keyboard_placement_and_non_finite_fields():
    # The default numbers pass (also flush with the plate edge is fine).
    BoardGeometry(**_geom_kwargs())
    BoardGeometry(**_geom_kwargs(kb_origin=[0.0, 0.0], key_area_origin=[0.0, 0.0]))
    # Keyboard plate hanging off the panel.
    with pytest.raises(ValueError, match="keyboard plate .* does not fit inside the panel"):
        BoardGeometry(**_geom_kwargs(kb_origin=[0.3, 0.1], key_area_origin=[0.31, 0.11]))
    with pytest.raises(ValueError, match="keyboard plate .* does not fit inside the panel"):
        BoardGeometry(**_geom_kwargs(kb_origin=[-0.001, KY]))
    with pytest.raises(ValueError, match="keyboard plate .* does not fit inside the panel"):
        BoardGeometry(**_geom_kwargs(kb_h=DEFAULT_GEOMETRY["panel_h"] - KY + 0.001))  # 1 mm too tall
    # Key grid (18.25u x 6.25u) hanging off the plate.
    with pytest.raises(ValueError, match="key grid .* does not fit inside the keyboard plate"):
        BoardGeometry(**_geom_kwargs(key_area_origin=[5.0, 5.0]))
    with pytest.raises(ValueError, match="key grid .* does not fit inside the keyboard plate"):
        # grid bottom 1 mm past the plate bottom
        BoardGeometry(**_geom_kwargs(key_area_origin=[GX, KY + DEFAULT_GEOMETRY["kb_h"] - GRID_H + 0.001]))
    with pytest.raises(ValueError, match="key grid .* does not fit inside the keyboard plate"):
        BoardGeometry(**_geom_kwargs(key_area_origin=[GX, KY - 0.0001]))  # above the plate
    with pytest.raises(ValueError, match="key grid .* does not fit inside the keyboard plate"):
        BoardGeometry(**_geom_kwargs(key_pitch=0.0210))  # 18.25u no longer fits
    # key_inset >= key_pitch / 2 inverts every registration rectangle (s.5).
    with pytest.raises(ValueError, match="key_inset must be < key_pitch / 2"):
        BoardGeometry(**_geom_kwargs(key_inset=0.05))
    with pytest.raises(ValueError, match="key_inset must be < key_pitch / 2"):
        BoardGeometry(**_geom_kwargs(key_inset=PITCH / 2))
    assert BoardGeometry(**_geom_kwargs(key_inset=0.0)).key_inset == 0.0
    # NaN / inf anywhere is rejected by name.
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="kb_origin must be finite"):
            BoardGeometry(**_geom_kwargs(kb_origin=[bad, KY]))
        with pytest.raises(ValueError, match="key_area_origin must be finite"):
            BoardGeometry(**_geom_kwargs(key_area_origin=[GX, bad]))
        with pytest.raises(ValueError, match=r"marker_centers\[2\] must be finite"):
            BoardGeometry(**_geom_kwargs(marker_centers={**GOOD_MARKERS, 2: [0.384, bad]}))
        for name in ("panel_w", "panel_h", "marker_size", "kb_w", "kb_h", "key_pitch", "key_inset"):
            with pytest.raises(ValueError, match=f"{name} must be finite"):
                BoardGeometry(**_geom_kwargs(**{name: bad}))
    # Duplicate ids after int normalisation ("0" and 0) do not silently collapse.
    with pytest.raises(ValueError, match="duplicate ids"):
        BoardGeometry(**_geom_kwargs(marker_centers={**GOOD_MARKERS, "0": [0.2, 0.1]}))


def test_geometry_validation_rejects_bad_markers(geometry):
    kw = {k: v for k, v in _geom_kwargs().items() if k != "marker_centers"}
    with pytest.raises(ValueError):
        BoardGeometry(marker_centers={0: [0.016, 0.016]}, **kw)  # missing ids
    with pytest.raises(ValueError):
        BoardGeometry(
            marker_centers={0: [0.005, 0.016], 1: [0.384, 0.016], 2: [0.384, 0.159], 3: [0.016, 0.159]},
            **kw,
        )  # marker 0 hangs off the panel edge
    with pytest.raises(ValueError):
        BoardGeometry(marker_centers=geometry.marker_centers, **{**kw, "panel_w": -0.4})
    # String ids (as some YAML writers emit) are normalised to ints.
    g = BoardGeometry(
        marker_centers={"0": [0.016, 0.016], "1": [0.384, 0.016], "2": [0.384, 0.159], "3": [0.016, 0.159]},
        **kw,
    )
    assert tuple(sorted(g.marker_centers)) == MARKER_IDS


def test_geometry_works_as_generate_tkl_input_shape(geometry, geometry_dict):
    """keymap.generate_tkl reads geometry by attribute or by key; the fields it
    needs exist and both spellings give the same map."""
    assert isinstance(geometry.key_pitch, float)
    assert len(geometry.key_area_origin) == 2
    assert isinstance(geometry.key_inset, float)
    assert geometry.kb_w < geometry.panel_w and geometry.kb_h < geometry.panel_h
    assert generate_tkl(geometry) == generate_tkl(geometry_dict)
