"""Tests for core/kinematics.py against the design spec section 2.

All randomness comes from seeded numpy Generators; tolerances are tight
because every function here is closed-form.
"""

import numpy as np
import pytest

from autotype_sim.core.config import DEG, NJ, ArmConfig, CameraConfig
from autotype_sim.core.kinematics import (
    ArmPose,
    aim_inverse,
    forward_kinematics,
    in_frame,
    project_points,
    rot_x,
    rot_y,
    rot_z,
)

ARM = ArmConfig()
CAM = CameraConfig()
L0, L1, L2 = ARM.base_height, ARM.upper_arm, ARM.forearm
EX, EY, EZ = np.eye(3)


def _random_in_limit_q(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(ARM.q_min, ARM.q_max, size=(n, NJ))


def _assert_rotation(R: np.ndarray, tol: float = 1e-12) -> None:
    assert R.shape == (3, 3)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=tol, rtol=0)
    assert abs(np.linalg.det(R) - 1.0) < tol


# --------------------------------------------------------------------------- #
# Elementary rotations
# --------------------------------------------------------------------------- #


def test_rot_matrices_are_right_handed_active_rotations():
    tol = 1e-12
    np.testing.assert_allclose(rot_x(90 * DEG) @ EY, EZ, atol=tol)
    np.testing.assert_allclose(rot_y(90 * DEG) @ EZ, EX, atol=tol)
    np.testing.assert_allclose(rot_z(90 * DEG) @ EX, EY, atol=tol)
    rng = np.random.default_rng(1)
    for a in rng.uniform(-np.pi, np.pi, size=50):
        for R in (rot_x(a), rot_y(a), rot_z(a)):
            _assert_rotation(R)
    np.testing.assert_allclose(rot_x(0.0), np.eye(3), atol=0)
    np.testing.assert_allclose(rot_y(0.0), np.eye(3), atol=0)
    np.testing.assert_allclose(rot_z(0.0), np.eye(3), atol=0)


# --------------------------------------------------------------------------- #
# Forward kinematics: closed-form poses
# --------------------------------------------------------------------------- #


def test_neutral_pose_points_forward():
    pose = forward_kinematics(ARM, np.zeros(NJ))
    assert isinstance(pose, ArmPose)
    tol = 1e-12
    np.testing.assert_allclose(pose.p_shoulder, [0.0, 0.0, L0], atol=tol)
    np.testing.assert_allclose(pose.p_elbow, [L1, 0.0, L0], atol=tol)
    np.testing.assert_allclose(pose.p_head, [L1 + L2, 0.0, L0], atol=tol)
    np.testing.assert_allclose(pose.R_head[:, 0], EX, atol=tol)
    np.testing.assert_allclose(pose.R_head[:, 1], EY, atol=tol)
    np.testing.assert_allclose(pose.R_head[:, 2], EZ, atol=tol)
    np.testing.assert_allclose(pose.aim, EX, atol=tol)
    np.testing.assert_allclose(pose.t_cam, pose.p_head, atol=0)


def test_positive_yaw_turns_toward_world_plus_y():
    q = np.array([90 * DEG, 0.0, 0.0, 0.0, 0.0])
    pose = forward_kinematics(ARM, q)
    tol = 1e-12
    np.testing.assert_allclose(pose.p_head, [0.0, L1 + L2, L0], atol=tol)
    np.testing.assert_allclose(pose.R_head[:, 0], EY, atol=tol)
    np.testing.assert_allclose(pose.R_head[:, 1], -EX, atol=tol)
    np.testing.assert_allclose(pose.R_head[:, 2], EZ, atol=tol)


def test_positive_shoulder_pitch_raises_the_arm():
    q = np.array([0.0, 90 * DEG, 0.0, 0.0, 0.0])
    pose = forward_kinematics(ARM, q)
    tol = 1e-12
    np.testing.assert_allclose(pose.p_elbow, [0.0, 0.0, L0 + L1], atol=tol)
    np.testing.assert_allclose(pose.p_head, [0.0, 0.0, L0 + L1 + L2], atol=tol)
    np.testing.assert_allclose(pose.R_head[:, 0], EZ, atol=tol)
    np.testing.assert_allclose(pose.R_head[:, 2], -EX, atol=tol)


def test_elbow_pitch_is_relative_to_upper_arm():
    # Upper arm horizontal, forearm folded straight down: phi = -90 deg.
    q = np.array([0.0, 0.0, -90 * DEG, 0.0, 0.0])
    pose = forward_kinematics(ARM, q)
    tol = 1e-12
    np.testing.assert_allclose(pose.p_elbow, [L1, 0.0, L0], atol=tol)
    np.testing.assert_allclose(pose.p_head, [L1, 0.0, L0 - L2], atol=tol)
    # Shoulder up 45, elbow down 45: forearm horizontal again.
    q = np.array([0.0, 45 * DEG, -45 * DEG, 0.0, 0.0])
    pose = forward_kinematics(ARM, q)
    np.testing.assert_allclose(pose.R_head[:, 0], EX, atol=tol)


def test_forward_kinematics_rejects_wrong_shape():
    with pytest.raises(ValueError):
        forward_kinematics(ARM, np.zeros(NJ - 1))


# --------------------------------------------------------------------------- #
# Forward kinematics: frame invariants over random in-limit q
# --------------------------------------------------------------------------- #


def test_R_head_orthonormal_right_handed_over_random_q():
    rng = np.random.default_rng(42)
    tol = 1e-12
    for q in _random_in_limit_q(rng, 1000):
        pose = forward_kinematics(ARM, q)
        R = pose.R_head
        _assert_rotation(R, tol)
        x_h, y_h, z_h = R[:, 0], R[:, 1], R[:, 2]
        np.testing.assert_allclose(np.cross(x_h, y_h), z_h, atol=tol, rtol=0)
        # Y_H is horizontal (no roll about the forearm axis).
        assert abs(y_h[2]) < tol
        # Link lengths are preserved.
        assert abs(np.linalg.norm(pose.p_elbow - pose.p_shoulder) - L1) < tol
        assert abs(np.linalg.norm(pose.p_head - pose.p_elbow) - L2) < tol
        # aim is a unit vector.
        assert abs(np.linalg.norm(pose.aim) - 1.0) < tol


def test_R_head_equals_rz_yaw_times_ry_minus_pitch():
    rng = np.random.default_rng(7)
    for q in _random_in_limit_q(rng, 200):
        pose = forward_kinematics(ARM, q)
        expected = rot_z(q[0]) @ rot_y(-(q[1] + q[2]))
        np.testing.assert_allclose(pose.R_head, expected, atol=1e-12, rtol=0)


def test_aim_at_zero_pan_tilt_is_x_head():
    rng = np.random.default_rng(3)
    for q in _random_in_limit_q(rng, 100):
        q[3] = q[4] = 0.0
        pose = forward_kinematics(ARM, q)
        np.testing.assert_allclose(pose.aim, pose.R_head[:, 0], atol=1e-12)


# --------------------------------------------------------------------------- #
# Aim inverse
# --------------------------------------------------------------------------- #


def test_aim_inverse_round_trip_random_targets():
    rng = np.random.default_rng(2024)
    for q in _random_in_limit_q(rng, 1000):
        pose = forward_kinematics(ARM, q)
        # Random target within 1 m of the head, at least 5 cm away.
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        radius = rng.uniform(0.05, 1.0)
        target = pose.p_head + radius * direction

        pan, tilt, rng_out = aim_inverse(pose.R_head, pose.p_head, target)
        assert isinstance(pan, float) and isinstance(tilt, float)
        assert isinstance(rng_out, float)
        assert abs(rng_out - radius) < 1e-12
        assert -np.pi <= pan <= np.pi
        assert -np.pi / 2 <= tilt <= np.pi / 2

        q2 = q.copy()
        q2[3], q2[4] = pan, tilt
        pose2 = forward_kinematics(ARM, q2)
        np.testing.assert_allclose(pose2.p_head, pose.p_head, atol=0)
        recovered = pose2.p_head + rng_out * pose2.aim
        np.testing.assert_allclose(recovered, target, atol=1e-9, rtol=0)


def test_aim_inverse_recovers_known_pan_tilt_range():
    rng = np.random.default_rng(99)
    for q in _random_in_limit_q(rng, 500):
        pan_true = rng.uniform(-170 * DEG, 170 * DEG)
        tilt_true = rng.uniform(-80 * DEG, 80 * DEG)
        range_true = rng.uniform(0.05, 1.0)
        q[3], q[4] = pan_true, tilt_true
        pose = forward_kinematics(ARM, q)
        target = pose.p_head + range_true * pose.aim

        pan, tilt, rng_out = aim_inverse(pose.R_head, pose.p_head, target)
        assert abs(pan - pan_true) < 1e-9
        assert abs(tilt - tilt_true) < 1e-9
        assert abs(rng_out - range_true) < 1e-12


def test_aim_inverse_sign_conventions():
    pose = forward_kinematics(ARM, np.zeros(NJ))
    # Target to the arm's left (+Y_H) -> positive pan.
    pan, tilt, _ = aim_inverse(pose.R_head, pose.p_head, pose.p_head + [1.0, 1.0, 0.0])
    assert abs(pan - 45 * DEG) < 1e-12 and abs(tilt) < 1e-12
    # Target above (+Z_H) -> positive tilt.
    pan, tilt, _ = aim_inverse(pose.R_head, pose.p_head, pose.p_head + [1.0, 0.0, 1.0])
    assert abs(pan) < 1e-12 and abs(tilt - 45 * DEG) < 1e-12


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #


def test_camera_reticle_and_image_axis_directions():
    rng = np.random.default_rng(11)
    tol = 1e-9
    for q in _random_in_limit_q(rng, 200):
        pose = forward_kinematics(ARM, q)
        x_h, y_h, z_h = pose.R_head.T
        pts = np.stack(
            [
                pose.p_head + 2.0 * x_h,  # on the optical axis
                pose.p_head + 2.0 * x_h - 0.1 * y_h,  # to the arm's right
                pose.p_head + 2.0 * x_h - 0.1 * z_h,  # down
                pose.p_head - x_h,  # behind the camera
            ]
        )
        uv, z = project_points(CAM, pose.R_cam, pose.t_cam, pts)
        assert uv.shape == (4, 2) and z.shape == (4,)

        np.testing.assert_allclose(uv[0], [CAM.cx, CAM.cy], atol=tol, rtol=0)
        assert abs(z[0] - 2.0) < tol

        # Right in the world -> larger u; exactly fx * 0.1 / 2 px right.
        assert uv[1, 0] > CAM.cx
        assert abs(uv[1, 0] - (CAM.cx + CAM.fx * 0.1 / 2.0)) < tol
        assert abs(uv[1, 1] - CAM.cy) < tol

        # Down in the world -> larger v; exactly fy * 0.1 / 2 px down.
        assert uv[2, 1] > CAM.cy
        assert abs(uv[2, 1] - (CAM.cy + CAM.fy * 0.1 / 2.0)) < tol
        assert abs(uv[2, 0] - CAM.cx) < tol

        # Behind: z < 0, NaN pixel, never in frame.
        assert z[3] < 0 and abs(z[3] + 1.0) < tol
        assert np.all(np.isnan(uv[3]))

        mask = in_frame(CAM, uv, z)
        assert mask.dtype == bool and mask.shape == (4,)
        assert mask.tolist() == [True, True, True, False]


def test_camera_does_not_move_with_pan_tilt():
    rng = np.random.default_rng(5)
    for q in _random_in_limit_q(rng, 100):
        base = forward_kinematics(ARM, q)
        q2 = q.copy()
        q2[3], q2[4] = rng.uniform(ARM.q_min[3:], ARM.q_max[3:])
        moved = forward_kinematics(ARM, q2)
        np.testing.assert_allclose(moved.R_cam, base.R_cam, atol=0)
        np.testing.assert_allclose(moved.t_cam, base.t_cam, atol=0)
        np.testing.assert_allclose(moved.R_head, base.R_head, atol=0)


def test_R_cam_columns_and_orthonormality():
    rng = np.random.default_rng(13)
    tol = 1e-12
    for q in _random_in_limit_q(rng, 1000):
        pose = forward_kinematics(ARM, q)
        x_h, y_h, z_h = pose.R_head.T
        _assert_rotation(pose.R_cam, tol)
        np.testing.assert_allclose(pose.R_cam[:, 0], -y_h, atol=0)
        np.testing.assert_allclose(pose.R_cam[:, 1], -z_h, atol=0)
        np.testing.assert_allclose(pose.R_cam[:, 2], x_h, atol=0)
        np.testing.assert_allclose(pose.t_cam, pose.p_head, atol=0)


def test_project_points_matches_explicit_formula_and_K():
    rng = np.random.default_rng(17)
    for q in _random_in_limit_q(rng, 50):
        pose = forward_kinematics(ARM, q)
        pts = pose.p_head + rng.uniform(-1.0, 1.0, size=(64, 3))
        uv, z = project_points(CAM, pose.R_cam, pose.t_cam, pts)
        for P, (u, v), zi in zip(pts, uv, z):
            p_c = pose.R_cam.T @ (P - pose.t_cam)
            assert abs(zi - p_c[2]) < 1e-12
            if p_c[2] > 0:
                # Points a few mm in front of the lens project to |u| ~ 1e5 px;
                # allow float64 relative error there, keep 1e-9 px on-screen.
                homog = CAM.K @ p_c
                u_ref, v_ref = homog[0] / homog[2], homog[1] / homog[2]
                assert abs(u - u_ref) < 1e-9 + 1e-12 * abs(u_ref)
                assert abs(v - v_ref) < 1e-9 + 1e-12 * abs(v_ref)
            else:
                assert np.isnan(u) and np.isnan(v)


def test_project_points_accepts_single_point_and_rejects_bad_shape():
    pose = forward_kinematics(ARM, ARM.q_home)
    uv, z = project_points(CAM, pose.R_cam, pose.t_cam, pose.p_head + pose.R_head[:, 0])
    assert uv.shape == (1, 2) and z.shape == (1,)
    np.testing.assert_allclose(uv[0], [CAM.cx, CAM.cy], atol=1e-9)
    uv, z = project_points(CAM, pose.R_cam, pose.t_cam, np.zeros((0, 3)))
    assert uv.shape == (0, 2) and z.shape == (0,)
    with pytest.raises(ValueError):
        project_points(CAM, pose.R_cam, pose.t_cam, np.zeros((3, 4)))


def test_in_frame_is_inclusive_of_pixel_edges():
    W, H = CAM.width, CAM.height
    uv = np.array(
        [
            [0.0, 0.0],
            [W - 1.0, H - 1.0],
            [-1e-9, 10.0],
            [10.0, H - 1.0 + 1e-9],
            [W / 2.0, H / 2.0],
            [W / 2.0, H / 2.0],
            [np.nan, np.nan],
        ]
    )
    z = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    expected = [True, True, False, False, True, False, False]
    assert in_frame(CAM, uv, z).tolist() == expected
