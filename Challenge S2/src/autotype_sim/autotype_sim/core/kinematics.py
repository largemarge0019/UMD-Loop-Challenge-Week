"""Forward kinematics, stylus aim and pinhole projection for the 5-DOF arm.

Conventions:

  q = [base_yaw, shoulder_pitch, elbow_pitch, head_pan, head_tilt]  (radians)
  Rotation matrices are world-from-local: P_world = R_WX @ p_local + t_WX.
  head   H : X = neutral aim (camera optical axis), Y = left, Z = up-ish.
             R_WH = [X_H | Y_H | Z_H] = Rz(base_yaw) @ Ry(-(shoulder + elbow)).
  camera C : OpenCV axes, X = -Y_H (right), Y = -Z_H (down), Z = +X_H (forward).
             Co-located with the head origin and rigid with the *forearm* --
             it does not move with pan/tilt.

Positive pitch raises a link; positive yaw turns toward world +Y (left).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autotype_sim.core.config import NJ, ArmConfig, CameraConfig


# --------------------------------------------------------------------------- #
# Elementary rotations (active, right-handed, world-from-local)
# --------------------------------------------------------------------------- #


def rot_x(angle: float) -> np.ndarray:
    """Rotation about +X by `angle`; rot_x(+90 deg) maps +Y onto +Z."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    """Rotation about +Y by `angle`; rot_y(+90 deg) maps +Z onto +X."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    """Rotation about +Z by `angle`; rot_z(+90 deg) maps +X onto +Y."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
# Forward kinematics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArmPose:
    """World-frame pose of the arm at one joint vector (the design spec section 2).

    p_*    : joint origins in world, shape (3,).
    R_head : R_WH = [X_H | Y_H | Z_H], columns are the head axes in world.
    aim    : unit stylus direction in world, R_WH @ u_H(pan, tilt).
    R_cam  : R_WC = [-Y_H | -Z_H | X_H]  (OpenCV camera axes in world).
    t_cam  : camera origin in world (== p_head).
    """

    p_shoulder: np.ndarray
    p_elbow: np.ndarray
    p_head: np.ndarray
    R_head: np.ndarray
    aim: np.ndarray
    R_cam: np.ndarray
    t_cam: np.ndarray


def forward_kinematics(arm: ArmConfig, q: np.ndarray) -> ArmPose:
    """Evaluate the design spec section 2 chain at joint vector `q` (shape (NJ,)).

    No joint-limit clipping is applied: this is the pure geometric map, and
    the plant is responsible for keeping `q` legal.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (NJ,):
        raise ValueError(f"q must have shape ({NJ},), got {q.shape}")
    th0, th1, th2, pan, tilt = q

    c0, s0 = np.cos(th0), np.sin(th0)
    c1, s1 = np.cos(th1), np.sin(th1)
    phi = th1 + th2  # absolute forearm pitch from horizontal
    cphi, sphi = np.cos(phi), np.sin(phi)

    p_shoulder = np.array([0.0, 0.0, arm.base_height])
    p_elbow = p_shoulder + arm.upper_arm * np.array([c1 * c0, c1 * s0, s1])
    p_head = p_elbow + arm.forearm * np.array([cphi * c0, cphi * s0, sphi])

    x_h = np.array([cphi * c0, cphi * s0, sphi])
    y_h = np.array([-s0, c0, 0.0])
    z_h = np.array([-sphi * c0, -sphi * s0, cphi])
    R_head = np.column_stack((x_h, y_h, z_h))

    # Spherical aim in the head frame: pan = azimuth, tilt = elevation.
    ct = np.cos(tilt)
    u_h = np.array([ct * np.cos(pan), ct * np.sin(pan), np.sin(tilt)])
    aim = R_head @ u_h

    R_cam = np.column_stack((-y_h, -z_h, x_h))

    return ArmPose(
        p_shoulder=p_shoulder,
        p_elbow=p_elbow,
        p_head=p_head,
        R_head=R_head,
        aim=aim,
        R_cam=R_cam,
        t_cam=p_head.copy(),
    )


# --------------------------------------------------------------------------- #
# Aim inverse: the exact inverse of the spherical aim above
# --------------------------------------------------------------------------- #


def aim_inverse(
    R_head: np.ndarray, p_head: np.ndarray, target: np.ndarray
) -> tuple[float, float, float]:
    """Return (pan, tilt, range) that aims the stylus from p_head at `target`.

    v_H = R_WH^T (target - p_head); pan = atan2(v_H.y, v_H.x);
    tilt = atan2(v_H.z, hypot(v_H.x, v_H.y)); range = |v|.
    Exact inverse of the spherical aim in forward_kinematics.
    """
    v = np.asarray(target, dtype=float) - np.asarray(p_head, dtype=float)
    v_h = np.asarray(R_head, dtype=float).T @ v
    pan = float(np.arctan2(v_h[1], v_h[0]))
    tilt = float(np.arctan2(v_h[2], np.hypot(v_h[0], v_h[1])))
    rng = float(np.linalg.norm(v))
    return pan, tilt, rng


# --------------------------------------------------------------------------- #
# Pinhole projection
# --------------------------------------------------------------------------- #


def project_points(
    cam: CameraConfig,
    R_cam: np.ndarray,
    t_cam: np.ndarray,
    pts_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points through the pinhole; returns (uv[N,2], z[N]).

    P_C = R_WC^T (P - t_WC); u = fx*P_C.x/P_C.z + cx; v = fy*P_C.y/P_C.z + cy.
    A projection is valid iff z > 0; rows with z <= 0 are filled with NaN so
    behind-camera points can never masquerade as pixels. Accepts shape (N,3)
    or a single point (3,) (returned as N == 1).
    """
    pts = np.asarray(pts_world, dtype=float)
    if pts.ndim == 1 and pts.shape == (3,):
        pts = pts[np.newaxis, :]
    elif pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts_world must have shape (N, 3) or (3,), got {pts.shape}")

    R = np.asarray(R_cam, dtype=float)
    t = np.asarray(t_cam, dtype=float)
    p_c = (pts - t) @ R  # row i == R_WC^T @ (P_i - t_WC)
    z = p_c[:, 2].copy()

    valid = z > 0.0
    uv = np.full((pts.shape[0], 2), np.nan)
    uv[valid, 0] = cam.fx * p_c[valid, 0] / z[valid] + cam.cx
    uv[valid, 1] = cam.fy * p_c[valid, 1] / z[valid] + cam.cy
    return uv, z


def in_frame(cam: CameraConfig, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Boolean mask: z > 0 and 0 <= u <= width-1 and 0 <= v <= height-1."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    z = np.asarray(z, dtype=float).reshape(-1)
    u, v = uv[:, 0], uv[:, 1]
    return (
        (z > 0.0)
        & (u >= 0.0)
        & (u <= cam.width - 1)
        & (v >= 0.0)
        & (v <= cam.height - 1)
    )
