"""Mounting-panel geometry, board pose and per-episode pose randomisation.

Conventions:

  board B : origin at the panel's top-left corner as seen from the arm,
            X right, Y down, Z into the panel (away from the arm). This is
            the image-pixel convention scaled to metres, so texture (u, v)
            and board (x, y) differ by one scalar. Right-handed.
            P_world = R_WB @ b + t_WB. The face normal pointing toward the
            arm is -Z_B; the arm sits on the negative-Z side of the panel.
  nominal : X_B = (0, -1, 0)_W, Y_B = (0, 0, -1)_W, Z_B = (1, 0, 0)_W, i.e.
            the panel faces the arm squarely, top edge up.
  markers : ArUco corner order TL, TR, BR, BL as seen from the arm. Markers
            are upright: marker x along +X_B, marker y along -Y_B (ArUco has
            y up). ID <-> corner: 0 = TL, 1 = TR, 2 = BR, 3 = BL.

Geometry is read from a site configuration file; only the URC-public subset
is exposed through BoardGeometry.public_dict().

``pose_salt`` is a per-deployment integer mixed into the per-episode pose
stream, so the episode seed alone does not determine the board pose. It is
never part of ``public_dict()``.
"""

from __future__ import annotations

import numbers
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

from autotype_sim.core.config import DEG, ArmConfig, CameraConfig
from autotype_sim.core.keymap import TKL_HEIGHT_U, TKL_WIDTH_U
from autotype_sim.core.kinematics import (
    forward_kinematics,
    in_frame,
    project_points,
    rot_x,
    rot_y,
    rot_z,
)

# Nominal board orientation, columns [X_B | Y_B | Z_B] in world (the design spec s.1).
R_WB_NOMINAL = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)
R_WB_NOMINAL.setflags(write=False)  # shared module constant: never mutable in place

MARKER_IDS = (0, 1, 2, 3)  # TL, TR, BR, BL, clockwise as seen from the arm

# Fields URC publishes to competitors (the design spec s.4); everything else in the
# YAML describes where the keyboard sits inside the panel and stays private.
PUBLIC_FIELDS = ("panel_w", "panel_h", "marker_size", "marker_dict", "marker_centers")

# Slack for the geometry containment checks (metres). Absorbs decimal->binary
# rounding (0.175 - 0.01 == 0.16499999999999998) so a marker or plate flush
# with an edge is accepted, while anything physically off the panel is not.
_EPS = 1e-9


def _finite_float(value: Any, name: str) -> float:
    val = float(value)
    if not np.isfinite(val):
        raise ValueError(f"{name} must be finite, got {val}")
    return val


def _xy(value: Any, name: str) -> tuple[float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (2,):
        raise ValueError(f"{name} must be an [x, y] pair, got {value!r}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(arr[0]), float(arr[1])


def _check_rect_inside(
    what: str,
    origin: tuple[float, float],
    size: tuple[float, float],
    container: str,
    c_origin: tuple[float, float],
    c_size: tuple[float, float],
) -> None:
    """Raise unless the axis-aligned rectangle `what` lies inside `container`."""
    x0, y0 = origin
    x1, y1 = x0 + size[0], y0 + size[1]
    cx0, cy0 = c_origin
    cx1, cy1 = cx0 + c_size[0], cy0 + c_size[1]
    if x0 < cx0 - _EPS or y0 < cy0 - _EPS or x1 > cx1 + _EPS or y1 > cy1 + _EPS:
        raise ValueError(
            f"{what} x[{x0:.5f}, {x1:.5f}] y[{y0:.5f}, {y1:.5f}] m does not fit inside the "
            f"{container} x[{cx0:.5f}, {cx1:.5f}] y[{cy0:.5f}, {cy1:.5f}] m"
        )


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BoardGeometry:
    """Panel, marker and keyboard placement in the board frame (metres).

    Field names match the keys of board_geometry.yaml so the object can be
    used in place of the raw mapping (keymap.generate_tkl reads either).

    ``pose_salt`` is the only non-geometric field: an integer mixed into the
    board-pose random stream (see :func:`pose_rng`). It is REQUIRED --
    ``from_dict`` / ``load`` reject a YAML without it -- and it is not part
    of :meth:`public_dict`, which carries the URC-published fields only.
    """

    panel_w: float
    panel_h: float
    marker_size: float
    marker_dict: str
    marker_centers: dict[int, tuple[float, float]]
    kb_origin: tuple[float, float]
    kb_w: float
    kb_h: float
    key_pitch: float
    key_area_origin: tuple[float, float]
    key_inset: float
    pose_salt: int

    def __post_init__(self) -> None:
        for name in ("panel_w", "panel_h", "marker_size", "kb_w", "kb_h", "key_pitch"):
            val = _finite_float(getattr(self, name), name)
            if not val > 0.0:
                raise ValueError(f"{name} must be positive, got {val}")
            object.__setattr__(self, name, val)
        inset = _finite_float(self.key_inset, "key_inset")
        if inset < 0.0:
            raise ValueError(f"key_inset must be non-negative, got {inset}")
        if not inset < 0.5 * self.key_pitch:
            # The design spec s.5: a cell shrunk by >= half a pitch per side has no
            # registration rectangle left, so every press would be NO_KEY.
            raise ValueError(
                f"key_inset must be < key_pitch / 2 = {0.5 * self.key_pitch}, got {inset}"
            )
        object.__setattr__(self, "key_inset", inset)
        salt = self.pose_salt
        if isinstance(salt, bool) or not isinstance(salt, numbers.Integral):
            raise ValueError(f"pose_salt must be an integer, got {salt!r}")
        object.__setattr__(self, "pose_salt", int(salt))
        object.__setattr__(self, "marker_dict", str(self.marker_dict))
        kb_origin = _xy(self.kb_origin, "kb_origin")
        key_origin = _xy(self.key_area_origin, "key_area_origin")
        object.__setattr__(self, "kb_origin", kb_origin)
        object.__setattr__(self, "key_area_origin", key_origin)

        # The design spec s.4: the keyboard plate sits inside the panel and the
        # TKL_WIDTH_U x TKL_HEIGHT_U key grid (s.5) sits inside the plate.
        plate = (self.kb_w, self.kb_h)
        _check_rect_inside(
            "keyboard plate", kb_origin, plate, "panel", (0.0, 0.0), (self.panel_w, self.panel_h)
        )
        grid = (TKL_WIDTH_U * self.key_pitch, TKL_HEIGHT_U * self.key_pitch)
        _check_rect_inside("key grid", key_origin, grid, "keyboard plate", kb_origin, plate)

        raw = self.marker_centers
        centers = {int(k): _xy(v, f"marker_centers[{k}]") for k, v in raw.items()}
        if len(centers) != len(raw):
            raise ValueError(
                f"marker_centers has duplicate ids after int normalisation: {list(raw)}"
            )
        if tuple(sorted(centers)) != MARKER_IDS:
            raise ValueError(f"marker_centers must have ids {MARKER_IDS}, got {sorted(centers)}")
        half = self.marker_size / 2.0
        lo, hi_x, hi_y = half - _EPS, self.panel_w - half + _EPS, self.panel_h - half + _EPS
        for mid, (cx, cy) in centers.items():
            if not (lo <= cx <= hi_x and lo <= cy <= hi_y):
                raise ValueError(f"marker {mid} at {(cx, cy)} does not fit inside the panel")
        object.__setattr__(self, "marker_centers", centers)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, source: str = "geometry") -> "BoardGeometry":
        """Build from a mapping with the board_geometry.yaml keys (the design spec s.4).

        Extra keys are ignored; a missing key is a ValueError naming ``source``.
        """
        if not isinstance(raw, Mapping):
            raise ValueError(f"{source}: expected a mapping, got {type(raw).__name__}")
        missing = [k for k in cls.__dataclass_fields__ if k not in raw]
        if missing:
            raise ValueError(f"{source}: missing keys {missing}")
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "BoardGeometry":
        """Read private/board_geometry.yaml (the design spec section 4 keys)."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping at top level")
        return cls.from_dict(raw, source=str(path))

    def public_dict(self) -> dict[str, Any]:
        """URC-public subset only: panel size, marker size/dictionary/centres.

        Plain Python scalars and lists so the result serialises to YAML/JSON
        unchanged. Nothing about the keyboard's placement is included, and
        ``pose_salt`` is never included.
        """
        return {
            "panel_w": self.panel_w,
            "panel_h": self.panel_h,
            "marker_size": self.marker_size,
            "marker_dict": self.marker_dict,
            "marker_centers": {
                mid: [float(cx), float(cy)] for mid, (cx, cy) in sorted(self.marker_centers.items())
            },
        }

    def marker_center(self, marker_id: int) -> np.ndarray:
        """Marker centre in the board frame, shape (3,), z = 0."""
        cx, cy = self.marker_centers[int(marker_id)]
        return np.array([cx, cy, 0.0])

    def panel_corners_board(self) -> np.ndarray:
        """Panel corners TL, TR, BR, BL in the board frame, shape (4, 3)."""
        w, h = self.panel_w, self.panel_h
        return np.array(
            [[0.0, 0.0, 0.0], [w, 0.0, 0.0], [w, h, 0.0], [0.0, h, 0.0]]
        )


def marker_corners_board(geometry: BoardGeometry) -> dict[int, np.ndarray]:
    """Marker corners in the board frame, {id: [4,3]}, ArUco order TL,TR,BR,BL.

    Upright marker: marker x along +X_B, marker y along -Y_B, so its top-left
    (as seen from the arm) is the corner with the smallest board x and y.
    """
    half = geometry.marker_size / 2.0
    offsets = np.array(
        [[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]]
    )
    return {
        mid: geometry.marker_center(mid) + offsets for mid in sorted(geometry.marker_centers)
    }


# --------------------------------------------------------------------------- #
# Pose
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BoardPose:
    """World-from-board rigid transform: P_W = R @ b + t.

    R = R_WB (columns are X_B, Y_B, Z_B in world); t = t_WB is the panel's
    top-left corner in world.
    """

    R: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        R = np.array(self.R, dtype=float)
        t = np.array(self.t, dtype=float).reshape(-1)
        if R.shape != (3, 3):
            raise ValueError(f"R must have shape (3, 3), got {R.shape}")
        if t.shape != (3,):
            raise ValueError(f"t must have shape (3,), got {t.shape}")
        if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
            raise ValueError("R and t must be finite (no NaN / inf)")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-9) or np.linalg.det(R) < 0.0:
            raise ValueError("R must be a proper rotation (orthonormal, det +1)")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    def to_world(self, b: np.ndarray) -> np.ndarray:
        """Board -> world for points of shape (..., 3): R @ b + t."""
        b = np.asarray(b, dtype=float)
        if b.shape[-1:] != (3,):
            raise ValueError(f"board points must have shape (..., 3), got {b.shape}")
        return b @ self.R.T + self.t

    def to_board(self, P: np.ndarray) -> np.ndarray:
        """World -> board for points of shape (..., 3): R^T (P - t)."""
        P = np.asarray(P, dtype=float)
        if P.shape[-1:] != (3,):
            raise ValueError(f"world points must have shape (..., 3), got {P.shape}")
        return (P - self.t) @ self.R

    @property
    def normal_toward_arm(self) -> np.ndarray:
        """Unit face normal pointing back toward the arm: n = -Z_B."""
        return -self.R[:, 2]

    def corners_world(self, geometry: BoardGeometry) -> np.ndarray:
        """Panel corners TL, TR, BR, BL in world, shape (4, 3)."""
        return self.to_world(geometry.panel_corners_board())

    def center_world(self, geometry: BoardGeometry) -> np.ndarray:
        """Panel centre in world, shape (3,)."""
        return self.to_world(np.array([geometry.panel_w / 2.0, geometry.panel_h / 2.0, 0.0]))


def marker_corners_world(geometry: BoardGeometry, pose: BoardPose) -> dict[int, np.ndarray]:
    """Marker corners in world, {id: [4,3]}, ArUco order TL, TR, BR, BL.

    Order is that of an upright marker seen from the arm (the design spec s.4), so
    it matches what cv2.aruco returns for the rendered panel.
    """
    return {mid: pose.to_world(c) for mid, c in marker_corners_board(geometry).items()}


# --------------------------------------------------------------------------- #
# Randomisation (the design spec section 6)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SampleRanges:
    """Uniform ranges for sample_board_pose.

    The panel centre is drawn from the axis-aligned world box
    [box_min, box_max]; yaw / pitch / roll are drawn symmetrically from
    [-yaw, +yaw] etc. (radians) and applied about world Z / Y / X.

    Defaults are tuned together with ArmConfig.q_home so that every draw
    keeps the whole panel in frame with >= 40 px to spare at the home pose
    (The design spec s.6; the tuned pair gives >= 63 px). The vertical field of
    view is the binding constraint, which is why the z range is the narrow
    one and the box starts 1.0 m out.
    """

    box_min: np.ndarray = field(default_factory=lambda: np.array([1.00, -0.12, 0.30]))
    box_max: np.ndarray = field(default_factory=lambda: np.array([1.15, 0.12, 0.42]))
    yaw: float = 25.0 * DEG
    pitch: float = 10.0 * DEG
    roll: float = 10.0 * DEG

    def __post_init__(self) -> None:
        for name in ("box_min", "box_max"):
            # np.array (not asarray): own copy, never aliasing the caller's
            # buffer; read-only so the shared default instance in the
            # sample_board_pose signature cannot drift (seed reproducibility).
            arr = np.array(getattr(self, name), dtype=float)
            if arr.shape != (3,):
                raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite, got {arr.tolist()}")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        if np.any(self.box_min > self.box_max):
            raise ValueError("every box_min must be <= its box_max")
        for name in ("yaw", "pitch", "roll"):
            val = float(getattr(self, name))
            if not np.isfinite(val) or val < 0.0:
                raise ValueError(
                    f"{name} half-range must be a finite non-negative angle, got {val}"
                )
            object.__setattr__(self, name, val)


def board_pose_from_center(
    geometry: BoardGeometry, center_W: np.ndarray, yaw: float, pitch: float, roll: float
) -> BoardPose:
    """R_WB = Rz(yaw) Ry(pitch) Rx(roll) R_WB_nominal; t_WB places the panel
    centre at `center_W` (the design spec s.6)."""
    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll) @ R_WB_NOMINAL
    half = np.array([geometry.panel_w / 2.0, geometry.panel_h / 2.0, 0.0])
    t = np.asarray(center_W, dtype=float) - R @ half
    return BoardPose(R=R, t=t)


def _frame_failure(cam: CameraConfig, uv: np.ndarray, z: np.ndarray) -> str | None:
    """Name the first violated in-frame constraint, or None if all points pass."""
    if np.any(z <= 0.0):
        return "marker corner behind the camera (P_C.z <= 0)"
    u, v = uv[:, 0], uv[:, 1]
    if np.any(u < 0.0):
        return "marker corner left of the image (u < 0)"
    if np.any(u > cam.width - 1):
        return f"marker corner right of the image (u > {cam.width - 1})"
    if np.any(v < 0.0):
        return "marker corner above the image (v < 0)"
    if np.any(v > cam.height - 1):
        return f"marker corner below the image (v > {cam.height - 1})"
    return None


def pose_rng(seed: int, pose_salt: int) -> np.random.Generator:
    """The board-pose stream of an episode: ``default_rng(SeedSequence([seed,
    pose_salt]))``.

    ``pose_salt`` is part of the site configuration, so the source plus the
    (public) episode seed is not enough to reproduce a pose:
    ``default_rng(seed)`` alone draws a completely different one.
    """
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or int(seed) < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if isinstance(pose_salt, bool) or not isinstance(pose_salt, numbers.Integral):
        raise ValueError(f"pose_salt must be an integer, got {pose_salt!r}")
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(pose_salt)]))


def sample_board_pose(
    seed: int | None,
    geometry: BoardGeometry,
    arm: ArmConfig,
    cam: CameraConfig,
    ranges: SampleRanges = SampleRanges(),
    *,
    rng: np.random.Generator | None = None,
    max_draws: int = 1000,
) -> BoardPose:
    """Draw a board pose (the design spec s.6), rejecting any whose 16 marker
    corners are not all in frame with the arm at `arm.q_home`.

    The stream is `pose_rng(seed, geometry.pose_salt)`: the episode seed is
    public, the salt is not, so the seed alone does NOT determine the pose.
    Tests and tools may inject an explicit Generator with `rng=`; `seed` is
    then ignored and may be None.

    Draw order per attempt (fixed, so a (seed, salt) pair reproduces a pose):
    centre ~ U(box), yaw, pitch, roll. Raises RuntimeError after `max_draws`
    rejections -- the box and q_home are mis-tuned, a configuration bug.
    `max_draws` must be >= 1 (ValueError otherwise).
    """
    if max_draws < 1:
        raise ValueError(f"max_draws must be >= 1, got {max_draws}")
    if rng is None:
        if seed is None:
            raise ValueError("sample_board_pose needs either a seed or an explicit rng=")
        rng = pose_rng(seed, geometry.pose_salt)
    elif not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")
    home = forward_kinematics(arm, arm.q_home)
    corners_b = np.concatenate(list(marker_corners_board(geometry).values()), axis=0)
    failures: dict[str, int] = {}
    last_failure = "none"

    for _ in range(max_draws):
        center = rng.uniform(ranges.box_min, ranges.box_max)
        yaw = rng.uniform(-ranges.yaw, ranges.yaw)
        pitch = rng.uniform(-ranges.pitch, ranges.pitch)
        roll = rng.uniform(-ranges.roll, ranges.roll)
        pose = board_pose_from_center(geometry, center, yaw, pitch, roll)

        uv, z = project_points(cam, home.R_cam, home.t_cam, pose.to_world(corners_b))
        if bool(np.all(in_frame(cam, uv, z))):
            return pose
        last_failure = _frame_failure(cam, uv, z) or "unknown"
        failures[last_failure] = failures.get(last_failure, 0) + 1

    worst = max(failures, key=failures.get)
    raise RuntimeError(
        f"sample_board_pose: {max_draws} consecutive draws rejected; most common "
        f"failing constraint: {worst!r} ({failures[worst]}x), last: {last_failure!r}. "
        f"Breakdown: {failures}. The sampling box {ranges.box_min.tolist()}..{ranges.box_max.tolist()} "
        f"and/or ArmConfig.q_home are mis-tuned (configuration bug, not a runtime condition)."
    )
