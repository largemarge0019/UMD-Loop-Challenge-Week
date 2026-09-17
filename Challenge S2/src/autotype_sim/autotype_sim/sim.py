"""SimCore: the rclpy-free orchestrator behind ``nodes/sim_node.py``.

``SimCore`` owns the plant, the board pose, the key map, the episode and the
panel texture, and exposes everything the ROS node and the dashboard need as
plain Python / numpy values -- no ROS types anywhere, so the whole episode
lifecycle is unit-testable on a laptop.

Conventions (the design spec s.1-2 unless noted):

  * Units: metres, radians, seconds internally; ``float64`` numpy. Only the
    two dashboard views (``static_info`` / ``snapshot``) convert angles to
    degrees, exactly as the dashboard protocol requires.
  * Time is injected. Every method that needs the clock takes ``t`` in
    seconds on the same clock as the ``t0`` given to ``reset``; the State's
    ``t`` is ``t - t0``. Nothing here reads a wall clock.
  * Seeds: ``SimCore(seed=s)`` starts episode 0 with seed ``s``; ``reset()``
    without a seed uses the previous seed + 1; ``reset(seed=k)`` uses ``k``.
    Both random streams are SALTED with ``geometry.pose_salt``: the board
    pose uses ``SeedSequence([seed, pose_salt])`` and the plant noise its
    ``spawn_key=(1,)`` child. So a (seed, salt) pair reproduces both the
    board and the joint trajectory, while the seed alone -- which members can
    read off ``/rosout`` and the parameter server -- reproduces neither.
  * The board pose stays in-process. It is available through ``board_pose``
    and ``snapshot()["board"]`` (for the dashboard) and nowhere else:
    ``tf_chain`` has no board frame, ``camera_info`` / ``static_info`` carry
    only the URC-published geometry, and no method here logs.
  * Rotations are world-from-local: a ``tf_chain`` entry
    ``(parent, child, R, t)`` means ``p_parent = R @ p_child + t``.

Public surface (kept stable for ``sim_node.py`` and ``dashboard.py``)::

    SimCore(cfg, geometry, keymap, launch_key, seed, texture, t0=0.0, *, px_per_m=None)
      .reset(seed=None, t0=0.0) -> int
      .command(names, velocities, t) -> list[str]     unknown names
      .step(t) / .press(t) -> PressResult / .done(t) -> EpisodeResult
      .q / .qd_measured / .arm_pose() / .board_pose / .finished / .seed
      .episode_index / .launch_key / .events / .result / .typed
      .render(out=None) -> (H, W, 3) uint8 / .camera_info() -> dict
      .tf_chain(static) -> [(parent, child, R, t)]
      .static_info(teleop=False) -> dict / .snapshot(t) -> dict
      .reachable_mask() -> list[bool] / .cmd_age(t) / .watchdog_tripped(t)
      .teleport(q)                                     tests / teleop only
    build_default(cfg, private_dir, launch_key, seed, texture_photo_path) -> SimCore
    resolve_private_dir(param) -> Path ; default_texture_photo() -> Path | None
"""

from __future__ import annotations

import dataclasses
import math
import numbers
import os
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from autotype_sim.core.board import BoardGeometry, BoardPose, sample_board_pose
from autotype_sim.core.config import DEG, JOINT_NAMES, NJ, SimConfig
from autotype_sim.core.episode import Episode, EpisodeResult, normalise_launch_key
from autotype_sim.core.keymap import KeyMap, generate_tkl, load_keymap
from autotype_sim.core.kinematics import (
    ArmPose,
    forward_kinematics,
    project_points,
    rot_y,
    rot_z,
)
from autotype_sim.core.plant import Plant
from autotype_sim.core.press import (
    PARALLEL_EPS,
    PressResult,
    Reason,
    evaluate_press,
    ray_board_intersection,
)
from autotype_sim.core.renderer import DEFAULT_PX_PER_M, build_texture, render

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Reason string of a press fired after ``done()``. Deliberately NOT a member
#: of ``press.Reason`` (the ladder never produces it); it appears only in the
#: returned ``PressResult.reason`` and the dashboard event.
FINISHED_REASON = "FINISHED"

#: Dashboard keeps the last N events (the dashboard protocol State.events).
EVENT_HISTORY = 50

#: ``reachable`` is recomputed at most this often in injected seconds (5 Hz).
REACHABLE_PERIOD = 0.2

#: TF frame names (REP-105 style). No board frame exists on purpose.
FRAME_WORLD = "world"
FRAME_BASE = "base_link"
FRAME_SHOULDER = "shoulder_link"
FRAME_UPPER_ARM = "upper_arm_link"
FRAME_FOREARM = "forearm_link"
FRAME_CAMERA_LINK = "camera_link"
FRAME_CAMERA_OPTICAL = "camera_optical_frame"
FRAME_HEAD = "head_link"
FRAME_STYLUS_TIP = "stylus_tip"

#: R_link<-optical: columns are the optical x, y, z axes expressed in
#: camera_link (x forward, z up). optical x = -y_link (right), optical y =
#: -z_link (down), optical z = +x_link (forward) -- i.e. cam_x = -Y_H,
#: cam_y = -Z_H, cam_z = +X_H, matching ``kinematics.ArmPose.R_cam``.
R_LINK_FROM_OPTICAL = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
)
R_LINK_FROM_OPTICAL.setflags(write=False)

#: Child stream index used to derive the plant's noise generator from the
#: salted episode entropy, so it never coincides with the board-pose stream.
_PLANT_STREAM = 1

#: Key of the optional ``plant:`` section in private/board_geometry.yaml and
#: the PlantConfig fields it may override. The magnitudes are data, not
#: algorithm, so they are loaded at runtime instead of shipping in
#: ``PlantConfig`` defaults (which are 0.0 = noiseless).
PLANT_SECTION = "plant"
PLANT_OVERRIDE_FIELDS = ("noise_rel", "noise_abs")

PRIVATE_DIR_ENV = "AUTOTYPE_PRIVATE_DIR"
GEOMETRY_FILENAME = "board_geometry.yaml"
LAYOUT_FILENAME = "keyboard_layout.yaml"
INSTALLED_PRIVATE_DIR = Path("/opt/autotype/private")
TEXTURE_PHOTO_NAME = "keyboard_grid.png"
PACKAGE_NAME = "autotype_sim"

_ZERO3 = np.zeros(3)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _v3(v: np.ndarray) -> list[float]:
    """A length-3 vector as plain floats (JSON-ready)."""
    return [float(v[0]), float(v[1]), float(v[2])]


def _deg_list(v: np.ndarray) -> list[float]:
    return [float(x) / DEG for x in np.asarray(v, dtype=float)]


def _check_seed(seed: Any) -> int:
    """A non-negative Python int (bools rejected); what ``default_rng`` accepts."""
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"seed must be an integer, got {seed!r}")
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    return seed


def _check_time(t: Any, name: str = "t") -> float:
    t = float(t)
    if not math.isfinite(t):
        raise ValueError(f"{name} must be finite, got {t!r}")
    return t


def _plant_rng(seed: int, pose_salt: int) -> np.random.Generator:
    """Plant noise generator for an episode: a fixed child of the SALTED entropy.

    ``SeedSequence([seed, pose_salt], spawn_key=(1,))`` is distinct from the
    board-pose stream ``SeedSequence([seed, pose_salt])`` yet fully determined
    by the same (seed, salt) pair. Without the private salt the public seed
    reproduces neither stream.
    """
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(pose_salt)], spawn_key=(_PLANT_STREAM,))
    )


# --------------------------------------------------------------------------- #
# Private-dir / asset resolution (the node design "Parameters")
# --------------------------------------------------------------------------- #


def _package_dir() -> Path:
    """Directory of the ``autotype_sim`` Python package (this file's parent)."""
    return Path(__file__).resolve().parent


def _source_repo_root() -> Path | None:
    """``<repo>`` when this file runs from the source checkout
    ``<repo>/src/autotype_sim/autotype_sim/sim.py``, else ``None``."""
    pkg = _package_dir()
    if len(pkg.parents) < 3:
        return None
    root = pkg.parents[2]
    if (root / "src" / PACKAGE_NAME / "package.xml").is_file():
        return root
    return None


def resolve_private_dir(param: str | os.PathLike[str] | None = "") -> Path:
    """Where the private YAML files live, per the ``private_dir`` parameter rule.

    ``param`` non-empty -> that path; else ``$AUTOTYPE_PRIVATE_DIR`` when set;
    else ``<repo>/private`` when running from a source checkout; else
    ``/opt/autotype/private`` (the Docker image). The returned directory may
    not exist -- ``build_default`` reports that with a clear error.
    """
    if param is not None and str(param) != "":
        return Path(param).expanduser()
    env = os.environ.get(PRIVATE_DIR_ENV)
    if env is not None and env != "":
        return Path(env).expanduser()
    root = _source_repo_root()
    if root is not None:
        return root / "private"
    return INSTALLED_PRIVATE_DIR


def default_texture_photo() -> Path | None:
    """The shipped key-grid photo, or ``None`` when absent (-> synthetic caps).

    Looks in the source package (``autotype_sim/assets/``) first, then in the
    installed ``share/autotype_sim/assets/`` via ament (import guarded, so
    this module stays importable without ROS).
    """
    candidate = _package_dir() / "assets" / TEXTURE_PHOTO_NAME
    if candidate.is_file():
        return candidate
    try:  # pragma: no cover - only meaningful inside a colcon install
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory(PACKAGE_NAME)) / "assets" / TEXTURE_PHOTO_NAME
        if share.is_file():
            return share
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# SimCore
# --------------------------------------------------------------------------- #


class SimCore:
    """Episode orchestrator: plant + board pose + key map + episode + texture.

    See the module docstring for conventions. Construction starts episode 0
    with ``seed`` at time ``t0``. ``texture`` is the BGR uint8 panel image
    from ``renderer.build_texture``; ``px_per_m`` (its scale) is derived from
    the texture width and ``geometry.panel_w`` unless given.
    """

    def __init__(
        self,
        cfg: SimConfig,
        geometry: BoardGeometry,
        keymap: KeyMap,
        launch_key: str,
        seed: int,
        texture: np.ndarray,
        t0: float = 0.0,
        *,
        px_per_m: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.geometry = geometry
        self.keymap = keymap
        self._launch_key = normalise_launch_key(launch_key)

        texture = np.asarray(texture)
        if texture.ndim != 3 or texture.shape[2] != 3 or texture.dtype != np.uint8:
            raise ValueError(
                f"texture must be an (h, w, 3) uint8 BGR image, got shape {texture.shape} "
                f"dtype {texture.dtype}"
            )
        self.texture = texture
        if px_per_m is None:
            px_per_m = texture.shape[1] / geometry.panel_w
        self.px_per_m = float(px_per_m)
        if not (math.isfinite(self.px_per_m) and self.px_per_m > 0.0):
            raise ValueError(f"px_per_m must be positive and finite, got {px_per_m!r}")

        # Key centres in the board frame, StaticInfo key order, shape (K, 3).
        self._key_names: list[str] = [k.name for k in keymap]
        self._key_centres_board = np.array(
            [[*KeyMap.center(k), 0.0] for k in keymap], dtype=float
        ).reshape(-1, 3)

        self._episode_index = -1
        self._seed: int = _check_seed(seed)
        self._plant: Plant | None = None
        self._pose_cache: ArmPose | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=EVENT_HISTORY)
        self._start(self._seed, _check_time(t0, "t0"))

    # ----------------------------------------------------------- lifecycle

    def _start(self, seed: int, t0: float) -> None:
        """(Re)initialise every per-episode object for ``seed`` at ``t0``."""
        self._seed = seed
        self._episode_index += 1
        self._t0 = t0
        self._board = sample_board_pose(
            seed, self.geometry, self.cfg.arm, self.cfg.camera
        )
        self._key_centres_world = self._board.to_world(self._key_centres_board)
        self._plant = Plant(
            self.cfg.arm, self.cfg.plant, _plant_rng(seed, self.geometry.pose_salt)
        )
        self._plant.reset(self.cfg.arm.q_home)
        self._qd_cmd = np.zeros(NJ)
        self._t_last_cmd: float | None = None
        self._t_last_step: float | None = None
        self._episode = Episode(self._launch_key, t0)
        self._result: EpisodeResult | None = None
        self._finished = False
        self._t_last_accepted: float | None = None
        self._pose_cache = None
        self._reach_cache: list[bool] | None = None
        self._reach_cache_t: float | None = None
        self.events.clear()
        self._event(t0, "info", text=f"episode started, seed {seed}")

    def reset(self, seed: int | None = None, t0: float = 0.0) -> int:
        """Start a new episode and return the seed used.

        ``seed=None`` continues the sequence (previous seed + 1). Everything
        per-episode is rebuilt: board pose, plant (parked at ``q_home`` with
        no command latched), episode, events, result and the reachability
        cache. The launch key is unchanged.
        """
        seed = self._seed + 1 if seed is None else _check_seed(seed)
        self._start(seed, _check_time(t0, "t0"))
        return seed

    def done(self, t: float) -> EpisodeResult:
        """Finish the episode at ``t``: score it, freeze commands, keep the result.

        The latched command is zeroed so the plant decelerates to rest and
        later ``command()`` calls are ignored until ``reset``. A second call
        raises ``RuntimeError`` (the node logs a warning and ignores it).
        """
        t = _check_time(t)
        if self._finished:
            raise RuntimeError("episode already finished; call reset() first")
        result = self._episode.finish(t)  # may raise ValueError (t < t0)
        self._result = result
        self._finished = True
        self._qd_cmd = np.zeros(NJ)
        self._plant.command(self._qd_cmd, t)
        self._t_last_cmd = t  # the freeze counts as the last command: cmd_age restarts here
        self._event(t, "info", text=f"episode finished: typed {result.typed!r}")
        return result

    # ------------------------------------------------------------ control

    def command(self, names: Sequence[str], velocities: Sequence[float], t: float) -> list[str]:
        """Latch a by-name joint-velocity command at ``t``; return unknown names.

        Joints named in ``names`` take the paired velocity (rad/s); joints not
        named keep their previous command; unknown names are ignored and
        returned so the caller can warn. ``ValueError`` if the two sequences
        differ in length or ANY paired velocity is not finite (the previous
        command is then left untouched). Finiteness is checked over the whole
        message BEFORE names are resolved, exactly in the order INTERFACES.md
        s.9.5 states it: rule 1 (drop the message) outranks rule 2 (ignore
        unknown names), so a NaN paired with a misspelt joint still drops the
        message instead of applying its siblings. After ``done()`` the call
        is a no-op returning ``[]``.
        """
        names = list(names)
        velocities = list(velocities)
        if len(names) != len(velocities):
            raise ValueError(
                f"name/velocity length mismatch: {len(names)} names, {len(velocities)} velocities"
            )
        # Rule 1 first, over every pair -- including pairs whose joint name is
        # unknown (the two failure modes correlate: both come from a bad joint
        # table, and INTERFACES.md promises the message is dropped).
        finite: list[float] = []
        for name, vel in zip(names, velocities):
            v = float(vel)
            if not math.isfinite(v):
                raise ValueError(f"velocity for {name!r} must be finite, got {vel!r}")
            finite.append(v)
        t = _check_time(t)
        if self._finished:
            return []  # frozen by done(): nothing latched, nothing to report
        unknown: list[str] = []
        updates: dict[int, float] = {}
        for name, v in zip(names, finite):
            try:
                idx = JOINT_NAMES.index(name)
            except ValueError:
                unknown.append(str(name))
                continue
            updates[idx] = v
        qd = self._qd_cmd.copy()
        for idx, v in updates.items():
            qd[idx] = v
        self._plant.command(qd, t)
        self._qd_cmd = qd
        self._t_last_cmd = t
        return unknown

    def step(self, t: float) -> None:
        """Advance the plant one tick ending at ``t`` (see ``core.plant.Plant.step``)."""
        t = _check_time(t)
        self._plant.step(t)
        self._t_last_step = t
        self._pose_cache = None

    def teleport(self, q: np.ndarray) -> None:
        """Park the arm at rest at ``q`` (inside the limits), forgetting any command.

        For tests and teleop tooling only -- there is no ROS path to this.
        ``cmd_age`` reads ``None`` afterwards until the next ``command()``.
        """
        self._plant.reset(np.asarray(q, dtype=float))
        self._qd_cmd = np.zeros(NJ)
        self._t_last_cmd = None
        self._pose_cache = None
        self._reach_cache = None
        self._reach_cache_t = None

    def press(self, t: float) -> PressResult:
        """Fire ``/arm/press`` at ``t``: run the ladder, record it, log an event.

        After ``done()`` nothing is recorded; the returned result is rejected
        with the plain string reason ``'FINISHED'`` (not a ``press.Reason``).
        """
        t = _check_time(t)
        if self._finished:
            res = PressResult(False, FINISHED_REASON, None, None, None, None)  # type: ignore[arg-type]
            self._event(t, "press", accepted=False, reason=FINISHED_REASON, key=None, board_xy=None)
            return res
        res = evaluate_press(
            self.arm_pose(),
            self._plant.qd_measured,
            self._board,
            self.keymap,
            self.cfg.arm,
            self.cfg.press,
            t,
            self._t_last_accepted,
        )
        self._episode.record_attempt(res)
        if res.accepted:
            self._t_last_accepted = t
        self._event(
            t,
            "press",
            accepted=bool(res.accepted),
            reason=str(res.reason),
            key=res.key.name if res.key is not None else None,
            board_xy=[float(res.board_xy[0]), float(res.board_xy[1])] if res.board_xy else None,
        )
        return res

    # ------------------------------------------------------------- state

    @property
    def q(self) -> np.ndarray:
        """True joint positions, shape (NJ,), radians (copy)."""
        return self._plant.q

    @property
    def qd_measured(self) -> np.ndarray:
        """Measured joint velocities from the last step, rad/s (copy)."""
        return self._plant.qd_measured

    @property
    def qd_cmd(self) -> np.ndarray:
        """The currently latched (unclipped) command, rad/s (copy)."""
        return self._qd_cmd.copy()

    @property
    def board_pose(self) -> BoardPose:
        """The world-from-board pose of this episode. In-process only."""
        return self._board

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def episode_index(self) -> int:
        """0 for the episode started at construction, +1 per ``reset``."""
        return self._episode_index

    @property
    def launch_key(self) -> str:
        return self._launch_key

    @property
    def result(self) -> EpisodeResult | None:
        return self._result

    @property
    def typed(self) -> str:
        """Typed string so far. Shown on the dashboard; never published on a topic."""
        return self._episode.typed

    @property
    def t0(self) -> float:
        return self._t0

    def arm_pose(self) -> ArmPose:
        """Forward kinematics at the current ``q`` (cached until the next step)."""
        if self._pose_cache is None:
            self._pose_cache = forward_kinematics(self.cfg.arm, self._plant.q)
        return self._pose_cache

    def cmd_age(self, t: float) -> float | None:
        """Seconds since the last accepted command, or ``None`` if none yet."""
        if self._t_last_cmd is None:
            return None
        return float(t) - self._t_last_cmd

    def watchdog_tripped(self, t: float) -> bool:
        """True iff a command was latched and ``cmd_age(t) > plant.watchdog``."""
        age = self.cmd_age(t)
        return age is not None and age > self.cfg.plant.watchdog

    # ------------------------------------------------------------ camera

    def render(self, out: np.ndarray | None = None) -> np.ndarray:
        """Camera image ``(H, W, 3)`` uint8 BGR from the current camera pose."""
        pose = self.arm_pose()
        return render(
            self.cfg.camera, pose.R_cam, pose.t_cam, self._board, self.texture, self.geometry, out=out
        )

    def camera_info(self) -> dict[str, Any]:
        """``sensor_msgs/CameraInfo`` fields as plain Python (row-major lists).

        ``K`` from ``CameraConfig``, ``D = [0]*5`` (``plumb_bob``), ``R = I``,
        ``P = [K | 0]``, ``frame_id = 'camera_optical_frame'``.
        """
        cam = self.cfg.camera
        K = cam.K
        P = np.hstack([K, np.zeros((3, 1))])
        return {
            "width": int(cam.width),
            "height": int(cam.height),
            "distortion_model": "plumb_bob",
            "D": [0.0] * 5,
            "K": [float(x) for x in K.reshape(-1)],
            "R": [float(x) for x in np.eye(3).reshape(-1)],
            "P": [float(x) for x in P.reshape(-1)],
            "frame_id": FRAME_CAMERA_OPTICAL,
        }

    # ---------------------------------------------------------------- TF

    def tf_chain(self, static: bool) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
        """The node design TF tree as ``(parent, child, R, t)`` entries.

        ``static=True`` returns the fixed transforms (``/tf_static``),
        ``static=False`` the joint-dependent ones at the current ``q``
        (``/tf``). ``R`` is 3x3 world-from-local style (``p_parent = R @
        p_child + t``). Composing both sets reproduces ``forward_kinematics``:
        ``camera_optical_frame`` in ``world`` is ``(R_cam, t_cam)`` and the
        ``head_link`` x-axis is ``aim``. There is no board frame.
        """
        arm = self.cfg.arm
        if static:
            return [
                (FRAME_WORLD, FRAME_BASE, np.eye(3), _ZERO3.copy()),
                (FRAME_FOREARM, FRAME_CAMERA_LINK, np.eye(3), np.array([arm.forearm, 0.0, 0.0])),
                (FRAME_CAMERA_LINK, FRAME_CAMERA_OPTICAL, R_LINK_FROM_OPTICAL.copy(), _ZERO3.copy()),
                (FRAME_HEAD, FRAME_STYLUS_TIP, np.eye(3), np.array([arm.stylus_max, 0.0, 0.0])),
            ]
        th0, th1, th2, pan, tilt = (float(x) for x in self._plant.q)
        return [
            (FRAME_BASE, FRAME_SHOULDER, rot_z(th0), np.array([0.0, 0.0, arm.base_height])),
            (FRAME_SHOULDER, FRAME_UPPER_ARM, rot_y(-th1), _ZERO3.copy()),
            (FRAME_UPPER_ARM, FRAME_FOREARM, rot_y(-th2), np.array([arm.upper_arm, 0.0, 0.0])),
            (FRAME_FOREARM, FRAME_HEAD, rot_z(pan) @ rot_y(-tilt), np.array([arm.forearm, 0.0, 0.0])),
        ]

    # ------------------------------------------------------- dashboard

    def static_info(self, teleop: bool = False) -> dict[str, Any]:
        """The dashboard protocol ``StaticInfo`` (angles in degrees).

        Panel and markers come from ``BoardGeometry.public_dict()`` -- the
        URC-public subset only. The key rectangles are what the page draws
        its overlay from; the dashboard is a spectator view for a person,
        outside the contract a node is written against.
        """
        arm, cam, press = self.cfg.arm, self.cfg.camera, self.cfg.press
        pub = self.geometry.public_dict()
        return {
            "joint_names": list(JOINT_NAMES),
            "q_min_deg": _deg_list(arm.q_min),
            "q_max_deg": _deg_list(arm.q_max),
            "v_max_deg_s": _deg_list(arm.v_max),
            "links": {
                "base_height": float(arm.base_height),
                "upper_arm": float(arm.upper_arm),
                "forearm": float(arm.forearm),
            },
            "stylus": {"min": float(arm.stylus_min), "max": float(arm.stylus_max)},
            "press": {
                "max_incidence_deg": float(press.max_incidence) / DEG,
                "max_joint_speed_deg_s": float(press.max_joint_speed) / DEG,
                "debounce_s": float(press.debounce),
            },
            "camera": {
                "width": int(cam.width),
                "height": int(cam.height),
                "fx": float(cam.fx),
                "fy": float(cam.fy),
                "cx": float(cam.cx),
                "cy": float(cam.cy),
            },
            "panel": {"w": float(pub["panel_w"]), "h": float(pub["panel_h"])},
            "markers": [
                {"id": int(mid), "center": [float(c[0]), float(c[1])], "size": float(pub["marker_size"])}
                for mid, c in sorted(pub["marker_centers"].items())
            ],
            "keys": [
                {"name": k.name, "kind": k.kind, "rect": [k.x0, k.y0, k.x1, k.y1]} for k in self.keymap
            ],
            "texture": {
                "px_per_m": self.px_per_m,
                "w": int(self.texture.shape[1]),
                "h": int(self.texture.shape[0]),
            },
            "teleop": bool(teleop),
            "launch_key": self._launch_key,
            "seed": self._seed,
        }

    def snapshot(self, t: float) -> dict[str, Any]:
        """The dashboard protocol ``State`` at injected time ``t`` (degrees).

        ``stylus`` / ``reticle_px`` are ``None`` when the aimed ray misses the
        panel plane; ``would_accept`` is the full ladder's verdict if
        ``press(t)`` fired now (``'FINISHED'`` after ``done``). ``reachable``
        is ``reachable_mask()`` cached for ``REACHABLE_PERIOD`` seconds of
        ``t``; ``result`` is the ``EpisodeResult`` as a dict once finished.
        """
        t = _check_time(t)
        pose = self.arm_pose()
        q = self._plant.q
        qd = self._plant.qd_measured

        stylus: dict[str, Any] | None = None
        reticle: list[float] | None = None
        hit = ray_board_intersection(pose, self._board)
        if hit is not None:
            rng, point, incidence = hit
            b = self._board.to_board(point)
            key = self.keymap.lookup(float(b[0]), float(b[1]))
            if self._finished:
                would = FINISHED_REASON
            else:
                would = str(
                    evaluate_press(
                        pose, qd, self._board, self.keymap, self.cfg.arm, self.cfg.press, t,
                        self._t_last_accepted,
                    ).reason
                )
            stylus = {
                "board_xy": [float(b[0]), float(b[1])],
                "range": float(rng),
                "incidence_deg": float(incidence) / DEG,
                "key": key.name if key is not None else None,
                "would_accept": would,
            }
            uv, z = project_points(self.cfg.camera, pose.R_cam, pose.t_cam, point)
            if z[0] > 0.0 and bool(np.all(np.isfinite(uv[0]))):
                reticle = [float(uv[0, 0]), float(uv[0, 1])]

        if (
            self._reach_cache is None
            or self._reach_cache_t is None
            or t < self._reach_cache_t
            or t - self._reach_cache_t >= REACHABLE_PERIOD
        ):
            self._reach_cache = self.reachable_mask()
            self._reach_cache_t = t

        return {
            "t": t - self._t0,
            "episode": {
                "status": "finished" if self._finished else "running",
                "seed": self._seed,
                "launch_key": self._launch_key,
            },
            "q_deg": _deg_list(q),
            "qd_deg_s": _deg_list(qd),
            "cmd_age_s": self.cmd_age(t),
            "watchdog_tripped": self.watchdog_tripped(t),
            "arm": {
                "shoulder": _v3(pose.p_shoulder),
                "elbow": _v3(pose.p_elbow),
                "head": _v3(pose.p_head),
                "aim": _v3(pose.aim),
                "cam_axis": _v3(pose.R_head[:, 0]),
            },
            "board": {
                "corners": [_v3(c) for c in self._board.corners_world(self.geometry)],
                "center": _v3(self._board.center_world(self.geometry)),
            },
            "stylus": stylus,
            "reticle_px": reticle,
            "reachable": list(self._reach_cache),
            "typed": self._episode.typed,
            "events": list(self.events),
            "result": dataclasses.asdict(self._result) if self._result is not None else None,
        }

    def reachable_mask(self) -> list[bool]:
        """Per key (StaticInfo order): could the stylus press it from *here*?

        The node design "Reachability mask": from the current ``ArmPose``,
        aim at the key centre with ``aim_inverse``; reachable iff pan / tilt
        lie within the head joint limits and the ladder's geometric rungs
        1-4 accept that ray -- it meets the panel front (``aim . n <
        -PARALLEL_EPS``, ``t > 0``), ``stylus_min <= t <= stylus_max``,
        ``incidence <= max_incidence``, and the hit lands on a key (true by
        construction: the ray ends exactly at the key's own centre, which
        lies inside its closed rectangle). Vectorised over all keys; the
        maths is ``press.ray_board_intersection`` with ``t == range``.
        """
        arm, press = self.cfg.arm, self.cfg.press
        pose = self.arm_pose()
        v = self._key_centres_world - pose.p_head  # (K, 3)
        rng = np.linalg.norm(v, axis=1)
        safe = np.where(rng > 0.0, rng, np.inf)
        v_h = v @ pose.R_head  # rows = R_WH^T v
        pan = np.arctan2(v_h[:, 1], v_h[:, 0])
        tilt = np.arctan2(v_h[:, 2], np.hypot(v_h[:, 0], v_h[:, 1]))
        denom = (v @ self._board.normal_toward_arm) / safe  # d . n
        incidence = np.arccos(np.clip(-denom, -1.0, 1.0))
        ok = (
            (rng > 0.0)
            & (denom < -PARALLEL_EPS)
            & (rng >= arm.stylus_min)
            & (rng <= arm.stylus_max)
            & (incidence <= press.max_incidence)
            & (pan >= arm.q_min[3])
            & (pan <= arm.q_max[3])
            & (tilt >= arm.q_min[4])
            & (tilt <= arm.q_max[4])
        )
        return [bool(b) for b in ok]

    # ------------------------------------------------------------ events

    def _event(self, t: float, kind: str, **fields: Any) -> None:
        """Append a protocol event; ``t`` is stored relative to the episode start."""
        ev: dict[str, Any] = {"t": float(t) - self._t0, "kind": kind}
        ev.update(fields)
        self.events.append(ev)


# --------------------------------------------------------------------------- #
# Construction from the private directory
# --------------------------------------------------------------------------- #


def load_private(private_dir: str | os.PathLike[str]) -> tuple[BoardGeometry, KeyMap]:
    """``(geometry, keymap)`` from ``private_dir``.

    ``board_geometry.yaml`` is required (``FileNotFoundError`` otherwise, no
    fallback). ``keyboard_layout.yaml`` is used when present, else the TKL map
    is generated from the geometry.
    """
    pdir = Path(private_dir)
    geom_path = pdir / GEOMETRY_FILENAME
    if not geom_path.is_file():
        raise FileNotFoundError(
            f"board geometry not found: {geom_path}. Set the private_dir parameter "
            f"or {PRIVATE_DIR_ENV} to the directory holding {GEOMETRY_FILENAME}."
        )
    geometry = BoardGeometry.load(geom_path)
    layout_path = pdir / LAYOUT_FILENAME
    keymap = load_keymap(layout_path) if layout_path.is_file() else generate_tkl(geometry)
    return geometry, keymap


def load_texture_photo(path: str | os.PathLike[str] | None) -> np.ndarray | None:
    """BGR photo at ``path``, or ``None`` when ``path`` is empty / missing / unreadable."""
    if path is None or str(path) == "":
        return None
    p = Path(path)
    if not p.is_file():
        return None
    photo = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return None if photo is None else photo


def load_plant_overrides(private_dir: str | os.PathLike[str]) -> dict[str, float]:
    """The ``plant:`` section of private/board_geometry.yaml, or ``{}``.

    Only ``PLANT_OVERRIDE_FIELDS`` are honoured; anything else in the section
    is ignored. A missing file or a missing/empty section returns ``{}``, so
    the shipped ``PlantConfig`` defaults (0.0 = noiseless) stand. Values are
    validated by ``PlantConfig.__post_init__`` when they are applied.
    """
    path = Path(private_dir) / GEOMETRY_FILENAME
    if not path.is_file():
        return {}
    import yaml  # local: keeps YAML off sim.py's import path for callers that never load private/

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    section = raw.get(PLANT_SECTION) if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        return {}
    out: dict[str, float] = {}
    for key in PLANT_OVERRIDE_FIELDS:
        if key not in section:
            continue
        try:
            out[key] = float(section[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: plant.{key} must be a number, got {section[key]!r}") from exc
    return out


def apply_private_plant(cfg: SimConfig, private_dir: str | os.PathLike[str]) -> SimConfig:
    """``cfg`` with ``cfg.plant`` carrying the configured noise magnitudes.

    Returns ``cfg`` unchanged when the directory has no ``plant:`` section.
    Nothing here logs the magnitudes (INTERFACES.md s.13).
    """
    overrides = load_plant_overrides(private_dir)
    if not overrides:
        return cfg
    return dataclasses.replace(cfg, plant=dataclasses.replace(cfg.plant, **overrides))


def build_default(
    cfg: SimConfig,
    private_dir: str | os.PathLike[str],
    launch_key: str,
    seed: int,
    texture_photo_path: str | os.PathLike[str] | None = None,
    *,
    px_per_m: float = DEFAULT_PX_PER_M,
    t0: float = 0.0,
) -> SimCore:
    """``SimCore`` from the private directory and (optionally) a plate photo.

    Geometry + key map come from ``load_private(private_dir)`` and the plant
    noise magnitudes from the same directory (``apply_private_plant``, which
    logs nothing); the texture is
    ``renderer.build_texture`` with the photo at ``texture_photo_path`` when
    that file exists and decodes, else the synthetic keyboard drawn from the
    key map.
    """
    geometry, keymap = load_private(private_dir)
    cfg = apply_private_plant(cfg, private_dir)
    photo = load_texture_photo(texture_photo_path)
    texture = build_texture(geometry, keymap, px_per_m, photo=photo)
    return SimCore(cfg, geometry, keymap, launch_key, seed, texture, t0, px_per_m=px_per_m)


__all__ = [
    "EVENT_HISTORY",
    "FINISHED_REASON",
    "FRAME_BASE",
    "FRAME_CAMERA_LINK",
    "FRAME_CAMERA_OPTICAL",
    "FRAME_FOREARM",
    "FRAME_HEAD",
    "FRAME_SHOULDER",
    "FRAME_STYLUS_TIP",
    "FRAME_UPPER_ARM",
    "FRAME_WORLD",
    "GEOMETRY_FILENAME",
    "INSTALLED_PRIVATE_DIR",
    "LAYOUT_FILENAME",
    "PLANT_OVERRIDE_FIELDS",
    "PLANT_SECTION",
    "PRIVATE_DIR_ENV",
    "REACHABLE_PERIOD",
    "R_LINK_FROM_OPTICAL",
    "SimCore",
    "apply_private_plant",
    "build_default",
    "default_texture_photo",
    "load_plant_overrides",
    "load_private",
    "load_texture_photo",
    "resolve_private_dir",
]
