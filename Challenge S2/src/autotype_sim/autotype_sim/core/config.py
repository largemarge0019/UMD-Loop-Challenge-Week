"""Robot, camera and board parameters for the URC autonomous-typing simulator.

All lengths are metres, all angles radians, all times seconds.

Frame conventions (documented verbatim in docs/INTERFACES.md):

  world   : X forward, Y left, Z up. Arm base at the origin, on the ground.
  head    : X = neutral aim direction (also the camera optical axis),
            Y = left, Z = up. Right-handed.
  camera  : OpenCV convention -- Z forward, X right, Y down.
            Hence cam_x = -head_Y, cam_y = -head_Z, cam_z = +head_X.
  board   : origin at the mounting panel's top-left corner, X right, Y down,
            Z INTO the panel (right-handed; the face normal toward the arm is -Z).

Joint sign conventions:
  base_yaw       positive turns the arm to its left (toward world +Y).
  shoulder_pitch measured from horizontal, positive raises the upper arm.
  elbow_pitch    measured relative to the upper arm, positive raises
                 the forearm. Absolute forearm pitch is shoulder + elbow.
  head_pan       azimuth within the head frame, positive toward head +Y.
  head_tilt      elevation within the head frame, positive toward head +Z.

The pan/tilt pair is a plain spherical aim: the stylus points along
  u_head = [cos(tilt)cos(pan), cos(tilt)sin(pan), sin(tilt)]
so pan/tilt and range are uniquely determined by any target point. There is
no redundancy and no extension axis -- a press is a stroke the simulator
executes along the aimed ray (see core/press.py).
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field, fields

import numpy as np

JOINT_NAMES = (
    "base_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "head_pan",
    "head_tilt",
)
NJ = len(JOINT_NAMES)

DEG = np.pi / 180.0


# ---------------------------------------------------------------------------
# Validation helpers. Every config below checks its fields at construction:
# a mis-tuned configuration is a bug and must raise here, not surface later as
# a ZeroDivisionError in the plant, a NaN joint state or a mirrored image
# (The design spec s.6 states the policy; s.2/s.3/s.7 give the formulas these
# guards keep well-defined).
# ---------------------------------------------------------------------------


def _finite_real(name: str, value) -> float:
    """``value`` as a finite float; TypeError for anything that is not a plain
    real scalar (strings, bools, arrays), ValueError for nan/inf."""
    if isinstance(value, (bool, str, bytes)) or np.ndim(value) != 0:
        raise TypeError(f"{name} must be a real number, got {value!r}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a real number, got {value!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _positive(name: str, value) -> float:
    v = _finite_real(name, value)
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return v


def _non_negative(name: str, value) -> float:
    v = _finite_real(name, value)
    if v < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return v


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer pixel count, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return int(value)


def _joint_vector(name: str, value) -> np.ndarray:
    """A private, read-only, finite float copy of shape ``(NJ,)``.

    Always copies (``np.asarray`` would alias a caller's float64 array) and
    locks the result so the invariants checked in ``__post_init__`` cannot be
    broken afterwards through the ndarray -- ``frozen=True`` only protects the
    attribute binding, not the buffer behind it."""
    arr = np.array(value, dtype=float)
    if arr.shape != (NJ,):
        raise ValueError(f"{name} must have shape ({NJ},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {arr}")
    arr.setflags(write=False)
    return arr


def _value_key(cfg) -> tuple:
    """Hashable, comparable tuple of a config's fields (ndarrays -> tuples)."""
    return tuple(
        tuple(v.tolist()) if isinstance(v, np.ndarray) else v
        for v in (getattr(cfg, f.name) for f in fields(cfg))
    )


@dataclass(frozen=True)
class ArmConfig:
    """Link geometry, joint limits and actuator capability.

    The link lengths are chosen so that the task is solvable: for every
    board pose the randomiser can produce, some parked arm pose inside the
    joint limits reaches every alphanumeric key with the stylus alone.
    upper_arm = 0.60 with forearm = 0.40 gives a reach of 1.00 m from the
    shoulder; the earlier 0.45 / 0.40 arm (reach 0.85 m) could not cover
    the ``board.SampleRanges`` box at all.
    """

    base_height: float = 0.30
    upper_arm: float = 0.60
    forearm: float = 0.40

    # Per joint, in JOINT_NAMES order.
    q_min: np.ndarray = field(
        default_factory=lambda: np.array(
            [-120.0, -30.0, -140.0, -45.0, -35.0]
        )
        * DEG
    )
    q_max: np.ndarray = field(
        default_factory=lambda: np.array(
            [120.0, 100.0, 0.0, 45.0, 35.0]
        )
        * DEG
    )
    v_max: np.ndarray = field(
        default_factory=lambda: np.array([0.6, 0.6, 0.8, 1.0, 1.0])
    )
    a_max: np.ndarray = field(
        default_factory=lambda: np.array([1.5, 1.5, 2.0, 3.0, 3.0])
    )

    # Home pose the arm resets to at the start of every episode. Chosen so the
    # board is fully in frame for every pose the randomiser can produce:
    # upper arm near vertical, forearm (= camera axis) pitched 28 deg down so
    # it points from the elbow at the centre of the board.SampleRanges box,
    # camera ~0.76 m from that centre. Tuned jointly with SampleRanges
    # (>= 55 px edge margin over the whole box and all angle extremes) while
    # keeping every joint >= 0.26 rad from its stops.
    q_home: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 85.0, -113.0, 0.0, 0.0]) * DEG
    )

    # Stylus stroke limits, measured along the aimed ray from the head origin.
    stylus_min: float = 0.05
    stylus_max: float = 0.35

    def __post_init__(self) -> None:
        # Link geometry (DESIGN s.2: L0 = base_height, L1 = upper_arm,
        # L2 = forearm). A negative length flips the joint sign conventions.
        _non_negative("base_height", self.base_height)
        _positive("upper_arm", self.upper_arm)
        _positive("forearm", self.forearm)
        for name in ("q_min", "q_max", "v_max", "a_max", "q_home"):
            object.__setattr__(self, name, _joint_vector(name, getattr(self, name)))
        if np.any(self.q_min >= self.q_max):
            raise ValueError("every q_min must be strictly below its q_max")
        if np.any(self.q_home < self.q_min) or np.any(self.q_home > self.q_max):
            raise ValueError("q_home must lie within the joint limits")
        # DESIGN s.3: clip(qd_cmd, -v_max, v_max) and clip(dv, -a_max*dt,
        # a_max*dt) are only well-formed for positive bounds; a zero or
        # negative capability freezes or sign-flips the joint silently.
        if np.any(self.v_max <= 0.0):
            raise ValueError("every v_max must be > 0")
        if np.any(self.a_max <= 0.0):
            raise ValueError("every a_max must be > 0")
        _finite_real("stylus_min", self.stylus_min)
        _finite_real("stylus_max", self.stylus_max)
        if not 0.0 < self.stylus_min < self.stylus_max:
            raise ValueError("need 0 < stylus_min < stylus_max")

    # The generated __eq__/__hash__ break on ndarray fields (elementwise ==
    # has no truth value; ndarrays are unhashable). Compare and hash by value
    # so configs behave as the immutable value objects they are declared as.
    def __eq__(self, other) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return _value_key(self) == _value_key(other)

    def __hash__(self) -> int:
        return hash(_value_key(self))


@dataclass(frozen=True)
class PlantConfig:
    """Actuator imperfection.

    The defaults are a noiseless plant. A deployment supplies its own
    magnitudes; nothing about the arm's tracking error is a constant of the
    model, and none of it is reproducible from the episode seed.
    """

    rate: float = 50.0

    # Commanded velocity is followed imperfectly. The multiplicative term
    # dominates, so moving fast is punished more than moving slowly -- which
    # is what makes a settle-before-press discipline necessary. 0.0 is a
    # perfectly-tracking actuator.
    noise_rel: float = 0.0
    noise_abs: float = 0.0

    # Velocity decays to zero if no command arrives within this window, so a
    # member's node cannot fire one command and coast: it must run a loop.
    watchdog: float = 0.10

    def __post_init__(self) -> None:
        # DESIGN s.3: integration at dt = 1/rate; N(0, sigma) needs sigma >= 0;
        # a watchdog <= 0 expires before any step so no command is ever followed.
        _positive("rate", self.rate)
        _non_negative("noise_rel", self.noise_rel)
        _non_negative("noise_abs", self.noise_abs)
        _positive("watchdog", self.watchdog)

    @property
    def dt(self) -> float:
        return 1.0 / self.rate


@dataclass(frozen=True)
class CameraConfig:
    """Pinhole camera rigidly attached to the forearm (NOT to the pan/tilt
    head), so it stops moving once the arm parks. Its optical axis is the
    head's neutral aim, which is what makes the stylus a centred reticle at
    pan = tilt = 0."""

    width: int = 1280
    height: int = 720
    fx: float = 900.0
    fy: float = 900.0

    def __post_init__(self) -> None:
        # DESIGN s.2: the in-frame test is 0 <= u <= width-1 and the image is
        # (height, width, 3) uint8, so the dimensions are positive integers;
        # u = fx * x / z + cx collapses (fx = 0) or mirrors (fx < 0) otherwise.
        _positive_int("width", self.width)
        _positive_int("height", self.height)
        _positive("fx", self.fx)
        _positive("fy", self.fy)

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    @property
    def hfov(self) -> float:
        return 2.0 * np.arctan2(self.width / 2.0, self.fx)


@dataclass(frozen=True)
class PressConfig:
    """Thresholds the press ladder checks, in order. See core/press.py."""

    max_incidence: float = 55.0 * DEG
    max_joint_speed: float = 0.02
    debounce: float = 0.15

    def __post_init__(self) -> None:
        # Thresholds compared with ">" / "<" (DESIGN s.7); zero is the
        # strictest legitimate setting for each, nan/negative make the ladder
        # unconditionally pass or fail a rung.
        _non_negative("max_incidence", self.max_incidence)
        _non_negative("max_joint_speed", self.max_joint_speed)
        _non_negative("debounce", self.debounce)


@dataclass(frozen=True)
class SimConfig:
    arm: ArmConfig = field(default_factory=ArmConfig)
    plant: PlantConfig = field(default_factory=PlantConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    press: PressConfig = field(default_factory=PressConfig)
