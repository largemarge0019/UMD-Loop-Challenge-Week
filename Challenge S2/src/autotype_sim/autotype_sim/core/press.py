"""The ``/arm/press`` acceptance ladder.

A press is a stroke along the aimed stylus ray from the head origin. The
ladder intersects that ray with the panel plane and then checks, **in this
order**, returning the first failure:

    o = p_head, d = aim_W, n = -Z_B (toward the arm), p0 = t_WB
    denom = d . n
    1. denom >= -1e-9  or  t = ((p0 - o) . n) / denom <= 0   -> NO_INTERSECT
    2. t > stylus_max                                        -> OUT_OF_REACH
       t < stylus_min                                        -> TOO_CLOSE
    3. arccos(-denom) > max_incidence                        -> GLANCING
    4. b = R_WB^T (o + t d - t_WB); keymap.lookup(b.x, b.y) is None -> NO_KEY
    5. ||qd_measured||_inf > max_joint_speed                 -> MOVING
    6. t_now - t_last_accepted < debounce                    -> DEBOUNCE
    else ACCEPTED with key, b.xy, t, incidence

Geometry comes before dynamics on purpose: a stroke that misses every key is
reported as ``NO_KEY`` even if the arm was also moving, so the dashboard log
names the most useful reason.

Conventions: the plane is the *infinite* panel plane -- a ray that meets it
outside the keyboard (or outside the panel) is ``NO_KEY``, not
``NO_INTERSECT``. The arm sits on the negative-Z_B side, so a ray pointing at
the panel face has ``denom < 0``; ``denom >= -1e-9`` covers both "parallel"
and "pointing away". ``incidence`` is the angle between the aim and the
inward panel normal ``+Z_B``, in ``[0, pi/2)`` whenever the ray hits.

The ladder never consults link lengths or joint limits: it reads only
``ArmPose.p_head`` and ``ArmPose.aim``. Reachability is the plant's job.

Input hygiene matches the plant's (``core/plant.py`` rejects non-finite
commands and times): ``p_head``, ``aim``, ``qd_measured`` and ``t_now`` must
be finite and ``aim`` must be unit, else ``ValueError``. NaN would otherwise
fall through every comparison in the ladder -- a NaN joint velocity reads as
"settled" and a NaN time as "not debounced" -- and a non-unit aim silently
rescales the reported range. ``t_last_accepted`` may be ``None`` or ``-inf``
(both mean "no press accepted yet"); NaN and ``+inf`` raise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from autotype_sim.core.board import BoardPose
from autotype_sim.core.config import NJ, ArmConfig, PressConfig
from autotype_sim.core.keymap import Key, KeyMap
from autotype_sim.core.kinematics import ArmPose

# Step 1 tolerance on d . n: at or above this the ray is treated as parallel
# to (or pointing away from) the panel face.
PARALLEL_EPS = 1e-9

# Tolerance on | |aim| - 1 |. ``ArmPose.aim`` is unit by construction
# (The design spec s.2: aim_W = R_WH @ u_H; forward_kinematics holds it to ~5e-16)
# and the stroke length ``t`` is only a length in metres when it is. Loose
# enough to pass a float32 round trip, tight enough that any real scaling
# (which would rescale the range and collapse the incidence) raises.
UNIT_AIM_TOL = 1e-6


def _require_finite(x: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be finite, got {x}")


class Reason(str, Enum):
    """Outcome of one press, in ladder order (``ACCEPTED`` first).

    A ``str`` mixin so a reason serialises as its own name in JSON/YAML and
    compares equal to that name; ``.name`` and ``.value`` coincide.
    """

    ACCEPTED = "ACCEPTED"
    NO_INTERSECT = "NO_INTERSECT"
    OUT_OF_REACH = "OUT_OF_REACH"
    TOO_CLOSE = "TOO_CLOSE"
    GLANCING = "GLANCING"
    NO_KEY = "NO_KEY"
    MOVING = "MOVING"
    DEBOUNCE = "DEBOUNCE"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class PressResult:
    """What one evaluation of the ladder found.

    ``accepted`` is ``reason is Reason.ACCEPTED``. The remaining fields are
    filled as far as the ladder got before stopping, ``None`` beyond that:

      range, incidence : known once the ray meets the plane (every reason
                         except NO_INTERSECT). ``range`` is the stroke length
                         ``t`` in metres; ``incidence`` is in radians.
      board_xy         : known once the hit point is mapped into the board
                         frame (NO_KEY and later). Board metres, (x, y).
      key              : the registered key (MOVING, DEBOUNCE, ACCEPTED);
                         ``None`` whenever no key was hit or looked up.
    """

    accepted: bool
    reason: Reason
    key: Key | None
    board_xy: tuple[float, float] | None
    range: float | None
    incidence: float | None

    def __post_init__(self) -> None:
        if self.accepted != (self.reason is Reason.ACCEPTED):
            raise ValueError(
                f"accepted={self.accepted} is inconsistent with reason {self.reason}"
            )
        if self.accepted and self.key is None:
            raise ValueError("an ACCEPTED result must carry the key it registered")


def ray_board_intersection(
    arm_pose: ArmPose, board_pose: BoardPose
) -> tuple[float, np.ndarray, float] | None:
    """Intersect the stylus ray with the panel plane (ladder step 1).

    Returns ``(t, point_world, incidence)`` -- stroke length along the unit
    aim, the hit point ``p_head + t * aim`` in world, and the angle between
    the aim and the inward normal ``+Z_B`` -- or ``None`` when the ray is
    parallel to the face, points away from it, or the plane lies behind the
    head (``t <= 0``). Also drives the dashboard's live reticle dot.

    Raises ``ValueError`` if ``p_head`` or ``aim`` is non-finite, or ``aim``
    is not unit within ``UNIT_AIM_TOL``.
    """
    o = np.asarray(arm_pose.p_head, dtype=float)
    d = np.asarray(arm_pose.aim, dtype=float)
    _require_finite(o, "ArmPose.p_head")
    _require_finite(d, "ArmPose.aim")
    norm = float(np.linalg.norm(d))
    if abs(norm - 1.0) > UNIT_AIM_TOL:
        raise ValueError(
            f"ArmPose.aim must be a unit vector, got |aim| = {norm!r}; "
            "the stroke length is only metres along a unit aim"
        )
    n = board_pose.normal_toward_arm
    p0 = board_pose.t

    denom = float(d @ n)
    if denom >= -PARALLEL_EPS:
        return None
    t = float(((p0 - o) @ n) / denom)
    if t <= 0.0:
        return None

    point = o + t * d
    incidence = float(np.arccos(np.clip(-denom, -1.0, 1.0)))
    return t, point, incidence


def evaluate_press(
    arm_pose: ArmPose,
    qd_measured: np.ndarray,
    board_pose: BoardPose,
    keymap: KeyMap,
    arm_cfg: ArmConfig,
    press_cfg: PressConfig,
    t_now: float,
    t_last_accepted: float | None,
) -> PressResult:
    """Run the design spec section 7 ladder and return the first failure.

    ``qd_measured`` is the plant's reported joint velocity, shape ``(NJ,)``;
    its infinity norm is compared with ``press_cfg.max_joint_speed``.
    ``t_last_accepted`` is the time of the previous ACCEPTED press, or
    ``None`` when none has been accepted yet (debounce then cannot fire).

    Raises ``ValueError`` -- before any rung is evaluated -- for a
    ``qd_measured`` of the wrong shape or with a non-finite entry, a
    non-finite ``t_now``, a NaN or ``+inf`` ``t_last_accepted``, or an
    ``ArmPose`` that ``ray_board_intersection`` rejects.
    """
    qd = np.asarray(qd_measured, dtype=float)
    if qd.shape != (NJ,):
        raise ValueError(f"qd_measured must have shape ({NJ},), got {qd.shape}")
    _require_finite(qd, "qd_measured")
    t_now = float(t_now)
    if not math.isfinite(t_now):
        raise ValueError(f"t_now must be finite, got {t_now}")
    if t_last_accepted is not None:
        t_last_accepted = float(t_last_accepted)
        if math.isnan(t_last_accepted) or t_last_accepted == math.inf:
            raise ValueError(
                f"t_last_accepted must be None, -inf or finite, got {t_last_accepted}"
            )

    # 1. Ray / plane.
    hit = ray_board_intersection(arm_pose, board_pose)
    if hit is None:
        return PressResult(False, Reason.NO_INTERSECT, None, None, None, None)
    t, point, incidence = hit

    # 2. Stroke length.
    if t > arm_cfg.stylus_max:
        return PressResult(False, Reason.OUT_OF_REACH, None, None, t, incidence)
    if t < arm_cfg.stylus_min:
        return PressResult(False, Reason.TOO_CLOSE, None, None, t, incidence)

    # 3. Angle of attack.
    if incidence > press_cfg.max_incidence:
        return PressResult(False, Reason.GLANCING, None, None, t, incidence)

    # 4. Registration.
    b = board_pose.to_board(point)
    board_xy = (float(b[0]), float(b[1]))
    key = keymap.lookup(board_xy[0], board_xy[1])
    if key is None:
        return PressResult(False, Reason.NO_KEY, None, board_xy, t, incidence)

    # 5. Settled?
    if float(np.max(np.abs(qd))) > press_cfg.max_joint_speed:
        return PressResult(False, Reason.MOVING, key, board_xy, t, incidence)

    # 6. Debounce.
    if t_last_accepted is not None and (t_now - t_last_accepted) < press_cfg.debounce:
        return PressResult(False, Reason.DEBOUNCE, key, board_xy, t, incidence)

    return PressResult(True, Reason.ACCEPTED, key, board_xy, t, incidence)


__all__ = [
    "PARALLEL_EPS",
    "UNIT_AIM_TOL",
    "PressResult",
    "Reason",
    "evaluate_press",
    "ray_board_intersection",
]
