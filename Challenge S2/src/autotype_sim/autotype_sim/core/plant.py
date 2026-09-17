"""Velocity-integration plant: the actuator model behind ``/joint_states``.

Implements the design spec section 3. Per joint the plant keeps the true position
``q`` and an internal actuator velocity ``qd_act``. Every ``step(t_now)`` --
one tick of ``PlantConfig.dt`` -- does, in this order:

    target  = 0                              if t_now - t_last_cmd > watchdog
            = clip(qd_cmd, -v_max, v_max)    otherwise
    dv      = clip(target - qd_act, -a_max*dt, a_max*dt)
    qd_act += dv
    qd_eff  = qd_act * (1 + N(0, noise_rel)) + N(0, noise_abs)    # per joint
    q_new   = clip(q + qd_eff*dt, q_min, q_max)
    qd_act  = 0 at every joint that hit a limit
    qd_measured = (q_new - q) / dt                                 # /joint_states

Conventions this module commits to:

  * All randomness comes from the ``numpy.random.Generator`` injected at
    construction: two ``standard_normal(NJ)`` draws per step, the relative
    term first, then the absolute one -- always, even when a noise level is
    zero -- so a trajectory is reproducible from its seed alone.
  * There is no latency and no encoder noise. ``q`` is the true position.
  * ``t_last_cmd`` is ``-inf`` until the first command, so a plant that has
    never been commanded holds a zero target through the watchdog rule.
  * A joint "hit a limit" when its clipped position lies on the stop
    (``q_new <= q_min`` or ``q_new >= q_max``): a joint resting on a hard
    stop carries no velocity into it, and it may leave again immediately.
  * Clamping to ``v_max`` happens in ``step`` (as written in the design spec), so
    ``qd_cmd`` records what the member actually asked for.
  * Public state accessors return copies; nothing outside can alias the
    plant's arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from autotype_sim.core.config import NJ, ArmConfig, PlantConfig


@dataclass(frozen=True)
class PlantState:
    """Atomic snapshot of the plant, for /joint_states and dashboards.

    ``q`` and ``qd_measured`` are what members see; ``qd_act`` and ``qd_cmd``
    are internal. All arrays are copies, shape ``(NJ,)``, float64.
    """

    q: np.ndarray
    qd_act: np.ndarray
    qd_measured: np.ndarray
    qd_cmd: np.ndarray
    t_last_cmd: float


def _as_joint_vector(x, name: str) -> np.ndarray:
    """Validate and copy a per-joint vector: shape ``(NJ,)``, finite, float64."""
    arr = np.array(x, dtype=float)
    if arr.shape != (NJ,):
        raise ValueError(f"{name} must have shape ({NJ},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {arr}")
    return arr


def _as_time(t, name: str) -> float:
    """Validate a timestamp: a finite float in seconds."""
    t = float(t)
    if not math.isfinite(t):
        raise ValueError(f"{name} must be finite, got {t}")
    return t


class Plant:
    """Five-joint velocity-mode actuator: rate and acceleration limits, hard
    stops, multiplicative plus additive velocity noise, and a command
    watchdog. See the module docstring for the per-step update."""

    def __init__(
        self, arm: ArmConfig, plant_cfg: PlantConfig, rng: np.random.Generator
    ) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "rng must be a numpy.random.Generator "
                "(build one with numpy.random.default_rng(seed))"
            )
        self.arm = arm
        self.cfg = plant_cfg
        self._rng = rng
        self._dt = float(plant_cfg.dt)
        self._q = np.zeros(NJ)
        self._qd_act = np.zeros(NJ)
        self._qd_measured = np.zeros(NJ)
        self._qd_cmd = np.zeros(NJ)
        self._t_last_cmd = -np.inf
        self.reset()

    # ------------------------------------------------------------------ state

    @property
    def q(self) -> np.ndarray:
        """True joint positions (copy). Reported verbatim on /joint_states."""
        return self._q.copy()

    @property
    def qd_measured(self) -> np.ndarray:
        """``(q_new - q_old) / dt`` from the last step (copy). Reported on
        /joint_states; zero until the first step."""
        return self._qd_measured.copy()

    @property
    def qd_act(self) -> np.ndarray:
        """Internal actuator velocity after the last step (copy). Not
        published to members."""
        return self._qd_act.copy()

    @property
    def qd_cmd(self) -> np.ndarray:
        """Last latched command, unclipped (copy). Zero until commanded."""
        return self._qd_cmd.copy()

    @property
    def t_last_cmd(self) -> float:
        """Timestamp of the last accepted command; ``-inf`` if none yet."""
        return self._t_last_cmd

    @property
    def state(self) -> PlantState:
        """Copy of everything at once, so /joint_states can read q and
        qd_measured from the same tick."""
        return PlantState(
            q=self.q,
            qd_act=self.qd_act,
            qd_measured=self.qd_measured,
            qd_cmd=self.qd_cmd,
            t_last_cmd=self._t_last_cmd,
        )

    # ---------------------------------------------------------------- control

    def reset(self, q0: np.ndarray | None = None) -> None:
        """Park at rest at ``q0`` (default ``arm.q_home``) and forget any
        command, so the watchdog holds a zero target until the next one.

        ``q0`` must lie within the joint limits; it is not clipped.
        """
        if q0 is None:
            q = np.array(self.arm.q_home, dtype=float)
        else:
            q = _as_joint_vector(q0, "q0")
            if np.any(q < self.arm.q_min) or np.any(q > self.arm.q_max):
                raise ValueError("q0 must lie within the joint limits")
        self._q = q
        self._qd_act = np.zeros(NJ)
        self._qd_measured = np.zeros(NJ)
        self._qd_cmd = np.zeros(NJ)
        self._t_last_cmd = -np.inf

    def command(self, qd_cmd: np.ndarray, t: float) -> None:
        """Latch a joint-velocity setpoint stamped ``t`` (seconds).

        Validates shape ``(NJ,)`` and finiteness; a rejected command leaves the
        plant untouched. Clamping to ``v_max`` is applied in ``step``.
        """
        qd_cmd = _as_joint_vector(qd_cmd, "qd_cmd")
        t = _as_time(t, "t")
        self._qd_cmd = qd_cmd
        self._t_last_cmd = t

    def watchdog_expired(self, t_now: float) -> bool:
        """True iff ``t_now - t_last_cmd > watchdog`` (strict, per DESIGN 3)."""
        return t_now - self._t_last_cmd > self.cfg.watchdog

    def target_velocity(self, t_now: float) -> np.ndarray:
        """Setpoint the actuator tracks at ``t_now``: zero once the watchdog
        has expired, otherwise the latched command clipped to ``+-v_max``."""
        if self.watchdog_expired(t_now):
            return np.zeros(NJ)
        return np.clip(self._qd_cmd, -self.arm.v_max, self.arm.v_max)

    def step(self, t_now: float) -> None:
        """Advance one tick of ``cfg.dt`` seconds, ending at ``t_now``.

        ``t_now`` only feeds the watchdog; the integration step is always
        ``1/rate``. Updates ``q``, ``qd_act`` and ``qd_measured``.
        """
        t_now = _as_time(t_now, "t_now")
        arm = self.arm
        dt = self._dt

        target = self.target_velocity(t_now)
        dv_max = arm.a_max * dt
        dv = np.clip(target - self._qd_act, -dv_max, dv_max)
        qd_act = self._qd_act + dv

        n_rel = self._rng.standard_normal(NJ)
        n_abs = self._rng.standard_normal(NJ)
        qd_eff = qd_act * (1.0 + self.cfg.noise_rel * n_rel) + self.cfg.noise_abs * n_abs

        q_new = np.clip(self._q + qd_eff * dt, arm.q_min, arm.q_max)
        hit = (q_new <= arm.q_min) | (q_new >= arm.q_max)
        qd_act[hit] = 0.0

        self._qd_measured = (q_new - self._q) / dt
        self._q = q_new
        self._qd_act = qd_act
