"""Tests for autotype_sim.core.plant (the design spec section 3).

Deterministic throughout: noise is either switched off with
``PlantConfig(noise_rel=0.0, noise_abs=0.0)`` or drawn from a seeded
``numpy.random.default_rng``. No wall clock is read anywhere.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autotype_sim.core.config import NJ, ArmConfig, PlantConfig
from autotype_sim.core.plant import Plant, PlantState

QUIET = PlantConfig(noise_rel=0.0, noise_abs=0.0)
TOL = 1e-12


# ---------------------------------------------------------------- helpers


def make_plant(cfg: PlantConfig = QUIET, arm: ArmConfig | None = None, seed: int = 0) -> Plant:
    return Plant(arm if arm is not None else ArmConfig(), cfg, np.random.default_rng(seed))


def unit_cmd(j: int, v: float) -> np.ndarray:
    cmd = np.zeros(NJ)
    cmd[j] = v
    return cmd


def run_loop(plant: Plant, qd_cmd: np.ndarray, n_steps: int, t0: float = 0.0):
    """Drive like a well-behaved node: re-issue the command every tick.

    Returns ``(q, qd_act, qd_measured)`` histories, each ``[n_steps, NJ]``,
    sampled after every step.
    """
    dt = plant.cfg.dt
    q, qa, qm = [], [], []
    for k in range(n_steps):
        t = t0 + k * dt
        plant.command(qd_cmd, t)
        plant.step(t + dt)
        q.append(plant.q)
        qa.append(plant.qd_act)
        qm.append(plant.qd_measured)
    return np.array(q), np.array(qa), np.array(qm)


def random_commands(seed: int, n: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    return g.uniform(-1.0, 1.0, size=(n, NJ)) * ArmConfig().v_max


#: Arbitrary non-zero levels for the reproducibility tests. PlantConfig's own
#: defaults are 0.0, so every noisy test picks its own; the numbers below are
#: chosen to make the statistics converge quickly and mean nothing else.
NOISY = PlantConfig(noise_rel=0.05, noise_abs=0.003)


def noisy_trajectory(seed: int, cmds: np.ndarray) -> np.ndarray:
    plant = make_plant(cfg=NOISY, seed=seed)
    dt = plant.cfg.dt
    out = []
    for k, c in enumerate(cmds):
        plant.command(c, k * dt)
        plant.step((k + 1) * dt)
        out.append(plant.q)
    return np.array(out)


# ---------------------------------------------------- 1. step response


@pytest.mark.parametrize("j", range(NJ))
def test_zero_noise_step_response(j):
    arm = ArmConfig()
    plant = make_plant(arm=arm)
    dt = QUIET.dt
    v_max = arm.v_max[j]
    dv_max = arm.a_max[j] * dt
    n_ramp = math.ceil(v_max / dv_max - 1e-9)
    n = 30  # short enough that no joint reaches a stop from q_home

    q, qa, qm = run_loop(plant, unit_cmd(j, v_max), n)

    # Acceleration limit holds for every joint on every step.
    dqa = np.diff(np.vstack([np.zeros(NJ), qa]), axis=0)
    assert np.all(np.abs(dqa) <= arm.a_max * dt + TOL)

    # Linear ramp at a_max, v_max reached in exactly n_ramp steps, then held.
    k = np.arange(1, n + 1)
    np.testing.assert_allclose(qa[:, j], np.minimum(k * dv_max, v_max), rtol=0, atol=TOL)
    assert qa[n_ramp - 2, j] < v_max - TOL
    assert np.all(np.abs(qa[n_ramp - 1 :, j] - v_max) <= TOL)

    # Other joints are untouched, bit for bit.
    others = np.arange(NJ) != j
    assert np.all(qa[:, others] == 0.0)
    assert np.all(q[:, others] == arm.q_home[others])

    # q integrates qd_act * dt; with noise off, measured == actual.
    np.testing.assert_allclose(
        q[:, j], arm.q_home[j] + dt * np.cumsum(qa[:, j]), rtol=0, atol=1e-11
    )
    np.testing.assert_allclose(qm, qa, rtol=0, atol=TOL)
    assert np.all(q[:, j] < arm.q_max[j])


# ---------------------------------------------------- 2. v_max clamp


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("j", range(NJ))
def test_overspeed_command_is_clamped_to_v_max(j, sign):
    arm = ArmConfig()
    v_max = arm.v_max[j]
    n = 30

    plant = make_plant(arm=arm)
    q, qa, qm = run_loop(plant, unit_cmd(j, sign * 10.0 * v_max), n)
    assert np.all(np.abs(qa[:, j]) <= v_max + TOL)
    assert np.all(np.abs(qm[:, j]) <= v_max + TOL)
    assert abs(qa[-1, j] - sign * v_max) <= TOL
    assert np.all(plant.qd_cmd == unit_cmd(j, sign * 10.0 * v_max))  # command kept raw

    # Bit-identical to commanding exactly v_max.
    ref = make_plant(arm=arm)
    q_ref, qa_ref, _ = run_loop(ref, unit_cmd(j, sign * v_max), n)
    assert np.array_equal(q, q_ref)
    assert np.array_equal(qa, qa_ref)


# ---------------------------------------------------- 3. joint limits


def test_joint_limit_stops_exactly_and_zeroes_velocity():
    arm = ArmConfig()
    dt = QUIET.dt
    j, other = 3, 0
    n = 20

    q0 = arm.q_home.copy()
    q0[j] = arm.q_max[j] - 0.05
    plant = make_plant(arm=arm)
    plant.reset(q0)

    cmd = np.zeros(NJ)
    cmd[j] = arm.v_max[j]
    cmd[other] = 0.3
    q, qa, qm = run_loop(plant, cmd, n)

    hit = int(np.argmax(q[:, j] >= arm.q_max[j]))
    assert 0 < hit < n - 1, "test must both reach the stop and keep pushing"
    assert np.all(q[:, j] <= arm.q_max[j])
    assert q[hit, j] == arm.q_max[j]  # lands exactly on the stop ...
    assert np.all(q[hit:, j] == arm.q_max[j])  # ... and stays there while pushed
    assert qa[hit - 1, j] > 0.0
    assert np.all(qa[hit:, j] == 0.0)  # actuator velocity zeroed at the stop
    dv_max = arm.a_max[j] * dt
    assert 0.0 < qm[hit, j] <= min((hit + 1) * dv_max, arm.v_max[j])  # partial last step
    assert np.all(qm[hit + 1 :, j] == 0.0)

    # The other driven joint is unaffected: identical to a plant driving it alone.
    twin = make_plant(arm=arm)
    twin.reset(q0)
    q_t, qa_t, qm_t = run_loop(twin, unit_cmd(other, 0.3), n)
    assert np.array_equal(q[:, other], q_t[:, other])
    assert np.array_equal(qa[:, other], qa_t[:, other])
    assert np.array_equal(qm[:, other], qm_t[:, other])
    assert qa[-1, other] == pytest.approx(0.3, abs=TOL)

    # Idle joints never move.
    rest = ~np.isin(np.arange(NJ), [j, other])
    assert np.all(q[:, rest] == q0[rest])
    assert np.all(qa[:, rest] == 0.0)

    # Commanding away from the stop releases it on the very next step.
    plant.command(unit_cmd(j, -arm.v_max[j]), n * dt)
    plant.step((n + 1) * dt)
    assert plant.q[j] < arm.q_max[j]
    assert plant.qd_act[j] == pytest.approx(-dv_max, abs=TOL)


def test_lower_limit_symmetry():
    arm = ArmConfig()
    j = 4
    q0 = arm.q_home.copy()
    q0[j] = arm.q_min[j] + 0.03
    plant = make_plant(arm=arm)
    plant.reset(q0)
    q, qa, _ = run_loop(plant, unit_cmd(j, -arm.v_max[j]), 20)
    assert np.all(q[:, j] >= arm.q_min[j])
    assert q[-1, j] == arm.q_min[j]
    assert qa[-1, j] == 0.0


# ---------------------------------------------------- 4. watchdog


def test_watchdog_zeroes_target_after_silence():
    arm = ArmConfig()
    cfg = QUIET
    dt = cfg.dt
    j = 3
    dv_max = arm.a_max[j] * dt
    plant = make_plant(cfg=cfg, arm=arm)

    cmd = unit_cmd(j, arm.v_max[j])
    plant.command(cmd, 0.0)
    assert plant.t_last_cmd == 0.0
    np.testing.assert_array_equal(plant.target_velocity(0.08), cmd)
    assert not plant.watchdog_expired(0.08)
    assert plant.watchdog_expired(cfg.watchdog + 1e-9)
    np.testing.assert_array_equal(plant.target_velocity(cfg.watchdog + 1e-9), np.zeros(NJ))

    # t = 0.02 .. 0.08: inside the window, tracking the ramp.
    for k in range(1, 5):
        plant.step(k * dt)
        assert plant.qd_act[j] == pytest.approx(k * dv_max, abs=TOL)

    # t = 0.10 is the boundary (strict inequality); take the step, don't judge it.
    plant.step(5 * dt)
    v_edge = plant.qd_act[j]
    assert v_edge > 0.0

    # t > 0.10 with no new command: target 0, decay at a_max, then rest.
    qa_hist, q_hist = [], []
    prev = v_edge
    for k in range(6, 26):
        plant.step(k * dt)
        v = plant.qd_act[j]
        assert -dv_max - TOL <= v - prev <= TOL
        assert v >= -TOL
        prev = v
        qa_hist.append(v)
        q_hist.append(plant.q[j])
    qa_hist, q_hist = np.array(qa_hist), np.array(q_hist)

    n_decay = math.ceil(v_edge / dv_max - 1e-9)
    assert np.all(np.abs(qa_hist[n_decay - 1 :]) <= TOL)  # at rest and staying there
    assert np.all(np.diff(q_hist) >= -TOL)  # never runs backwards
    assert np.all(np.abs(q_hist[n_decay - 1 :] - q_hist[-1]) <= TOL)  # q stops growing
    assert np.all(plant.qd_act[np.arange(NJ) != j] == 0.0)

    # A fresh command re-arms tracking.
    plant.command(cmd, 26 * dt)
    plant.step(27 * dt)
    assert plant.qd_act[j] == pytest.approx(dv_max, abs=TOL)


def test_watchdog_does_not_fire_while_commands_keep_arriving():
    plant = make_plant()
    arm, dt = plant.arm, plant.cfg.dt
    j = 3
    cmd = unit_cmd(j, arm.v_max[j])
    for k in range(30):
        t = k * dt
        if k % 4 == 0:  # gaps of 0.08 s < watchdog 0.10 s
            plant.command(cmd, t)
        plant.step(t + dt)
    assert plant.qd_act[j] == pytest.approx(arm.v_max[j], abs=TOL)


def test_never_commanded_plant_stays_parked():
    arm = ArmConfig()
    plant = make_plant(arm=arm)
    assert plant.t_last_cmd == -np.inf
    for k in range(1, 6):
        plant.step(k * plant.cfg.dt)
    assert np.array_equal(plant.q, arm.q_home)
    assert np.all(plant.qd_act == 0.0)
    assert np.all(plant.qd_measured == 0.0)


# ---------------------------------------------------- 5. reproducibility


def test_reproducible_from_seed():
    cmds = random_commands(11, 200)
    a = noisy_trajectory(7, cmds)
    b = noisy_trajectory(7, cmds)
    assert a.dtype == np.float64
    assert np.array_equal(a, b)
    c = noisy_trajectory(8, cmds)
    assert not np.array_equal(a, c)


# ---------------------------------------------------- 6. noise statistics


def test_relative_noise_statistics():
    # Wide limits so one joint can run at v_max for 400 s without a stop.
    arm = ArmConfig(q_min=np.full(NJ, -1e3), q_max=np.full(NJ, 1e3))
    cfg = PlantConfig(noise_rel=0.05, noise_abs=0.0)
    plant = Plant(arm, cfg, np.random.default_rng(123))
    j, n, skip = 0, 20000, 100  # skip covers the 20-step accel ramp

    _, qa, qm = run_loop(plant, unit_cmd(j, arm.v_max[j]), n)
    np.testing.assert_allclose(qa[skip:, j], arm.v_max[j], rtol=0, atol=TOL)

    ratio = qm[skip:, j] / qa[skip:, j] - 1.0
    assert abs(ratio.std() - 0.05) <= 0.15 * 0.05
    assert abs(ratio.mean()) <= 0.05 * 0.05

    # With noise_abs = 0 the idle joints report exactly zero.
    others = np.arange(NJ) != j
    assert np.all(qm[:, others] == 0.0)


def test_absolute_noise_statistics_on_idle_joints():
    cfg = PlantConfig(noise_rel=0.0, noise_abs=0.003)
    plant = make_plant(cfg=cfg, seed=5)
    _, qa, qm = run_loop(plant, np.zeros(NJ), 20000)
    assert np.all(qa == 0.0)
    std = qm.std(axis=0)
    assert np.all(np.abs(std - 0.003) <= 0.15 * 0.003)
    assert np.all(np.abs(qm.mean(axis=0)) <= 0.05 * 0.003)


# ---------------------------------------------------- 7. qd_measured


def test_qd_measured_is_finite_difference_of_q():
    arm = ArmConfig()
    plant = make_plant(cfg=PlantConfig(), seed=3)  # noise on
    dt = plant.cfg.dt
    q0 = arm.q_home.copy()
    q0[4] = arm.q_max[4] - 0.01  # joint 4 slams into its stop early on
    plant.reset(q0)

    cmds = random_commands(21, 300)
    cmds[:, 4] = arm.v_max[4]
    touched_stop = False
    for k, c in enumerate(cmds):
        q_old = plant.q
        plant.command(c, k * dt)
        plant.step((k + 1) * dt)
        np.testing.assert_allclose(
            plant.qd_measured, (plant.q - q_old) / dt, rtol=0, atol=TOL
        )
        touched_stop |= bool(plant.q[4] == arm.q_max[4])
    assert touched_stop


# ---------------------------------------------------- validation & hygiene


def test_command_validation():
    plant = make_plant()
    with pytest.raises(ValueError):
        plant.command(np.zeros(NJ - 1), 0.0)
    with pytest.raises(ValueError):
        plant.command(np.zeros((NJ, 1)), 0.0)
    bad = np.zeros(NJ)
    bad[2] = np.nan
    with pytest.raises(ValueError):
        plant.command(bad, 0.0)
    bad[2] = np.inf
    with pytest.raises(ValueError):
        plant.command(bad, 0.0)
    with pytest.raises(ValueError):
        plant.command(np.zeros(NJ), float("nan"))
    # Rejected commands leave the plant untouched.
    assert plant.t_last_cmd == -np.inf
    assert np.all(plant.qd_cmd == 0.0)
    # Plain sequences are accepted and copied.
    plant.command([0.1] * NJ, 1.5)
    assert plant.t_last_cmd == 1.5
    assert np.all(plant.qd_cmd == 0.1)


def test_constructor_reset_and_step_validation():
    arm = ArmConfig()
    with pytest.raises(TypeError):
        Plant(arm, QUIET, np.random.RandomState(0))
    plant = make_plant(arm=arm)
    with pytest.raises(ValueError):
        plant.reset(arm.q_max + 0.1)
    with pytest.raises(ValueError):
        plant.reset(np.zeros(3))
    with pytest.raises(ValueError):
        plant.step(float("inf"))
    # A rejected reset leaves the plant untouched.
    assert np.array_equal(plant.q, arm.q_home)


def test_initial_state_and_reset():
    arm = ArmConfig()
    plant = make_plant(arm=arm)
    assert np.array_equal(plant.q, arm.q_home)
    assert plant.q.dtype == np.float64
    assert np.all(plant.qd_act == 0.0)
    assert np.all(plant.qd_measured == 0.0)
    assert plant.t_last_cmd == -np.inf

    run_loop(plant, 0.5 * arm.v_max, 10)
    assert np.all(plant.qd_act != 0.0)
    assert plant.t_last_cmd > 0.0

    q0 = arm.q_home + 0.1
    plant.reset(q0)
    assert np.array_equal(plant.q, q0)
    assert np.all(plant.qd_act == 0.0)
    assert np.all(plant.qd_measured == 0.0)
    assert np.all(plant.qd_cmd == 0.0)
    assert plant.t_last_cmd == -np.inf

    plant.reset()
    assert np.array_equal(plant.q, arm.q_home)


def test_state_accessors_return_copies():
    plant = make_plant()
    home = plant.arm.q_home.copy()

    q = plant.q
    q[:] = 99.0
    assert np.array_equal(plant.q, home)

    cmd = np.ones(NJ)
    plant.command(cmd, 0.0)
    cmd[:] = 5.0
    assert np.all(plant.qd_cmd == 1.0)

    for name in ("qd_act", "qd_measured", "qd_cmd"):
        arr = getattr(plant, name)
        arr[:] = 42.0
        assert not np.any(getattr(plant, name) == 42.0)

    s = plant.state
    assert isinstance(s, PlantState)
    assert s.t_last_cmd == 0.0
    s.q[:] = 7.0
    assert np.array_equal(plant.q, home)
    assert np.array_equal(s.qd_cmd, np.ones(NJ))
