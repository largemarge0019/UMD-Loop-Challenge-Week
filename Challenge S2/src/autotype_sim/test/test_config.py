"""Tests for autotype_sim.core.config.

The configs are frozen value objects: every field is validated once at
construction (the design spec s.6 -- a mis-tuned configuration is a bug that raises,
not a runtime condition), the ndarray fields are private read-only copies so
the s.2/s.3 invariants checked at construction hold for the object's
lifetime, and equality/hashing work by value.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from autotype_sim.core.config import (
    DEG,
    NJ,
    ArmConfig,
    CameraConfig,
    PlantConfig,
    PressConfig,
    SimConfig,
)

ARRAY_FIELDS = ("q_min", "q_max", "v_max", "a_max", "q_home")


# ---------------------------------------------------------------- defaults


def test_defaults_construct_and_match_design_home_pose():
    arm = ArmConfig()
    assert np.allclose(arm.q_home, np.array([0.0, 85.0, -113.0, 0.0, 0.0]) * DEG)
    # DESIGN s.2: every joint keeps >= 0.26 rad of travel to its nearest stop.
    assert np.all(arm.q_home - arm.q_min >= 0.26)
    assert np.all(arm.q_max - arm.q_home >= 0.26)
    sim = SimConfig()
    assert sim.plant.dt == pytest.approx(0.02)
    assert sim.camera.cx == pytest.approx(639.5)
    assert sim.camera.K[0, 0] == 900.0


# --------------------------------------------------------- array ownership


def test_array_fields_are_private_copies_of_caller_arrays():
    home = ArmConfig().q_home.copy()  # already float64: np.asarray would alias it
    cfg = ArmConfig(q_home=home)
    assert cfg.q_home is not home
    assert not np.shares_memory(cfg.q_home, home)
    home[1] = 5.0  # far outside q_max; must not leak into the config
    assert np.all(cfg.q_home <= cfg.q_max)
    assert cfg.q_home[1] == pytest.approx(85.0 * DEG)


@pytest.mark.parametrize("name", ARRAY_FIELDS)
def test_array_fields_are_read_only(name):
    cfg = ArmConfig()
    arr = getattr(cfg, name)
    assert arr.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        arr[0] = -10.0
    with pytest.raises(ValueError, match="read-only"):
        arr[:] = 0.0
    # Still within the validated invariants afterwards.
    assert np.all(cfg.q_min < cfg.q_max)
    assert np.all((cfg.q_min <= cfg.q_home) & (cfg.q_home <= cfg.q_max))


def test_arrays_are_float64_even_from_int_lists():
    cfg = ArmConfig(q_min=[-2, -1, -3, -1, -1])
    assert cfg.q_min.dtype == np.float64
    assert cfg.q_min.shape == (NJ,)


def test_dataclasses_replace_works_with_locked_arrays():
    base = ArmConfig()
    tweaked = dataclasses.replace(base, forearm=0.5)
    assert tweaked.forearm == 0.5
    assert np.array_equal(tweaked.q_home, base.q_home)
    assert tweaked.q_home is not base.q_home
    assert tweaked.q_home.flags.writeable is False


def test_copies_of_config_arrays_are_writable():
    # Callers that need a scratch pose copy first (as plant.reset does).
    q = np.array(ArmConfig().q_home)
    q[3] = 0.1
    assert q[3] == 0.1


# ------------------------------------------------------------ eq and hash


def test_arm_config_equality_and_hash_by_value():
    a, b = ArmConfig(), ArmConfig()
    assert a == b
    assert not (a != b)
    assert hash(a) == hash(b)
    c = ArmConfig(forearm=0.41)
    d = ArmConfig(v_max=ArmConfig().v_max * 2.0)
    assert a != c and a != d
    assert len({a, b, c, d}) == 3
    assert {a: "home"}[b] == "home"


def test_arm_config_compares_unequal_to_other_types():
    assert ArmConfig() != object()
    assert ArmConfig() != PlantConfig()
    assert (ArmConfig() == 3) is False


def test_sim_config_equality_and_hash():
    assert SimConfig() == SimConfig()
    assert hash(SimConfig()) == hash(SimConfig())
    assert SimConfig() != SimConfig(plant=PlantConfig(rate=100.0))
    assert SimConfig() != SimConfig(arm=ArmConfig(upper_arm=0.5))
    assert len({SimConfig(), SimConfig()}) == 1


def test_scalar_only_configs_keep_generated_equality():
    assert PlantConfig() == PlantConfig()
    assert CameraConfig() == CameraConfig()
    assert PressConfig() == PressConfig()
    assert PlantConfig(rate=50) == PlantConfig(rate=50.0)
    assert hash(PlantConfig(rate=50)) == hash(PlantConfig(rate=50.0))


# ------------------------------------------------------ ArmConfig rejects


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"q_min": [np.nan] * NJ}, "q_min must be finite"),
        ({"q_max": [np.inf] * NJ}, "q_max must be finite"),
        ({"q_home": [np.nan, 0.0, 0.0, 0.0, 0.0]}, "q_home must be finite"),
        ({"v_max": [np.nan, 1, 1, 1, 1]}, "v_max must be finite"),
        ({"v_max": -np.ones(NJ)}, "v_max must be > 0"),
        ({"v_max": [0.0, 1, 1, 1, 1]}, "v_max must be > 0"),
        ({"a_max": np.zeros(NJ)}, "a_max must be > 0"),
        ({"a_max": -np.ones(NJ)}, "a_max must be > 0"),
        ({"q_min": np.zeros(3)}, "shape"),
        ({"q_min": np.zeros((NJ, 1))}, "shape"),
        (
            {
                "q_min": np.array([0, -30, -140, -45, -35]) * DEG,
                "q_max": np.array([0, 100, 0, 45, 35]) * DEG,
            },
            "strictly below",
        ),
        ({"q_home": np.array([0.0, 101.0, -113.0, 0.0, 0.0]) * DEG}, "within the joint limits"),
        ({"q_home": np.array([-121.0, 85.0, -113.0, 0.0, 0.0]) * DEG}, "within the joint limits"),
        ({"base_height": np.nan}, "base_height must be finite"),
        ({"base_height": -0.1}, "base_height must be >= 0"),
        ({"upper_arm": -1.0}, "upper_arm must be > 0"),
        ({"upper_arm": 0.0}, "upper_arm must be > 0"),
        ({"forearm": np.inf}, "forearm must be finite"),
        ({"forearm": 0.0}, "forearm must be > 0"),
        ({"stylus_min": np.nan}, "stylus_min must be finite"),
        ({"stylus_max": np.inf}, "stylus_max must be finite"),
        ({"stylus_min": 0.0}, "0 < stylus_min < stylus_max"),
        ({"stylus_min": 0.4, "stylus_max": 0.35}, "0 < stylus_min < stylus_max"),
    ],
)
def test_arm_config_rejects_non_physical_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ArmConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_height": "0.3"},
        {"upper_arm": None},
        {"forearm": True},
        {"stylus_max": np.array([0.3, 0.4])},
    ],
)
def test_arm_config_rejects_non_real_scalars(kwargs):
    with pytest.raises(TypeError):
        ArmConfig(**kwargs)


def test_arm_config_rejects_non_numeric_arrays():
    with pytest.raises((ValueError, TypeError)):
        ArmConfig(q_min=["a", "b", "c", "d", "e"])


def test_arm_config_still_accepts_legitimate_edge_values():
    # Home pose resting on a stop is allowed (the checks are strict overshoot).
    on_stop = ArmConfig(q_home=ArmConfig().q_min)
    assert np.array_equal(on_stop.q_home, on_stop.q_min)
    # Shoulder on the ground and very wide (finite) limits, as test_plant uses.
    ArmConfig(base_height=0.0)
    wide = ArmConfig(q_min=np.full(NJ, -1e3), q_max=np.full(NJ, 1e3))
    assert wide.q_max[0] == 1e3
    # numpy scalars are real numbers.
    assert ArmConfig(forearm=np.float64(0.4)).forearm == 0.4


# --------------------------------------------------- PlantConfig rejects


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"rate": 0.0}, "rate must be > 0"),
        ({"rate": -50.0}, "rate must be > 0"),
        ({"rate": math.inf}, "rate must be finite"),
        ({"rate": math.nan}, "rate must be finite"),
        ({"noise_rel": -0.01}, "noise_rel must be >= 0"),
        ({"noise_rel": math.nan}, "noise_rel must be finite"),
        ({"noise_abs": -1e-3}, "noise_abs must be >= 0"),
        ({"noise_abs": math.inf}, "noise_abs must be finite"),
        ({"watchdog": 0.0}, "watchdog must be > 0"),
        ({"watchdog": -1.0}, "watchdog must be > 0"),
        ({"watchdog": math.inf}, "watchdog must be finite"),
    ],
)
def test_plant_config_rejects_non_physical_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PlantConfig(**kwargs)


def test_plant_config_rejects_non_real_scalars():
    with pytest.raises(TypeError):
        PlantConfig(rate="50")


def test_plant_config_accepts_quiet_plant_and_integer_rate():
    quiet = PlantConfig(noise_rel=0.0, noise_abs=0.0)
    assert quiet.noise_rel == 0.0 and quiet.noise_abs == 0.0
    assert PlantConfig(rate=100).dt == pytest.approx(0.01)


# -------------------------------------------------- CameraConfig rejects


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"width": 0}, ValueError),
        ({"height": -720}, ValueError),
        ({"width": 1280.5}, TypeError),
        ({"width": 1280.0}, TypeError),
        ({"height": True}, TypeError),
        ({"fx": 0.0}, ValueError),
        ({"fy": -900.0}, ValueError),
        ({"fx": math.nan}, ValueError),
        ({"fy": math.inf}, ValueError),
        ({"fx": "900"}, TypeError),
    ],
)
def test_camera_config_rejects_non_physical_values(kwargs, exc):
    with pytest.raises(exc):
        CameraConfig(**kwargs)


def test_camera_config_accepts_numpy_integer_dimensions():
    cam = CameraConfig(width=np.int64(640), height=np.int32(480))
    assert cam.cx == pytest.approx(319.5) and cam.cy == pytest.approx(239.5)
    assert cam.hfov == pytest.approx(2.0 * math.atan2(320.0, 900.0))


# --------------------------------------------------- PressConfig rejects


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"max_incidence": math.nan}, "max_incidence must be finite"),
        ({"max_incidence": -1e-3}, "max_incidence must be >= 0"),
        ({"max_joint_speed": -0.02}, "max_joint_speed must be >= 0"),
        ({"max_joint_speed": math.inf}, "max_joint_speed must be finite"),
        ({"debounce": -1}, "debounce must be >= 0"),
        ({"debounce": math.nan}, "debounce must be finite"),
    ],
)
def test_press_config_rejects_non_physical_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PressConfig(**kwargs)


def test_press_config_accepts_zero_thresholds():
    strict = PressConfig(max_incidence=0.0, max_joint_speed=0.0, debounce=0.0)
    assert strict.debounce == 0.0
    assert PressConfig(max_incidence=np.float64(0.5)).max_incidence == 0.5
