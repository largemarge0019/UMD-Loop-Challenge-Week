"""Tests for ``autotype_sim/sim.py`` (``SimCore``) against the node design
and the dashboard protocol.

Everything is deterministic: time is injected (``t`` arguments), the plant's
noise stream is derived from the episode seed, and the only ``Generator`` here
is seeded. Geometry and key map come from the session fixtures (conftest.py:
the ``board_geometry.yaml`` in private/ when available, else
``DEFAULT_GEOMETRY``); nothing below asserts a keyboard placement.

The parked pose for the press tests is found by a coarse grid over the arm
joints using ``SimCore.reachable_mask`` itself, so the tests stand alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autotype_sim.core.board import marker_corners_world
from autotype_sim.core.config import DEG, JOINT_NAMES, NJ, CameraConfig, SimConfig
from autotype_sim.core.keymap import KeyMap
from autotype_sim.core.kinematics import (
    aim_inverse,
    forward_kinematics,
    in_frame,
    project_points,
)
from autotype_sim.core.press import Reason
from autotype_sim.core.renderer import (
    DEFAULT_PX_PER_M,
    build_texture,
    detect_markers,
    texture_size,
)
from autotype_sim.sim import (
    FINISHED_REASON,
    FRAME_CAMERA_OPTICAL,
    FRAME_HEAD,
    FRAME_STYLUS_TIP,
    FRAME_WORLD,
    PRIVATE_DIR_ENV,
    REACHABLE_PERIOD,
    SimCore,
    apply_private_plant,
    build_default,
    default_texture_photo,
    load_plant_overrides,
    resolve_private_dir,
)

CFG = SimConfig()
ARM = CFG.arm
LAUNCH = "ROVER"
DT = CFG.plant.dt
N_KEYS = 87
MARKER_IDS = {0, 1, 2, 3}

_REPO = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------- #
# The dashboard protocol, written out as literals (the schema under test)
# --------------------------------------------------------------------------- #

STATIC_KEYS = {
    "joint_names", "q_min_deg", "q_max_deg", "v_max_deg_s", "links", "stylus", "press",
    "camera", "panel", "markers", "keys", "texture", "teleop", "launch_key", "seed",
}
STATIC_LINKS_KEYS = {"base_height", "upper_arm", "forearm"}
STATIC_STYLUS_KEYS = {"min", "max"}
STATIC_PRESS_KEYS = {"max_incidence_deg", "max_joint_speed_deg_s", "debounce_s"}
STATIC_CAMERA_KEYS = {"width", "height", "fx", "fy", "cx", "cy"}
STATIC_PANEL_KEYS = {"w", "h"}
STATIC_MARKER_KEYS = {"id", "center", "size"}
STATIC_KEY_KEYS = {"name", "kind", "rect"}
STATIC_TEXTURE_KEYS = {"px_per_m", "w", "h"}

STATE_KEYS = {
    "t", "episode", "q_deg", "qd_deg_s", "cmd_age_s", "watchdog_tripped", "arm", "board",
    "stylus", "reticle_px", "reachable", "typed", "events", "result",
}
STATE_EPISODE_KEYS = {"status", "seed", "launch_key"}
STATE_ARM_KEYS = {"shoulder", "elbow", "head", "aim", "cam_axis"}
STATE_BOARD_KEYS = {"corners", "center"}
STATE_STYLUS_KEYS = {"board_xy", "range", "incidence_deg", "key", "would_accept"}
EVENT_PRESS_KEYS = {"t", "kind", "accepted", "reason", "key", "board_xy"}
EVENT_INFO_KEYS = {"t", "kind", "text"}
RESULT_KEYS = {
    "target", "typed", "exact_match", "edit_distance", "presses_attempted",
    "presses_accepted", "rejections", "elapsed",
}
LADDER_REASONS = {r.value for r in Reason} | {FINISHED_REASON}


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def texture(geometry, keymap) -> np.ndarray:
    return build_texture(geometry, keymap, DEFAULT_PX_PER_M)


@pytest.fixture
def make_sim(geometry, keymap, texture):
    """Factory: ``make_sim(seed=1, launch_key=LAUNCH, t0=0.0) -> SimCore``."""

    def _make(seed: int = 1, launch_key: str = LAUNCH, t0: float = 0.0) -> SimCore:
        return SimCore(CFG, geometry, keymap, launch_key, seed, texture, t0)

    return _make


def _aimed_reticles_in_frame(sim: SimCore, q_park: np.ndarray, keys) -> bool:
    """From ``q_park``, does aiming at each of ``keys`` keep its reticle --
    the key centre seen by the camera at the aimed pose -- inside the image?"""
    for key in keys:
        cx, cy = KeyMap.center(key)
        target = sim.board_pose.to_world(np.array([cx, cy, 0.0]))
        pose = forward_kinematics(ARM, _aimed_q(sim, q_park, key.name))
        uv, z = project_points(CFG.camera, pose.R_cam, pose.t_cam, target)
        if not bool(in_frame(CFG.camera, uv, z)[0]):
            return False
    return True


def _grid_park(sim: SimCore, alnum_idx: list[int]) -> np.ndarray | None:
    """First pose on a 6 deg grid (base yaw within +-45 deg) from which
    ``reachable_mask`` says every alphanumeric key is pressable AND every
    alphanumeric reticle stays in the camera image -- a pose a person would
    actually park at, not merely one the press ladder tolerates."""
    step = 6.0 * DEG
    yaw = np.arange(-45.0 * DEG, 45.0 * DEG + 1e-9, step)
    sho = np.arange(ARM.q_min[1], ARM.q_max[1] + 1e-9, step)
    elb = np.arange(ARM.q_min[2], ARM.q_max[2] + 1e-9, step)
    alnum = list(sim.keymap.alnum())
    for th0 in yaw:
        for th1 in sho:
            for th2 in elb:
                q = np.array([th0, th1, th2, 0.0, 0.0])
                sim.teleport(q)
                mask = sim.reachable_mask()
                if all(mask[i] for i in alnum_idx) and _aimed_reticles_in_frame(sim, q, alnum):
                    return q
    return None


@pytest.fixture(scope="module")
def parked(geometry, keymap, texture) -> tuple[int, np.ndarray]:
    """``(seed, q_park)``: a seed and a parked pose (pan = tilt = 0, inside the
    joint limits) from which all 36 alphanumeric keys are pressable."""
    scratch = SimCore(CFG, geometry, keymap, LAUNCH, 0, texture)
    names = [k.name for k in keymap]
    alnum_idx = [names.index(k.name) for k in keymap.alnum()]
    for seed in range(10):
        scratch.reset(seed=seed)
        q = _grid_park(scratch, alnum_idx)
        if q is not None:
            return seed, q
    pytest.skip("no parked pose found on the coarse grid for seeds 0..9")


def _aimed_q(sim: SimCore, q_park: np.ndarray, key_name: str) -> np.ndarray:
    """``q_park`` with pan / tilt set by ``aim_inverse`` onto ``key_name``'s centre."""
    pose = forward_kinematics(ARM, q_park)
    cx, cy = KeyMap.center(sim.keymap.by_name[key_name])
    target = sim.board_pose.to_world(np.array([cx, cy, 0.0]))
    pan, tilt, _ = aim_inverse(pose.R_head, pose.p_head, target)
    q = np.array(q_park, dtype=float)
    q[3], q[4] = pan, tilt
    return q


def _in_limits(q: np.ndarray) -> bool:
    return bool(np.all(q >= ARM.q_min) and np.all(q <= ARM.q_max))


def _drive(sim: SimCore, t_end: float, names, vels, recommand_every: float = 0.04) -> np.ndarray:
    """Step from t = 0 to ``t_end`` re-issuing the same command; return the q history."""
    hist = []
    n = int(round(t_end / DT))
    sim.command(names, vels, 0.0)
    for i in range(1, n + 1):
        t = i * DT
        if abs((t / recommand_every) - round(t / recommand_every)) < 1e-9:
            sim.command(names, vels, t)
        sim.step(t)
        hist.append(sim.q)
    return np.array(hist)


def _compose_to_world(chain, frame: str) -> tuple[np.ndarray, np.ndarray]:
    """World-from-``frame`` by walking parent links up to ``world``."""
    by_child = {child: (parent, R, t) for parent, child, R, t in chain}
    R_acc, t_acc = np.eye(3), np.zeros(3)  # current-frame-from-target
    while frame != FRAME_WORLD:
        parent, R, t = by_child[frame]
        # p_parent = R p_child + t and p_child = R_acc p_target + t_acc
        t_acc = R @ t_acc + t
        R_acc = R @ R_acc
        frame = parent
    return R_acc, t_acc


# --------------------------------------------------------------------------- #
# 1. reset / seed sequence
# --------------------------------------------------------------------------- #


def test_reset_seed_sequence(make_sim):
    sim = make_sim(seed=5)
    assert sim.seed == 5 and sim.episode_index == 0
    assert sim.reset() == 6 and sim.seed == 6 and sim.episode_index == 1
    assert sim.reset(seed=42) == 42 and sim.seed == 42
    assert sim.reset() == 43 and sim.seed == 43 and sim.episode_index == 3
    assert sim.launch_key == LAUNCH
    assert len(sim.events) == 1 and sim.events[0]["kind"] == "info"
    assert "43" in sim.events[0]["text"]


def test_same_seed_reproduces_board_and_trajectory(make_sim):
    a, b = make_sim(seed=42), make_sim(seed=42)
    corners_a = a.snapshot(0.0)["board"]["corners"]
    assert corners_a == b.snapshot(0.0)["board"]["corners"]
    names = ["head_pan", "shoulder_pitch", "elbow_pitch"]
    vels = [0.3, -0.2, 0.4]
    qa = _drive(a, 0.5, names, vels)
    qb = _drive(b, 0.5, names, vels)
    assert np.array_equal(qa, qb)
    assert not np.allclose(qa[-1], ARM.q_home)  # the arm did move

    # reset(seed) rebuilds the same episode from scratch, plant noise included.
    a.reset(seed=42)
    assert a.snapshot(0.0)["board"]["corners"] == corners_a
    assert np.array_equal(a.q, ARM.q_home)
    assert np.array_equal(_drive(a, 0.5, names, vels), qa)

    # A different seed gives a different board.
    c = make_sim(seed=43)
    assert c.snapshot(0.0)["board"]["corners"] != corners_a


def test_reset_clears_result_and_events(make_sim):
    sim = make_sim(seed=1)
    sim.press(0.5)
    sim.done(1.0)
    assert sim.finished and sim.result is not None and len(sim.events) >= 3
    sim.reset(t0=100.0)
    assert not sim.finished and sim.result is None
    assert len(sim.events) == 1
    assert sim.snapshot(100.0)["t"] == 0.0
    assert sim.snapshot(100.0)["episode"]["status"] == "running"


def test_launch_key_is_normalised_and_validated(make_sim):
    assert make_sim(launch_key="mars").launch_key == "MARS"
    with pytest.raises(ValueError):
        make_sim(launch_key="TOO-LONG")
    with pytest.raises(ValueError):
        make_sim(seed=-1)


# --------------------------------------------------------------------------- #
# 2. command by name
# --------------------------------------------------------------------------- #


def test_command_matches_by_name_and_keeps_omitted_joints(make_sim):
    sim = make_sim()
    assert sim.command(["head_pan", "bogus_joint"], [0.5, 9.0], 0.0) == ["bogus_joint"]
    assert sim.qd_cmd.tolist() == [0.0, 0.0, 0.0, 0.5, 0.0]
    assert sim.command(["elbow_pitch"], [-0.2], 0.02) == []
    assert sim.qd_cmd.tolist() == [0.0, 0.0, -0.2, 0.5, 0.0]
    # Order does not matter; a repeated name takes the last value.
    sim.command(["head_tilt", "base_yaw", "head_tilt"], [0.1, 0.2, 0.3], 0.04)
    assert sim.qd_cmd.tolist() == [0.2, 0.0, -0.2, 0.5, 0.3]

    # The plant follows the latched command through the same path.
    for i in range(1, 11):
        sim.step(0.04 + i * DT)
    assert sim.q[3] > ARM.q_home[3] + 0.02
    assert sim.q[2] < ARM.q_home[2] - 0.01


def test_command_rejects_mismatched_lengths(make_sim):
    sim = make_sim()
    with pytest.raises(ValueError):
        sim.command(["head_pan", "head_tilt"], [0.1], 0.0)
    with pytest.raises(ValueError):
        sim.command(["head_pan"], [0.1, 0.2], 0.0)
    with pytest.raises(ValueError):
        sim.command(["head_pan"], [float("nan")], 0.0)
    assert sim.qd_cmd.tolist() == [0.0] * NJ  # nothing latched


def test_nonfinite_velocity_drops_the_message_even_on_an_unknown_joint(make_sim):
    """INTERFACES.md s.9.5 states its rules 'in this order': rule 1 (drop the
    whole message on a NaN/inf velocity) outranks rule 2 (ignore unknown
    names). A NaN paired with a misspelt joint name must therefore still drop
    the message -- the two failure modes correlate, both coming from a bad
    joint table, and a member who sends one expects a watchdog stop, not a
    moving arm."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        sim = make_sim()
        with pytest.raises(ValueError, match="must be finite"):
            sim.command(["wrist_roll", "base_yaw"], [bad, 0.3], 0.0)
        assert sim.qd_cmd.tolist() == [0.0] * NJ  # base_yaw was NOT applied
    # The unknown name on its own is still merely ignored and reported.
    sim = make_sim()
    assert sim.command(["wrist_roll", "base_yaw"], [0.1, 0.3], 0.0) == ["wrist_roll"]
    assert sim.qd_cmd[0] == 0.3


def test_commands_are_ignored_after_done(make_sim):
    sim = make_sim()
    sim.command(["head_pan"], [0.5], 0.0)
    sim.step(DT)
    sim.done(0.1)
    assert sim.command(["head_pan", "bogus"], [1.0, 1.0], 0.12) == []
    assert sim.qd_cmd.tolist() == [0.0] * NJ
    q_frozen = None
    for i in range(1, 31):  # 0.6 s: more than enough to decelerate to rest
        sim.step(0.1 + i * DT)
        q_frozen = sim.q
    for i in range(31, 41):
        sim.step(0.1 + i * DT)
    assert np.allclose(sim.q, q_frozen, atol=2e-3)
    assert np.all(np.abs(sim.qd_measured) < 0.02)


# --------------------------------------------------------------------------- #
# 3. watchdog through the command path
# --------------------------------------------------------------------------- #


def test_watchdog_through_command_path(make_sim):
    sim = make_sim()
    assert sim.cmd_age(0.0) is None and sim.watchdog_tripped(0.0) is False

    sim.command(["head_pan"], [0.3], 0.0)
    t = 0.0
    while t < 0.08 - 1e-9:
        t += DT
        sim.step(t)
    assert sim.qd_measured[3] > 0.1  # tracking the command
    assert sim.cmd_age(0.08) == pytest.approx(0.08)
    assert sim.watchdog_tripped(0.08) is False

    while t < 0.3 - 1e-9:  # no further command
        t += DT
        sim.step(t)
    assert np.all(np.abs(sim.qd_measured) < 0.02)
    assert sim.cmd_age(0.3) == pytest.approx(0.3)
    assert sim.watchdog_tripped(0.3) is True
    snap = sim.snapshot(0.3)
    assert snap["cmd_age_s"] == pytest.approx(0.3)
    assert snap["watchdog_tripped"] is True
    assert max(abs(v) for v in snap["qd_deg_s"]) < 0.02 / DEG

    # A fresh command re-arms it.
    sim.command(["head_pan"], [0.0], 0.31)
    assert sim.watchdog_tripped(0.32) is False


# --------------------------------------------------------------------------- #
# 4. press bookkeeping
# --------------------------------------------------------------------------- #


def test_press_bookkeeping_and_finished(make_sim, parked):
    seed, q_park = parked
    sim = make_sim(seed=seed)
    names = [k.name for k in sim.keymap]
    t = 1.0
    for i, ch in enumerate(LAUNCH):
        q = _aimed_q(sim, q_park, ch)
        assert _in_limits(q), f"aim at {ch!r} leaves the joint limits"
        sim.teleport(q)
        assert sim.reachable_mask()[names.index(ch)] is True
        res = sim.press(t)
        assert res.accepted and res.reason is Reason.ACCEPTED and res.key.name == ch
        ev = sim.events[-1]
        assert ev["kind"] == "press" and ev["accepted"] is True
        assert ev["reason"] == "ACCEPTED" and ev["key"] == ch
        assert ev["t"] == pytest.approx(t)
        assert len(ev["board_xy"]) == 2
        assert sim.typed == LAUNCH[: i + 1]
        t += 0.5

    # A second press within the debounce window is rejected, typed unchanged.
    t_second = t - 0.5 + 0.05
    res = sim.press(t_second)
    assert not res.accepted and res.reason is Reason.DEBOUNCE
    assert sim.events[-1]["reason"] == "DEBOUNCE" and sim.events[-1]["accepted"] is False
    assert sim.typed == LAUNCH
    assert sim.snapshot(t_second)["stylus"]["would_accept"] == "DEBOUNCE"

    result = sim.done(10.0)
    assert sim.finished and sim.result is result
    assert result.target == LAUNCH and result.typed == LAUNCH
    assert result.exact_match is True and result.edit_distance == 0
    assert result.presses_attempted == len(LAUNCH) + 1
    assert result.presses_accepted == len(LAUNCH)
    assert result.rejections == {"DEBOUNCE": 1}
    assert result.elapsed == pytest.approx(10.0)

    # Presses after done: rejected with the plain reason string 'FINISHED'.
    res = sim.press(11.0)
    assert res.accepted is False and res.reason == FINISHED_REASON
    assert not isinstance(res.reason, Reason)
    assert sim.events[-1]["reason"] == FINISHED_REASON and sim.events[-1]["kind"] == "press"
    assert sim.result.presses_attempted == len(LAUNCH) + 1  # nothing recorded
    snap = sim.snapshot(11.0)
    assert snap["episode"]["status"] == "finished"
    assert snap["result"]["typed"] == LAUNCH and snap["result"]["target"] == LAUNCH
    assert snap["stylus"]["would_accept"] == FINISHED_REASON
    with pytest.raises(RuntimeError):
        sim.done(12.0)


def test_press_from_home_is_rejected_and_tallied(make_sim):
    sim = make_sim(seed=2)
    res = sim.press(0.5)
    assert res.accepted is False and res.reason in set(Reason) - {Reason.ACCEPTED}
    assert sim.typed == ""
    assert sim.events[-1]["reason"] == res.reason.value
    result = sim.done(1.0)
    assert result.typed == "" and result.exact_match is False
    assert result.presses_attempted == 1 and result.presses_accepted == 0
    assert sum(result.rejections.values()) == 1
    assert result.edit_distance == len(LAUNCH)


def test_done_with_no_presses_scores_empty(make_sim):
    sim = make_sim()
    result = sim.done(3.5)
    assert result.typed == "" and result.presses_attempted == 0
    assert result.elapsed == pytest.approx(3.5)
    assert sim.snapshot(3.5)["result"] == {
        "target": LAUNCH, "typed": "", "exact_match": False, "edit_distance": len(LAUNCH),
        "presses_attempted": 0, "presses_accepted": 0, "rejections": {}, "elapsed": 3.5,
    }


# --------------------------------------------------------------------------- #
# 5. snapshot / static_info schema
# --------------------------------------------------------------------------- #


def test_static_info_schema(make_sim, geometry, keymap, texture):
    sim = make_sim(seed=7)
    info = sim.static_info()
    json.dumps(info)  # must be serialisable as-is
    assert set(info) == STATIC_KEYS
    assert info["joint_names"] == list(JOINT_NAMES)
    for name in ("q_min_deg", "q_max_deg", "v_max_deg_s"):
        assert len(info[name]) == NJ
    assert info["q_min_deg"] == pytest.approx((ARM.q_min / DEG).tolist())
    assert info["q_max_deg"] == pytest.approx((ARM.q_max / DEG).tolist())
    assert set(info["links"]) == STATIC_LINKS_KEYS
    assert set(info["stylus"]) == STATIC_STYLUS_KEYS
    assert set(info["press"]) == STATIC_PRESS_KEYS
    assert info["press"]["max_incidence_deg"] == pytest.approx(CFG.press.max_incidence / DEG)
    assert set(info["camera"]) == STATIC_CAMERA_KEYS
    assert info["camera"]["cx"] == CFG.camera.cx and info["camera"]["width"] == CFG.camera.width
    assert set(info["panel"]) == STATIC_PANEL_KEYS

    pub = geometry.public_dict()
    assert info["panel"] == {"w": pub["panel_w"], "h": pub["panel_h"]}
    assert len(info["markers"]) == 4
    for m in info["markers"]:
        assert set(m) == STATIC_MARKER_KEYS
        assert m["center"] == pub["marker_centers"][m["id"]]
        assert m["size"] == pub["marker_size"]
    assert [m["id"] for m in info["markers"]] == [0, 1, 2, 3]

    assert len(info["keys"]) == N_KEYS == len(keymap)
    assert [k["name"] for k in info["keys"]] == [k.name for k in keymap]
    for entry, key in zip(info["keys"], keymap):
        assert set(entry) == STATIC_KEY_KEYS
        assert entry["kind"] == key.kind
        assert entry["rect"] == [key.x0, key.y0, key.x1, key.y1]

    assert set(info["texture"]) == STATIC_TEXTURE_KEYS
    assert info["texture"]["w"] == texture.shape[1] and info["texture"]["h"] == texture.shape[0]
    assert info["texture"]["px_per_m"] == pytest.approx(DEFAULT_PX_PER_M)
    assert info["teleop"] is False and sim.static_info(teleop=True)["teleop"] is True
    assert info["launch_key"] == LAUNCH and info["seed"] == 7
    # Nothing about the keyboard placement itself is exported.
    assert not {"kb_origin", "kb_w", "kb_h", "key_area_origin", "key_inset"} & set(info)


def test_snapshot_schema(make_sim, parked, keymap):
    seed, q_park = parked
    sim = make_sim(seed=seed, t0=2.0)
    sim.teleport(_aimed_q(sim, q_park, LAUNCH[0]))
    sim.command(["head_pan"], [0.0], 2.5)
    sim.press(3.0)
    snap = sim.snapshot(3.25)
    json.dumps(snap)

    assert set(snap) == STATE_KEYS
    assert snap["t"] == pytest.approx(1.25)
    assert set(snap["episode"]) == STATE_EPISODE_KEYS
    assert snap["episode"] == {"status": "running", "seed": seed, "launch_key": LAUNCH}
    assert len(snap["q_deg"]) == NJ and len(snap["qd_deg_s"]) == NJ
    q_min_deg, q_max_deg = ARM.q_min / DEG, ARM.q_max / DEG
    for j, deg in enumerate(snap["q_deg"]):
        assert q_min_deg[j] - 1e-9 <= deg <= q_max_deg[j] + 1e-9
    assert snap["q_deg"] == pytest.approx((sim.q / DEG).tolist())
    assert snap["cmd_age_s"] == pytest.approx(0.75)
    assert snap["watchdog_tripped"] is True

    assert set(snap["arm"]) == STATE_ARM_KEYS
    pose = sim.arm_pose()
    for name, vec in (("shoulder", pose.p_shoulder), ("elbow", pose.p_elbow),
                      ("head", pose.p_head), ("aim", pose.aim), ("cam_axis", pose.R_head[:, 0])):
        assert snap["arm"][name] == pytest.approx(vec.tolist())
    assert set(snap["board"]) == STATE_BOARD_KEYS
    assert np.asarray(snap["board"]["corners"]).shape == (4, 3)
    assert len(snap["board"]["center"]) == 3

    assert set(snap["stylus"]) == STATE_STYLUS_KEYS
    assert snap["stylus"]["key"] == LAUNCH[0]
    assert snap["stylus"]["would_accept"] in LADDER_REASONS
    assert ARM.stylus_min <= snap["stylus"]["range"] <= ARM.stylus_max
    assert 0.0 <= snap["stylus"]["incidence_deg"] <= CFG.press.max_incidence / DEG
    key = keymap.by_name[LAUNCH[0]]
    bx, by = snap["stylus"]["board_xy"]
    assert key.x0 <= bx <= key.x1 and key.y0 <= by <= key.y1
    u, v = snap["reticle_px"]
    assert 0.0 <= u <= CFG.camera.width - 1 and 0.0 <= v <= CFG.camera.height - 1

    assert len(snap["reachable"]) == N_KEYS
    assert all(isinstance(b, bool) for b in snap["reachable"])
    assert snap["typed"] == LAUNCH[0]
    assert snap["result"] is None
    kinds = {e["kind"] for e in snap["events"]}
    assert kinds == {"info", "press"}
    for ev in snap["events"]:
        assert set(ev) == (EVENT_PRESS_KEYS if ev["kind"] == "press" else EVENT_INFO_KEYS)
    assert snap["events"][-1]["t"] == pytest.approx(1.0)  # relative to t0

    result_snap = sim.done(4.0) and sim.snapshot(4.0)
    assert set(result_snap["result"]) == RESULT_KEYS
    assert result_snap["episode"]["status"] == "finished"


def test_snapshot_stylus_is_null_when_ray_misses(make_sim):
    sim = make_sim()
    # Base yaw at its limit points the whole arm away from the panel (which is
    # always sampled in front of the arm), so the aim ray never meets its face.
    q = np.array(ARM.q_home)
    q[0] = ARM.q_max[0]
    sim.teleport(q)
    snap = sim.snapshot(0.0)
    assert snap["stylus"] is None and snap["reticle_px"] is None
    assert not any(snap["reachable"])
    assert sim.press(0.0).reason is Reason.NO_INTERSECT


def test_reachable_mask_matches_ladder(make_sim, parked):
    seed, q_park = parked
    sim = make_sim(seed=seed)
    sim.teleport(q_park)
    mask = sim.reachable_mask()
    assert len(mask) == N_KEYS
    names = [k.name for k in sim.keymap]
    alnum = {k.name for k in sim.keymap.alnum()}
    assert all(mask[names.index(n)] for n in alnum)
    geometric_failures = {Reason.NO_INTERSECT, Reason.OUT_OF_REACH, Reason.TOO_CLOSE,
                          Reason.GLANCING, Reason.NO_KEY}
    for i, key in enumerate(sim.keymap):
        q = _aimed_q(sim, q_park, key.name)
        if not _in_limits(q):
            assert mask[i] is False, key.name
            continue
        sim.teleport(q)
        res = sim.press(float(i))
        assert mask[i] == (res.reason not in geometric_failures), key.name
        if mask[i]:
            assert res.key is not None and res.key.name == key.name


def test_reachable_is_cached_at_five_hertz(make_sim):
    sim = make_sim()
    calls = []
    original = sim.reachable_mask
    sim.reachable_mask = lambda: (calls.append(1), original())[1]  # type: ignore[method-assign]
    sim.snapshot(1.0)
    sim.snapshot(1.0 + 0.5 * REACHABLE_PERIOD)
    sim.snapshot(1.0 + 0.9 * REACHABLE_PERIOD)
    assert len(calls) == 1
    sim.snapshot(1.0 + 1.5 * REACHABLE_PERIOD)
    assert len(calls) == 2
    sim.snapshot(0.0)  # clock went backwards (reset): recompute
    assert len(calls) == 3


def test_events_keep_the_last_fifty(make_sim):
    sim = make_sim()
    for i in range(60):
        sim.press(float(i))
    assert len(sim.events) == 50 and len(sim.snapshot(60.0)["events"]) == 50
    assert sim.events[-1]["t"] == pytest.approx(59.0)
    assert all(e["kind"] == "press" for e in sim.events)


# --------------------------------------------------------------------------- #
# 6. TF chain
# --------------------------------------------------------------------------- #


def test_tf_chain_reproduces_forward_kinematics(make_sim):
    sim = make_sim()
    static = sim.tf_chain(static=True)
    frames = {c for _, c, _, _ in static} | {p for p, _, _, _ in static}
    dyn0 = sim.tf_chain(static=False)
    frames |= {c for _, c, _, _ in dyn0} | {p for p, _, _, _ in dyn0}
    assert frames == {
        "world", "base_link", "shoulder_link", "upper_arm_link", "forearm_link",
        "camera_link", "camera_optical_frame", "head_link", "stylus_tip",
    }
    assert not any("board" in f.lower() for f in frames)
    static_children = {c for _, c, _, _ in static}
    assert static_children == {"base_link", "camera_link", "camera_optical_frame", "stylus_tip"}
    assert {c for _, c, _, _ in dyn0} == {"shoulder_link", "upper_arm_link", "forearm_link", "head_link"}
    for _, _, R, t in static + dyn0:
        assert R.shape == (3, 3) and t.shape == (3,) and R.dtype == np.float64
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and np.linalg.det(R) > 0

    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform(ARM.q_min, ARM.q_max)
        sim.teleport(q)
        chain = static + sim.tf_chain(static=False)
        pose = forward_kinematics(ARM, q)
        R_cam, t_cam = _compose_to_world(chain, FRAME_CAMERA_OPTICAL)
        assert np.allclose(R_cam, pose.R_cam, atol=1e-9)
        assert np.allclose(t_cam, pose.t_cam, atol=1e-9)
        R_head, t_head = _compose_to_world(chain, FRAME_HEAD)
        assert np.allclose(R_head[:, 0], pose.aim, atol=1e-9)
        assert np.allclose(t_head, pose.p_head, atol=1e-9)
        _, t_tip = _compose_to_world(chain, FRAME_STYLUS_TIP)
        assert np.allclose(t_tip, pose.p_head + ARM.stylus_max * pose.aim, atol=1e-9)


def test_dynamic_tf_follows_the_plant(make_sim):
    sim = make_sim()
    before = _compose_to_world(sim.tf_chain(True) + sim.tf_chain(False), FRAME_HEAD)[1]
    _drive(sim, 0.4, ["shoulder_pitch"], [-0.5])
    after = _compose_to_world(sim.tf_chain(True) + sim.tf_chain(False), FRAME_HEAD)[1]
    assert not np.allclose(before, after)
    assert np.allclose(after, sim.arm_pose().p_head, atol=1e-9)


# --------------------------------------------------------------------------- #
# 7. render / camera_info
# --------------------------------------------------------------------------- #


def test_render_detects_all_markers_at_home(make_sim, geometry):
    sim = make_sim(seed=0)
    for seed in range(5):
        sim.reset(seed=seed)
        assert np.array_equal(sim.q, ARM.q_home)
        frame = sim.render()
        assert frame.shape == (CFG.camera.height, CFG.camera.width, 3)
        assert frame.dtype == np.uint8
        det = detect_markers(frame, geometry.marker_dict)
        assert set(det) == MARKER_IDS, f"seed {seed}: detected {sorted(det)}"
        pose = sim.arm_pose()
        truth = marker_corners_world(geometry, sim.board_pose)
        for mid, corners in det.items():
            uv, z = project_points(CFG.camera, pose.R_cam, pose.t_cam, truth[mid])
            assert np.all(z > 0)
            assert np.max(np.linalg.norm(corners - uv, axis=1)) < 2.0, f"seed {seed} marker {mid}"


def test_render_into_caller_buffer(make_sim):
    sim = make_sim()
    out = np.zeros((CFG.camera.height, CFG.camera.width, 3), np.uint8)
    frame = sim.render(out=out)
    assert frame is out and frame.any()
    assert np.array_equal(frame, sim.render())


def test_camera_info(make_sim):
    info = make_sim().camera_info()
    K = CameraConfig().K
    assert info["K"] == K.reshape(-1).tolist()
    assert info["D"] == [0.0] * 5 and info["distortion_model"] == "plumb_bob"
    assert info["R"] == np.eye(3).reshape(-1).tolist()
    assert info["P"] == np.hstack([K, np.zeros((3, 1))]).reshape(-1).tolist()
    assert info["width"] == CFG.camera.width and info["height"] == CFG.camera.height
    assert info["frame_id"] == "camera_optical_frame"
    json.dumps(info)


# --------------------------------------------------------------------------- #
# 8. build_default / private dir resolution
# --------------------------------------------------------------------------- #


def test_build_default_with_photo_and_synthetic_fallback(private_dir, tmp_path):
    photo = default_texture_photo()
    if photo is None:
        pytest.skip("no shipped keyboard_grid.png next to the package")
    sim = build_default(CFG, private_dir, "mars", 3, photo)
    assert sim.launch_key == "MARS" and sim.seed == 3
    w, h = texture_size(sim.geometry, DEFAULT_PX_PER_M)
    assert sim.texture.shape == (h, w, 3) and sim.texture.dtype == np.uint8
    assert len(sim.keymap) == N_KEYS
    assert set(detect_markers(sim.render(), sim.geometry.marker_dict)) == MARKER_IDS

    missing = tmp_path / "no_such_photo.png"
    synth = build_default(CFG, private_dir, "MARS", 3, missing)
    assert synth.texture.shape == sim.texture.shape
    assert np.array_equal(synth.texture, build_texture(synth.geometry, synth.keymap, DEFAULT_PX_PER_M))
    assert not np.array_equal(synth.texture, sim.texture)
    assert np.array_equal(build_default(CFG, private_dir, "MARS", 3, None).texture, synth.texture)
    # Same seed, same board, regardless of texture.
    assert synth.snapshot(0.0)["board"] == sim.snapshot(0.0)["board"]


def test_build_default_requires_the_private_geometry(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_default(CFG, tmp_path, "MARS", 1, None)


def test_resolve_private_dir_parameter_rule(monkeypatch, tmp_path):
    monkeypatch.setenv(PRIVATE_DIR_ENV, str(tmp_path / "from_env"))
    assert resolve_private_dir("/explicit/dir") == Path("/explicit/dir")
    assert resolve_private_dir("") == tmp_path / "from_env"
    monkeypatch.delenv(PRIVATE_DIR_ENV)
    monkeypatch.setenv(PRIVATE_DIR_ENV, "")
    assert resolve_private_dir("") == _REPO / "private"  # running from the source tree
    monkeypatch.delenv(PRIVATE_DIR_ENV)
    assert resolve_private_dir("") == _REPO / "private"


# --------------------------------------------------------------------------- #
# Plant noise magnitudes (INTERFACES.md s.13: configuration data, not code)
# --------------------------------------------------------------------------- #


def test_plant_noise_defaults_are_zero_and_come_from_configuration(tmp_path):
    """``PlantConfig`` ships with 0.0 noise: the magnitudes are configuration
    data, so a noiseless plant is the fallback when none is configured."""
    from autotype_sim.core.config import PlantConfig

    assert PlantConfig().noise_rel == 0.0 and PlantConfig().noise_abs == 0.0
    cfg = SimConfig()

    # No file at all -> unchanged config, no crash.
    assert load_plant_overrides(tmp_path) == {}
    assert apply_private_plant(cfg, tmp_path) is cfg

    # A geometry file with no plant section -> still unchanged.
    (tmp_path / "board_geometry.yaml").write_text("panel_w: 0.4\n", encoding="utf-8")
    assert load_plant_overrides(tmp_path) == {}
    assert apply_private_plant(cfg, tmp_path) is cfg

    # With the section -> those magnitudes, and nothing else touched.
    (tmp_path / "board_geometry.yaml").write_text(
        "panel_w: 0.4\nplant:\n  noise_rel: 0.07\n  noise_abs: 0.003\n  rate: 999\n",
        encoding="utf-8",
    )
    assert load_plant_overrides(tmp_path) == {"noise_rel": 0.07, "noise_abs": 0.003}
    out = apply_private_plant(cfg, tmp_path)
    assert out.plant.noise_rel == 0.07 and out.plant.noise_abs == 0.003
    assert out.plant.rate == cfg.plant.rate  # `rate` is not an override field
    assert out.arm == cfg.arm and out.camera == cfg.camera and out.press == cfg.press

    # A garbled section is a loud configuration error naming the file and key,
    # not a silent fall back to a noiseless plant.
    (tmp_path / "board_geometry.yaml").write_text(
        "plant:\n  noise_rel: not-a-number\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"plant\.noise_rel must be a number"):
        load_plant_overrides(tmp_path)
    # A negative magnitude is caught by PlantConfig's own validation.
    (tmp_path / "board_geometry.yaml").write_text("plant:\n  noise_abs: -1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="noise_abs must be >= 0"):
        apply_private_plant(cfg, tmp_path)


@pytest.mark.private
def test_private_yaml_supplies_a_noisy_plant(private_dir):
    """With private/ present the shipped defaults are replaced by real noise."""
    over = load_plant_overrides(private_dir)
    assert set(over) == {"noise_rel", "noise_abs"}
    assert over["noise_rel"] > 0.0 and over["noise_abs"] > 0.0
    assert apply_private_plant(SimConfig(), private_dir).plant.noise_rel == over["noise_rel"]
