"""Tests for core/press.py against the design spec section 7.

One fixture throughout: a nominal (unperturbed) board with its panel centre
at world (1.00, 0, 0.36) -- the near face of the section-6 sampling box --
and the arm parked with its head 0.25 m from that centre: 0.24 m in front
and 0.07 m above it, the way the solvability sweep's best parked poses sit
(the elbow-only-down arm pitches its forearm ~40 deg down at typing
distance, so the head parks above the keys and tilts down onto them). The
park joint vector comes from a closed-form elbow-down IK and is checked to
lie inside the joint limits; head pan/tilt for each aim comes from
``aim_inverse``. Everything is deterministic: the only randomness is a
seeded Generator in the consistency sweep.

Geometry and key map come from the session fixtures (conftest.py): the
board_geometry.yaml in private/ when available, else ``DEFAULT_GEOMETRY``.
Nothing here depends on a particular keyboard placement -- the ``rig``
fixture bundles geometry, key map and the nominal board pose.

The ladder is pure geometry over ``ArmPose.p_head`` / ``ArmPose.aim`` and
never consults link lengths, so the TOO_CLOSE case -- which the 1.0 m arm
cannot physically produce against a panel 1 m away -- slides the parked
head forward along its own aim ray instead of re-solving IK.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from autotype_sim.core.board import BoardGeometry, BoardPose, board_pose_from_center
from autotype_sim.core.config import DEG, NJ, ArmConfig, PressConfig
from autotype_sim.core.keymap import KeyMap
from autotype_sim.core.kinematics import ArmPose, aim_inverse, forward_kinematics
from autotype_sim.core.press import (
    PARALLEL_EPS,
    UNIT_AIM_TOL,
    PressResult,
    Reason,
    evaluate_press,
    ray_board_intersection,
)

ARM = ArmConfig()
PRESS = PressConfig()
L0, L1, L2 = ARM.base_height, ARM.upper_arm, ARM.forearm
QD_STILL = np.zeros(NJ)
T_NOW = 10.0

# Nominal board: X_B = -Y_W, Y_B = -Z_W, Z_B = +X_W; panel centre 1.00 m out.
PANEL_CENTRE = np.array([1.00, 0.0, 0.36])
HEAD_PARK = PANEL_CENTRE + np.array([-0.24, 0.0, 0.07])  # |offset| = 0.25 exactly


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _q_for_head(p_head: np.ndarray) -> np.ndarray:
    """Closed-form IK: base yaw + planar 2R, elbow-down (elbow_pitch <= 0),
    pan = tilt = 0. Raises if `p_head` is beyond L1 + L2 from the shoulder."""
    x, y, z = np.asarray(p_head, dtype=float)
    th0 = np.arctan2(y, x)
    rho, dz = np.hypot(x, y), z - L0
    c2 = (rho**2 + dz**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    if abs(c2) > 1.0:
        raise ValueError(f"head position {p_head} is unreachable (cos elbow = {c2:.3f})")
    th2 = -np.arccos(c2)
    th1 = np.arctan2(dz, rho) - np.arctan2(L2 * np.sin(th2), L1 + L2 * np.cos(th2))
    return np.array([th0, th1, th2, 0.0, 0.0])


def _aimed(q: np.ndarray, target: np.ndarray) -> tuple[ArmPose, np.ndarray]:
    """Set head pan/tilt of `q` via aim_inverse so the stylus points at `target`."""
    pose = forward_kinematics(ARM, q)
    pan, tilt, _ = aim_inverse(pose.R_head, pose.p_head, target)
    q2 = np.array(q, dtype=float)
    q2[3], q2[4] = pan, tilt
    return forward_kinematics(ARM, q2), q2


def _slide(pose: ArmPose, dist: float) -> ArmPose:
    """Translate the head `dist` metres along its own aim ray (aim unchanged)."""
    p = pose.p_head + dist * pose.aim
    return replace(pose, p_head=p, t_cam=p.copy())


def _in_limits(q: np.ndarray) -> bool:
    return bool(np.all(q >= ARM.q_min) and np.all(q <= ARM.q_max))


Q_PARK = _q_for_head(HEAD_PARK)
PARKED = forward_kinematics(ARM, Q_PARK)
AIM_KEYS = ("A", "P", "5", "SPACE", "BACKSPACE")


@dataclass(frozen=True)
class Rig:
    """Geometry, key map and the nominal board pose the ladder is tested on."""

    geom: BoardGeometry
    km: KeyMap
    board: BoardPose

    def board_point_world(self, x: float, y: float) -> np.ndarray:
        return self.board.to_world(np.array([x, y, 0.0]))

    def key_world(self, name: str) -> np.ndarray:
        cx, cy = KeyMap.center(self.km.by_name[name])
        return self.board_point_world(cx, cy)

    def press(
        self,
        pose: ArmPose,
        qd: np.ndarray = QD_STILL,
        t_now: float = T_NOW,
        t_last: float | None = None,
    ) -> PressResult:
        return evaluate_press(pose, qd, self.board, self.km, ARM, PRESS, t_now, t_last)

    def glancing_pose(self) -> ArmPose:
        """Head parked far to the arm's left, low and close to the panel, stabbing
        across it at the RIGHT-arrow key: incidence ~61 deg at range ~0.33 m."""
        q_side = _q_for_head([0.84, 0.12, 0.30])
        pose, _ = _aimed(q_side, self.key_world("RIGHT"))
        return pose


@pytest.fixture(scope="module")
def rig(geometry, keymap) -> Rig:
    return Rig(geometry, keymap, board_pose_from_center(geometry, PANEL_CENTRE, 0.0, 0.0, 0.0))


# --------------------------------------------------------------------------- #
# Fixture sanity
# --------------------------------------------------------------------------- #


def test_fixture_board_is_nominal_and_faces_the_arm(rig):
    np.testing.assert_allclose(rig.board.t, [1.00, 0.20, 0.4475], atol=1e-15)
    np.testing.assert_allclose(rig.board.R, [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], atol=0)
    np.testing.assert_allclose(rig.board.normal_toward_arm, [-1.0, 0.0, 0.0], atol=0)
    np.testing.assert_allclose(rig.board.center_world(rig.geom), PANEL_CENTRE, atol=1e-15)


def test_fixture_park_pose_is_legal_and_puts_head_in_front_of_centre(rig):
    np.testing.assert_allclose(PARKED.p_head, HEAD_PARK, atol=1e-12)
    assert _in_limits(Q_PARK)
    assert Q_PARK[2] <= 0.0  # elbow-down branch, as the elbow limits require
    assert abs(np.linalg.norm(PARKED.p_head - PANEL_CENTRE) - 0.25) < 1e-12
    for name in AIM_KEYS:
        _, q = _aimed(Q_PARK, rig.key_world(name))
        assert _in_limits(q), f"aim at {name} leaves the joint limits: {q / DEG}"


# --------------------------------------------------------------------------- #
# ray_board_intersection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", AIM_KEYS)
def test_ray_intersection_lands_on_the_aimed_key_centre(rig, name):
    target = rig.key_world(name)
    pose, _ = _aimed(Q_PARK, target)
    _, _, expected_range = aim_inverse(PARKED.R_head, PARKED.p_head, target)

    hit = ray_board_intersection(pose, rig.board)
    assert hit is not None
    t, point, incidence = hit
    assert isinstance(t, float) and isinstance(incidence, float)
    assert abs(t - expected_range) < 1e-12
    np.testing.assert_allclose(point, target, atol=1e-12)
    np.testing.assert_allclose(point, pose.p_head + t * pose.aim, atol=1e-15)
    # On the plane, and in the board frame z == 0.
    assert abs((point - rig.board.t) @ rig.board.normal_toward_arm) < 1e-12
    assert abs(rig.board.to_board(point)[2]) < 1e-12
    # Incidence is the angle between the aim and the inward normal +Z_B.
    assert abs(incidence - np.arccos(pose.aim @ rig.board.R[:, 2])) < 1e-12
    assert 0.0 <= incidence < np.pi / 2


def test_ray_intersection_none_when_parallel_away_or_behind(rig):
    away, _ = _aimed(Q_PARK, PARKED.p_head - [0.3, 0.0, 0.0])  # back toward the base
    up, _ = _aimed(Q_PARK, PARKED.p_head + [0.0, 0.0, 1.0])  # parallel to the face
    left, _ = _aimed(Q_PARK, PARKED.p_head + [0.0, 1.0, 0.0])  # parallel to the face
    for pose in (away, up, left):
        assert ray_board_intersection(pose, rig.board) is None

    # Head on the far side of the panel, aiming further away: plane is behind.
    behind = _slide(_aimed(Q_PARK, PANEL_CENTRE)[0], 0.40)
    assert behind.p_head[0] > rig.board.t[0]
    assert ray_board_intersection(behind, rig.board) is None

    # Just inside the parallel tolerance counts as parallel.
    d = np.array([PARALLEL_EPS / 2.0, 0.0, 1.0])
    grazing = replace(PARKED, aim=d / np.linalg.norm(d))
    assert ray_board_intersection(grazing, rig.board) is None


# --------------------------------------------------------------------------- #
# 1. ACCEPTED at key centres
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", AIM_KEYS)
def test_aim_at_key_centre_is_accepted(rig, name):
    key = rig.km.by_name[name]
    pose, _ = _aimed(Q_PARK, rig.key_world(name))
    res = rig.press(pose)

    assert res.accepted is True
    assert res.reason is Reason.ACCEPTED
    assert res.key is key
    np.testing.assert_allclose(res.board_xy, KeyMap.center(key), atol=1e-9)
    assert ARM.stylus_min <= res.range <= ARM.stylus_max
    assert 0.0 <= res.incidence < PRESS.max_incidence
    assert isinstance(res.board_xy, tuple) and len(res.board_xy) == 2
    assert all(isinstance(v, float) for v in (*res.board_xy, res.range, res.incidence))


# --------------------------------------------------------------------------- #
# 2. NO_INTERSECT
# --------------------------------------------------------------------------- #


def _assert_no_intersect(res: PressResult) -> None:
    assert res.accepted is False
    assert res.reason is Reason.NO_INTERSECT
    assert res.key is None and res.board_xy is None
    assert res.range is None and res.incidence is None


def test_aim_away_from_panel_is_no_intersect(rig):
    # Stylus pointing back toward the base: d . n > 0.
    away, _ = _aimed(Q_PARK, PARKED.p_head - [0.3, 0.0, 0.0])
    _assert_no_intersect(rig.press(away))

    # "Tilt up 80 deg": with the forearm raised 15 deg the aim leaves at
    # 95 deg elevation -- past vertical, away from the face.
    q = np.array([0.0, 15.0 * DEG, 0.0, 0.0, 80.0 * DEG])
    tilted = forward_kinematics(ARM, q)
    assert tilted.p_head[0] < rig.board.t[0]  # head is still in front of the panel
    assert tilted.aim @ rig.board.normal_toward_arm > 0.0
    _assert_no_intersect(rig.press(tilted))


def test_aim_parallel_to_panel_is_no_intersect(rig):
    up, _ = _aimed(Q_PARK, PARKED.p_head + [0.0, 0.0, 1.0])
    _assert_no_intersect(rig.press(up))

    q = np.array(Q_PARK)
    q[3] = 90.0 * DEG  # along Y_H == world +Y at zero base yaw
    sideways = forward_kinematics(ARM, q)
    assert abs(sideways.aim @ rig.board.normal_toward_arm) < 1e-15
    _assert_no_intersect(rig.press(sideways))


def test_steep_tilt_from_park_still_meets_the_infinite_plane(rig):
    # The ladder intersects the plane, not the finite panel: from the parked
    # pose (forearm ~41 deg down) an 80 deg tilt still has a forward component
    # and meets the plane 0.31 m away, 0.18 m above the panel's top edge --
    # within reach, so the ladder gets as far as registration: NO_KEY.
    q = np.array(Q_PARK)
    q[4] = 80.0 * DEG
    pose = forward_kinematics(ARM, q)
    assert ray_board_intersection(pose, rig.board) is not None
    res = rig.press(pose)
    assert res.reason is Reason.NO_KEY
    assert res.range < ARM.stylus_max
    assert res.board_xy[1] < 0.0  # above the panel (board y is down)


# --------------------------------------------------------------------------- #
# 3. OUT_OF_REACH / TOO_CLOSE
# --------------------------------------------------------------------------- #


def test_head_moved_back_is_out_of_reach(rig):
    target = rig.key_world("A")
    q_back = _q_for_head(HEAD_PARK - [0.15, 0.0, 0.0])
    pose, _ = _aimed(q_back, target)
    _, _, expected_range = aim_inverse(pose.R_head, pose.p_head, target)
    assert expected_range > ARM.stylus_max

    res = rig.press(pose)
    assert res.accepted is False
    assert res.reason is Reason.OUT_OF_REACH
    assert abs(res.range - expected_range) < 1e-12
    assert res.incidence is not None and res.incidence < PRESS.max_incidence
    assert res.key is None and res.board_xy is None


def test_head_moved_forward_is_too_close(rig):
    pose, _ = _aimed(Q_PARK, rig.key_world("A"))
    t0, _, _ = ray_board_intersection(pose, rig.board)
    close = _slide(pose, t0 - 0.03)

    res = rig.press(close)
    assert res.accepted is False
    assert res.reason is Reason.TOO_CLOSE
    assert abs(res.range - 0.03) < 1e-12
    assert res.range < ARM.stylus_min
    assert res.incidence is not None
    assert res.key is None and res.board_xy is None


def test_reach_limits_are_inclusive(rig):
    pose, _ = _aimed(Q_PARK, rig.key_world("A"))
    t0, _, _ = ray_board_intersection(pose, rig.board)
    eps = 1e-6
    assert rig.press(_slide(pose, t0 - (ARM.stylus_max - eps))).reason is Reason.ACCEPTED
    assert rig.press(_slide(pose, t0 - (ARM.stylus_max + eps))).reason is Reason.OUT_OF_REACH
    assert rig.press(_slide(pose, t0 - (ARM.stylus_min + eps))).reason is Reason.ACCEPTED
    assert rig.press(_slide(pose, t0 - (ARM.stylus_min - eps))).reason is Reason.TOO_CLOSE


# --------------------------------------------------------------------------- #
# 4. GLANCING
# --------------------------------------------------------------------------- #


def test_glancing_stroke_is_rejected_after_reach_passes(rig):
    pose = rig.glancing_pose()
    t, _, incidence = ray_board_intersection(pose, rig.board)
    # Geometry chosen so reach passes first and only the angle fails.
    assert ARM.stylus_min < t < ARM.stylus_max
    assert incidence > PRESS.max_incidence

    res = rig.press(pose)
    assert res.accepted is False
    assert res.reason is Reason.GLANCING
    assert abs(res.range - t) < 1e-15
    assert abs(res.incidence - incidence) < 1e-15
    assert res.key is None and res.board_xy is None


def test_glancing_threshold_is_strict(rig):
    pose, _ = _aimed(Q_PARK, rig.key_world("A"))
    t, _, incidence = ray_board_intersection(pose, rig.board)
    assert rig.press(pose).reason is Reason.ACCEPTED
    tight = PressConfig(max_incidence=incidence)  # equal is not "greater"
    assert evaluate_press(pose, QD_STILL, rig.board, rig.km, ARM, tight, T_NOW, None).accepted
    tighter = PressConfig(max_incidence=incidence - 1e-9)
    res = evaluate_press(pose, QD_STILL, rig.board, rig.km, ARM, tighter, T_NOW, None)
    assert res.reason is Reason.GLANCING


# --------------------------------------------------------------------------- #
# 5. NO_KEY
# --------------------------------------------------------------------------- #


def test_gap_between_f_and_g_is_no_key(rig):
    F, G = rig.km.by_name["F"], rig.km.by_name["G"]
    assert F.x1 < G.x0  # the inset gap exists
    gx, gy = 0.5 * (F.x1 + G.x0), 0.5 * (F.y0 + F.y1)
    assert rig.km.lookup(gx, gy) is None

    pose, _ = _aimed(Q_PARK, rig.board_point_world(gx, gy))
    res = rig.press(pose)
    assert res.accepted is False
    assert res.reason is Reason.NO_KEY
    assert res.key is None
    np.testing.assert_allclose(res.board_xy, (gx, gy), atol=1e-9)
    assert ARM.stylus_min <= res.range <= ARM.stylus_max
    assert res.incidence < PRESS.max_incidence


@pytest.mark.parametrize(
    "bx, by",
    [
        (0.016, 0.016),  # marker 0 centre
        (0.010, 0.0875),  # left bezel, between panel edge and keyboard plate
        (0.200, 0.010),  # top bezel, above the keyboard plate
        (-0.020, 0.0875),  # on the plane but off the panel entirely
    ],
)
def test_panel_outside_keyboard_is_no_key(rig, bx, by):
    kbx, kby = rig.geom.kb_origin
    inside_plate = kbx <= bx <= kbx + rig.geom.kb_w and kby <= by <= kby + rig.geom.kb_h
    assert not inside_plate
    pose, _ = _aimed(Q_PARK, rig.board_point_world(bx, by))
    res = rig.press(pose)
    assert res.reason is Reason.NO_KEY
    assert res.key is None
    np.testing.assert_allclose(res.board_xy, (bx, by), atol=1e-9)
    assert res.range is not None and res.incidence is not None


# --------------------------------------------------------------------------- #
# 6. MOVING
# --------------------------------------------------------------------------- #


def test_moving_joint_rejects_and_threshold_is_strict(rig):
    pose, _ = _aimed(Q_PARK, rig.key_world("A"))
    key = rig.km.by_name["A"]

    qd = np.zeros(NJ)
    qd[2] = 0.05
    res = rig.press(pose, qd)
    assert res.accepted is False
    assert res.reason is Reason.MOVING
    assert res.key is key  # the key it would have hit is still reported
    np.testing.assert_allclose(res.board_xy, KeyMap.center(key), atol=1e-9)
    assert res.range is not None and res.incidence is not None

    qd[2] = 0.019
    assert rig.press(pose, qd).reason is Reason.ACCEPTED

    qd[2] = PRESS.max_joint_speed  # equal is not "greater"
    assert rig.press(pose, qd).reason is Reason.ACCEPTED

    # Infinity norm: sign and joint index do not matter.
    for j in range(NJ):
        qd = np.zeros(NJ)
        qd[j] = -0.05
        assert rig.press(pose, qd).reason is Reason.MOVING


def test_qd_measured_shape_is_validated(rig):
    pose, _ = _aimed(Q_PARK, rig.key_world("A"))
    with pytest.raises(ValueError):
        rig.press(pose, np.zeros(NJ - 1))
    with pytest.raises(ValueError):
        rig.press(pose, np.zeros((NJ, 1)))


# --------------------------------------------------------------------------- #
# 7. DEBOUNCE
# --------------------------------------------------------------------------- #


def test_debounce_window_and_boundary(rig):
    pose, _ = _aimed(Q_PARK, rig.key_world("A"))
    key = rig.km.by_name["A"]

    res = rig.press(pose, t_now=T_NOW, t_last=T_NOW - 0.10)
    assert res.accepted is False
    assert res.reason is Reason.DEBOUNCE
    assert res.key is key
    np.testing.assert_allclose(res.board_xy, KeyMap.center(key), atol=1e-9)
    assert res.range is not None and res.incidence is not None

    assert rig.press(pose, t_now=T_NOW, t_last=T_NOW - 0.16).reason is Reason.ACCEPTED
    # Exactly the debounce interval is allowed (strict '<').
    assert rig.press(pose, t_now=PRESS.debounce, t_last=0.0).reason is Reason.ACCEPTED
    # No press accepted yet: nothing to debounce against.
    assert rig.press(pose, t_now=0.0, t_last=None).reason is Reason.ACCEPTED
    assert rig.press(pose, t_now=0.0, t_last=-np.inf).reason is Reason.ACCEPTED


# --------------------------------------------------------------------------- #
# 8. Precedence: geometry before dynamics, in ladder order
# --------------------------------------------------------------------------- #


def test_precedence_geometry_before_dynamics(rig):
    moving = np.zeros(NJ)
    moving[1] = 0.05
    bounce = T_NOW - 0.05

    F, G = rig.km.by_name["F"], rig.km.by_name["G"]
    gap, _ = _aimed(Q_PARK, rig.board_point_world(0.5 * (F.x1 + G.x0), 0.5 * (F.y0 + F.y1)))
    assert rig.press(gap, moving, T_NOW, bounce).reason is Reason.NO_KEY

    away, _ = _aimed(Q_PARK, PARKED.p_head - [0.3, 0.0, 0.0])
    assert rig.press(away, moving, T_NOW, bounce).reason is Reason.NO_INTERSECT

    far, _ = _aimed(_q_for_head(HEAD_PARK - [0.15, 0.0, 0.0]), rig.key_world("A"))
    assert rig.press(far, moving, T_NOW, bounce).reason is Reason.OUT_OF_REACH

    on_a, _ = _aimed(Q_PARK, rig.key_world("A"))
    t0, _, _ = ray_board_intersection(on_a, rig.board)
    assert rig.press(_slide(on_a, t0 - 0.03), moving, T_NOW, bounce).reason is Reason.TOO_CLOSE

    assert rig.press(rig.glancing_pose(), moving, T_NOW, bounce).reason is Reason.GLANCING

    # Both dynamic checks failing: MOVING (step 5) beats DEBOUNCE (step 6).
    assert rig.press(on_a, moving, T_NOW, bounce).reason is Reason.MOVING
    assert rig.press(on_a, QD_STILL, T_NOW, bounce).reason is Reason.DEBOUNCE


# --------------------------------------------------------------------------- #
# 9. Input validation: non-finite / non-unit inputs raise instead of falling
#    through the ladder's comparisons (which are all False for NaN)
# --------------------------------------------------------------------------- #


def test_non_finite_qd_measured_raises_instead_of_reading_as_settled(rig):
    # Regression: ``np.max(np.abs(qd)) > max_joint_speed`` is False for NaN,
    # so a NaN joint velocity used to sail through step 5 and be ACCEPTED.
    pose, _ = _aimed(Q_PARK, rig.key_world("A"))
    assert rig.press(pose).reason is Reason.ACCEPTED
    for bad in (np.nan, np.inf, -np.inf):
        for j in range(NJ):
            qd = np.zeros(NJ)
            qd[j] = bad
            with pytest.raises(ValueError, match="qd_measured"):
                rig.press(pose, qd)
    with pytest.raises(ValueError, match="qd_measured"):
        rig.press(pose, np.full(NJ, np.nan))
    # Validation precedes the ladder: it raises even where geometry would
    # already have failed, exactly like the existing shape check.
    away, _ = _aimed(Q_PARK, PARKED.p_head - [0.3, 0.0, 0.0])
    with pytest.raises(ValueError, match="qd_measured"):
        rig.press(away, np.full(NJ, np.nan))


def test_non_finite_times_raise_but_minus_inf_last_accepted_means_never(rig):
    # Regression: ``t_now - t_last < debounce`` is False for a NaN on either
    # side and for t_now = +inf, so step 6 used to be skipped silently.
    pose, _ = _aimed(Q_PARK, rig.key_world("A"))
    for t_now in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="t_now"):
            rig.press(pose, t_now=t_now, t_last=None)
        with pytest.raises(ValueError, match="t_now"):
            rig.press(pose, t_now=t_now, t_last=T_NOW - 0.01)
    for t_last in (np.nan, np.inf):
        with pytest.raises(ValueError, match="t_last_accepted"):
            rig.press(pose, t_now=T_NOW, t_last=t_last)
    # Both spellings of "no press accepted yet" still work (the plant uses
    # -inf for "never commanded" in the same way).
    assert rig.press(pose, t_now=T_NOW, t_last=None).reason is Reason.ACCEPTED
    assert rig.press(pose, t_now=T_NOW, t_last=-np.inf).reason is Reason.ACCEPTED
    # And the debounce arithmetic itself is untouched.
    assert rig.press(pose, t_now=T_NOW, t_last=T_NOW - 0.10).reason is Reason.DEBOUNCE
    assert rig.press(pose, t_now=T_NOW, t_last=T_NOW - 0.16).reason is Reason.ACCEPTED


def test_non_finite_pose_raises_from_both_entry_points(rig):
    # Regression: a NaN aim/p_head used to come back NO_KEY with
    # board_xy=(nan, nan) and range=nan instead of raising.
    pose, _ = _aimed(Q_PARK, rig.key_world("A"))
    bad_poses = [
        replace(pose, aim=np.full(3, np.nan)),
        replace(pose, p_head=np.full(3, np.nan)),
        replace(pose, aim=np.array([np.inf, 0.0, 0.0])),
        replace(pose, p_head=np.array([np.inf, 0.0, 0.0])),
        replace(pose, p_head=np.array([0.75, np.nan, 0.36])),
    ]
    for bad in bad_poses:
        with pytest.raises(ValueError, match="ArmPose"):
            ray_board_intersection(bad, rig.board)
        with pytest.raises(ValueError, match="ArmPose"):
            rig.press(bad)


def test_non_unit_aim_raises_instead_of_rescaling_range(rig):
    # Regression: aim * 2 used to be ACCEPTED with half the range and zero
    # incidence; aim * 0.5 became OUT_OF_REACH at 1.11 rad.
    pose, _ = _aimed(Q_PARK, rig.key_world("A"))
    base = rig.press(pose)
    assert base.reason is Reason.ACCEPTED
    for scale in (2.0, 0.5, 0.0, 1.0 + 10 * UNIT_AIM_TOL, 1.0 - 10 * UNIT_AIM_TOL):
        bad = replace(pose, aim=pose.aim * scale)
        with pytest.raises(ValueError, match="unit"):
            ray_board_intersection(bad, rig.board)
        with pytest.raises(ValueError, match="unit"):
            rig.press(bad)
    # Inside the tolerance (a float32 round trip, say) still evaluates, and
    # the result moves by far less than a micron.
    ok = rig.press(replace(pose, aim=pose.aim * (1.0 + 0.1 * UNIT_AIM_TOL)))
    assert ok.reason is Reason.ACCEPTED and ok.key is base.key
    assert abs(ok.range - base.range) < 1e-6
    f32 = rig.press(replace(pose, aim=pose.aim.astype(np.float32)))
    assert f32.reason is Reason.ACCEPTED and f32.key is base.key
    assert abs(f32.range - base.range) < 1e-6
    # The guard can never fire on a real pose: every legal joint vector's FK
    # aim is unit to machine precision, six orders inside the tolerance.
    rng = np.random.default_rng(1)
    for _ in range(500):
        q = rng.uniform(ARM.q_min, ARM.q_max)
        pose_q = forward_kinematics(ARM, q)
        assert abs(np.linalg.norm(pose_q.aim) - 1.0) < 1e-12
        ray_board_intersection(pose_q, rig.board)  # must not raise


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #


def test_reason_enum_members_and_serialisation():
    assert [r.name for r in Reason] == [
        "ACCEPTED",
        "NO_INTERSECT",
        "OUT_OF_REACH",
        "TOO_CLOSE",
        "GLANCING",
        "NO_KEY",
        "MOVING",
        "DEBOUNCE",
    ]
    for r in Reason:
        assert r.value == r.name
        assert str(r) == r.name
        assert r == r.name  # str mixin: serialises as its own name
        assert isinstance(r, str)


def test_press_result_consistency_is_enforced(rig):
    key = rig.km.by_name["A"]
    with pytest.raises(ValueError):
        PressResult(True, Reason.NO_KEY, None, None, None, None)
    with pytest.raises(ValueError):
        PressResult(False, Reason.ACCEPTED, key, (0.0, 0.0), 0.2, 0.1)
    with pytest.raises(ValueError):
        PressResult(True, Reason.ACCEPTED, None, (0.0, 0.0), 0.2, 0.1)
    ok = PressResult(True, Reason.ACCEPTED, key, (0.0, 0.0), 0.2, 0.1)
    assert ok.accepted and ok.key is key


# --------------------------------------------------------------------------- #
# Seeded sweep: helper and ladder agree, results are self-consistent
# --------------------------------------------------------------------------- #


def test_random_aims_are_self_consistent(rig):
    rng = np.random.default_rng(20260915)
    seen: set[Reason] = set()
    Z_B = rig.board.R[:, 2]
    for _ in range(400):
        q = np.array(Q_PARK)
        q[3] = rng.uniform(ARM.q_min[3], ARM.q_max[3])
        q[4] = rng.uniform(ARM.q_min[4], ARM.q_max[4])
        pose = forward_kinematics(ARM, q)
        # Occasionally back the head off so every rung of the ladder is hit.
        pose = _slide(pose, rng.choice([0.0, 0.0, -0.20, 0.22]))

        hit = ray_board_intersection(pose, rig.board)
        res = rig.press(pose)
        seen.add(res.reason)
        assert isinstance(res.reason, Reason)
        assert res.accepted == (res.reason is Reason.ACCEPTED)

        if hit is None:
            assert res.reason is Reason.NO_INTERSECT
            assert res.range is None and res.incidence is None
            continue

        t, point, incidence = hit
        assert t > 0.0
        assert abs((point - rig.board.t) @ rig.board.normal_toward_arm) < 1e-12
        assert abs(incidence - np.arccos(np.clip(pose.aim @ Z_B, -1.0, 1.0))) < 1e-12
        assert res.range == t and res.incidence == incidence

        if res.board_xy is not None:
            b = rig.board.to_board(point)
            np.testing.assert_allclose(res.board_xy, b[:2], atol=1e-15)
            assert res.key is rig.km.lookup(*res.board_xy)
        if res.key is not None:
            k = res.key
            assert k.x0 <= res.board_xy[0] <= k.x1 and k.y0 <= res.board_xy[1] <= k.y1

    # The sweep exercised every geometric rung reachable within the joint
    # limits. NO_INTERSECT is not one of them: from park the forearm is ~41 deg
    # down, and within the 45 deg pan / 35 deg tilt stops every aim keeps a
    # forward component toward the panel.
    assert {Reason.ACCEPTED, Reason.OUT_OF_REACH, Reason.TOO_CLOSE, Reason.NO_KEY} <= seen
    assert Reason.NO_INTERSECT not in seen
